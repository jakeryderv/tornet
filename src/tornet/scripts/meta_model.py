"""Gradient-boosting meta model: the frozen backbone's pooled features as plain inputs, plus
sample metadata, hand-computed radar rotation metrics and near-storm HRRR environment.

One row per sample is assembled from
- `pooled_dir` (tornet-pool-features): per-scan max/mean-pooled feature vectors; `frames`
  selects which scans (indices, oldest first) so the same lead-time framing as the heads applies
- the cache metadata: range to radar, hour of day and month (group "meta")
- `radar_dir` (tornet-radar-metrics): per-frame rotation metrics for the selected frames and
  their frame-to-frame trends (group "radar")
- `env_csv` (tornet-env-fetch): environment fields with derived shear magnitudes and a
  significant-tornado-parameter-style composite (group "env")

`features` lists the groups used, so the same config with groups added or removed is an
ablation. The model is LightGBM with the authors' sample weights, early stopping on the
validation AUC, and the usual run directory: results.json, predictions.csv and scores.json for
tornet-score, plus feature_importance.csv (gain, with a per-group summary in results.json).

Usage: tornet-meta-model <config.json> [<config.json> ...]
"""

import copy
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from tornet.data import preprocess as pp
from tornet.scripts.test_keras import finish_predictions, score_predictions
from tornet.scripts.train_head import CATEGORY_CODES
from tornet.utils.general import git_revision, make_exp_dir

logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG = {
    "pooled_dir": "pooled_4f",  # relative paths are under $TORNET_ROOT
    "radar_dir": "radar_metrics",
    "env_csv": "env_hrrr/env_hrrr.csv",
    "features": ["embedding", "meta", "radar", "env"],
    "frames": None,  # scans used for embedding and radar groups (indices, oldest first)
    "train_years": [2013, 2015],
    "val_years": [2014],
    "test_years": [2013, 2014, 2015],
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "params": {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 0,
    },
    "num_rounds": 3000,
    "early_stopping": 150,
    "seed": 1234,
    "exp_name": "meta",
    "exp_dir": os.environ.get("EXP_DIR", "experiments"),
}

RADAR_METRICS = ("vrot", "azshear", "velmax", "dbzmax", "valid")
TREND_METRICS = ("vrot", "azshear", "dbzmax")


def resolve(path: str | None) -> str | None:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(os.environ["TORNET_ROOT"], path)


def load_pooled(pooled_dir: str, partition: str, years: list[int], cache: dict):
    key = ("pooled", pooled_dir, partition, tuple(years))
    if key in cache:
        return cache[key]
    xs, metas = [], []
    for year in years:
        xs.append(np.load(os.path.join(pooled_dir, f"{partition}_{year}_pooled.npy")))
        metas.append(pd.read_csv(os.path.join(pooled_dir, f"{partition}_{year}_meta.csv")))
    cache[key] = (np.concatenate(xs), pd.concat(metas, ignore_index=True))
    return cache[key]


def load_csvs(pattern_dir: str, suffix: str, cache: dict) -> pd.DataFrame | None:
    """All <partition>_<year><suffix> tables under a directory, concatenated (None if absent)."""
    key = ("csvs", pattern_dir, suffix)
    if key in cache:
        return cache[key]
    if not pattern_dir or not os.path.isdir(pattern_dir):
        cache[key] = None
        return None
    parts = [
        pd.read_csv(os.path.join(pattern_dir, f))
        for f in sorted(os.listdir(pattern_dir))
        if f.endswith(suffix)
    ]
    cache[key] = pd.concat(parts, ignore_index=True).drop_duplicates("filename") if parts else None
    return cache[key]


def load_csv(path: str | None, cache: dict) -> pd.DataFrame | None:
    key = ("csv", path)
    if key in cache:
        return cache[key]
    cache[key] = (
        pd.read_csv(path).drop_duplicates("filename") if path and os.path.exists(path) else None
    )
    return cache[key]


def embedding_columns(x: np.ndarray, frames: list[int] | None) -> pd.DataFrame:
    """(N, [T,] D) pooled features -> columns emb_f{k}_{j}."""
    if x.ndim == 2:
        x = x[:, None]
    idx = list(range(x.shape[1])) if frames is None else frames
    cols = {f"emb_f{k}_{j}": x[:, k, j].astype(np.float32) for k in idx for j in range(x.shape[2])}
    return pd.DataFrame(cols)


def meta_columns(meta: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(meta["time"], unit="s")
    hour = t.dt.hour + t.dt.minute / 60
    doy = t.dt.dayofyear
    return pd.DataFrame(
        {
            "meta_rng_center_km": meta["rng_center_km"].to_numpy(np.float32),
            "meta_hour_sin": np.sin(2 * np.pi * hour / 24),
            "meta_hour_cos": np.cos(2 * np.pi * hour / 24),
            "meta_doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "meta_doy_cos": np.cos(2 * np.pi * doy / 365.25),
        }
    )


def radar_columns(radar: pd.DataFrame, frames: list[int] | None) -> pd.DataFrame:
    """Per-frame metrics for the selected frames, trends between consecutive selected frames,
    and the static nyquist velocity. Column names are relative to the selection (r{i}) so a
    lead-10 model and a lead-0 model have the same feature names."""
    present = sorted(
        {int(c.split("_")[0][1:]) for c in radar.columns if c.startswith("f") and "_s" in c}
    )
    idx = present if frames is None else frames
    sweeps = sorted({c.split("_")[1] for c in radar.columns if c.startswith("f0_s")})
    out = {"radar_nyquist": radar["nyquist"].to_numpy(np.float32)}
    for i, k in enumerate(idx):
        for s in sweeps:
            for m in RADAR_METRICS:
                out[f"radar_r{i}_{s}_{m}"] = radar[f"f{k}_{s}_{m}"].to_numpy(np.float32)
            if i > 0:
                prev = idx[i - 1]
                for m in TREND_METRICS:
                    out[f"radar_r{i}_{s}_d{m}"] = (
                        radar[f"f{k}_{s}_{m}"] - radar[f"f{prev}_{s}_{m}"]
                    ).to_numpy(np.float32)
    return pd.DataFrame(out)


def env_columns(env: pd.DataFrame) -> pd.DataFrame:
    """Raw environment columns plus derived shear magnitudes and an STP-style composite."""
    skip = {"env_valid", "env_dist_km"}  # bookkeeping, not environment
    out = {
        c: env[c].to_numpy(np.float32)
        for c in env.columns
        if c.startswith("env_") and c not in skip
    }
    for stat in ("pt", "max", "mean"):
        for layer in ("01", "06"):
            u, v = env.get(f"env_ushr{layer}_{stat}"), env.get(f"env_vshr{layer}_{stat}")
            if u is not None and v is not None:
                out[f"env_shr{layer}_{stat}"] = np.hypot(u, v).astype(np.float32)
        cape, lcl = env.get(f"env_mlcape_{stat}"), env.get(f"env_lcl_{stat}")
        srh, cin = env.get(f"env_srh01_{stat}"), env.get(f"env_mlcin_{stat}")
        shr = out.get(f"env_shr06_{stat}")
        if all(v is not None for v in (cape, lcl, srh, cin, shr)):
            # Significant tornado parameter (fixed-layer form) with clipped terms
            shr_term = np.clip(shr / 20.0, 0, 1.5) * (shr >= 12.5)
            lcl_term = np.clip((2000 - lcl) / 1000.0, 0, 1)
            cin_term = np.clip((200 + cin) / 150.0, 0, 1)
            out[f"env_stp_{stat}"] = (
                (cape / 1500.0) * lcl_term * (srh / 150.0) * shr_term * cin_term
            ).astype(np.float32)
    return pd.DataFrame(out)


def build_table(
    config, partition: str, years: list[int], cache: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features DataFrame, metadata DataFrame) for the given years."""
    x, meta = load_pooled(resolve(config["pooled_dir"]), partition, years, cache)
    frames = config.get("frames")
    frames = None if frames is None else [int(i) for i in frames]
    groups = list(config["features"])
    parts = []
    if "embedding" in groups:
        parts.append(embedding_columns(x, frames))
    if "meta" in groups:
        parts.append(meta_columns(meta))
    if "radar" in groups:
        radar = load_csvs(resolve(config.get("radar_dir")), "_radar.csv", cache)
        if radar is None:
            raise FileNotFoundError("radar group needs radar_dir (tornet-radar-metrics)")
        aligned = meta[["filename"]].merge(radar, on="filename", how="left", sort=False)
        parts.append(radar_columns(aligned, frames))
    if "env" in groups:
        env = load_csv(resolve(config.get("env_csv")), cache)
        if env is None:
            raise FileNotFoundError("env group needs env_csv (tornet-env-fetch)")
        aligned = meta[["filename"]].merge(env, on="filename", how="left", sort=False)
        parts.append(env_columns(aligned))
    unknown = set(groups) - {"embedding", "meta", "radar", "env"}
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}")
    table = pd.concat(parts, axis=1)
    return table, meta


def sample_weights(meta: pd.DataFrame, config) -> np.ndarray:
    x = {
        "ef_number": meta["ef_number"].to_numpy(),
        "category": meta["category"].map(CATEGORY_CODES).to_numpy(),
    }
    _, _, w = pp.compute_sample_weight(
        x,
        meta["label"].to_numpy(),
        wN=config["wN"],
        w0=config["w0"],
        w1=config["w1"],
        w2=config["w2"],
        wW=config["wW"],
    )
    return np.asarray(w, dtype=np.float32)


def group_of(column: str) -> str:
    return column.split("_", 1)[0]


def main(config, cache: dict | None = None):
    import lightgbm as lgb

    cache = {} if cache is None else cache
    logging.info(config)
    x_train, m_train = build_table(config, "train", config["train_years"], cache)
    y_train, w_train = m_train["label"].to_numpy(int), sample_weights(m_train, config)
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=False)
    valid_sets, callbacks = [train_set], [lgb.log_evaluation(100)]
    if config["val_years"]:
        x_val, m_val = build_table(config, "train", config["val_years"], cache)
        valid_sets.append(
            lgb.Dataset(
                x_val,
                label=m_val["label"].to_numpy(int),
                weight=sample_weights(m_val, config),
                reference=train_set,
            )
        )
        callbacks.append(lgb.early_stopping(int(config["early_stopping"]), first_metric_only=True))
    logging.info(f"train {x_train.shape}, groups {sorted({group_of(c) for c in x_train.columns})}")
    params = {
        **config["params"],
        "metric": ["auc", "average_precision"],
        "seed": int(config["seed"]),
    }
    evals = {}
    callbacks.append(lgb.record_evaluation(evals))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(config["num_rounds"]),
        valid_sets=valid_sets,
        valid_names=["train", "val"][: len(valid_sets)],
        callbacks=callbacks,
    )

    expdir = make_exp_dir(exp_dir=config["exp_dir"], prefix=config["exp_name"])
    git_commit, git_dirty = git_revision()
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config}, f, indent=2)
    booster.save_model(os.path.join(expdir, "model.txt"))
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "split": booster.feature_importance("split"),
        }
    ).sort_values("gain", ascending=False)
    importance["group"] = importance["feature"].map(group_of)
    importance.to_csv(os.path.join(expdir, "feature_importance.csv"), index=False)
    total = importance["gain"].sum() or 1.0
    by_group = (importance.groupby("group")["gain"].sum() / total).round(4).to_dict()
    history = pd.DataFrame({f"{n}_{m}": v for n, d in evals.items() for m, v in d.items()})
    history.to_csv(os.path.join(expdir, "history.csv"), index_label="iteration")

    results = {
        "AUC": float(evals["val"]["auc"][booster.best_iteration - 1]) if "val" in evals else None,
        "AUCPR": float(evals["val"]["average_precision"][booster.best_iteration - 1])
        if "val" in evals
        else None,
        "best_iteration": int(booster.best_iteration),
        "exp_name": config["exp_name"],
        "features": config["features"],
        "frames_used": config.get("frames"),
        "n_features": int(x_train.shape[1]),
        "gain_by_group": by_group,
        "top_features": importance.head(25)["feature"].tolist(),
        "train_years": list(config["train_years"]),
        "val_years": list(config["val_years"]),
        "seed": int(config["seed"]),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "expdir": expdir,
    }
    if config["test_years"]:
        x_test, m_test = build_table(config, "test", config["test_years"], cache)
        prob = booster.predict(x_test, num_iteration=booster.best_iteration)
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        rows = m_test.drop(columns=["year"], errors="ignore").to_dict("records")
        preds = finish_predictions(
            [{**r, "logit": float(np.log(p / (1 - p)))} for r, p in zip(rows, prob, strict=True)]
        )
        preds.to_csv(os.path.join(expdir, "predictions.csv"), index=False)
        scores = score_predictions(preds)
        scores["years"] = list(config["test_years"])
        with open(os.path.join(expdir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
        results["test_AUC"], results["test_AUCPR"] = scores["AUC"], scores["AUCPR"]
        results["predictions"] = os.path.join(expdir, "predictions.csv")
        logging.info(f"test scores: {scores}")
    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"results: {results}")
    return results


def cli():
    paths = sys.argv[1:] or [None]
    cache: dict = {}
    for path in paths:
        config = copy.deepcopy(DEFAULT_CONFIG)
        if path:
            with open(path) as f:
                loaded = json.load(f)
            params = {**config["params"], **loaded.pop("params", {})}
            config.update(loaded)
            config["params"] = params
        main(config, cache)


if __name__ == "__main__":
    cli()
