"""Train a classification head on cached backbone features.

Reads the .npy/.csv pairs written by tornet-cache-features, trains one of the heads in
`tornet.models.heads` on the train years, selects the best epoch on the validation years, then
scores the best checkpoint on the test years. Writes the same run directory layout as tornet-train
(results.json, history.csv, checkpoints/) plus predictions.csv and scores.json for tornet-score.

Usage: tornet-train-head <config.json>
"""

import copy
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402

from tornet.data import preprocess as pp  # noqa: E402
from tornet.models.heads import SoftSupervisedContrastive, build_head_model  # noqa: E402
from tornet.scripts.test_keras import build_metrics, finish_predictions, score_predictions  # noqa: E402
from tornet.utils.general import git_revision, make_callback_dirs, make_exp_dir  # noqa: E402

logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG = {
    "features_dir": None,  # default: $TORNET_ROOT/features
    "train_years": [2013, 2015],
    "val_years": [2014],
    "test_years": [2013, 2014, 2015],
    "head": "transformer",
    "head_kwargs": {},
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "decay_steps": None,  # default: one epoch
    "decay_rate": 0.958,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0.1,
    # Optional auxiliary regression target trained jointly with the binary logit:
    #   "proximity": -min(minutes to the tornado interval, window)/window, in [-1, 0]; nulls -1
    #   "frame_fraction": share of the sample's frames that are tornadic, in [0, 1]
    "aux_target": None,
    "aux_weight": 1.0,
    "aux_window": 60.0,
    # For head "embedding": contrastive supervision on the embedding output. The target is a
    # scalar per sample ("label", or an aux_target name such as "proximity"); pairs with equal
    # targets are positives, decaying with |t_i - t_j| / sigma.
    "contrastive_target": "label",
    "contrastive_weight": 1.0,
    "contrastive_sigma": 0.25,
    "temperature": 0.1,
    "seed": 1234,
    "exp_name": "head",
    "exp_dir": os.environ.get("EXP_DIR", "experiments"),
    "save_all_checkpoints": False,
}

CATEGORY_CODES = {"TOR": 0, "NUL": 1, "WRN": 2}


def load_features(features_dir: str, partition: str, years: list[int]):
    """Concatenate cached features (float16, in RAM) and metadata for the given years."""
    feats, metas = [], []
    for year in years:
        path = os.path.join(features_dir, f"{partition}_{year}_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run tornet-cache-features for {year} first")
        feats.append(np.load(path))
        metas.append(pd.read_csv(os.path.join(features_dir, f"{partition}_{year}_meta.csv")))
    return np.concatenate(feats), pd.concat(metas, ignore_index=True)


def second_target(meta: pd.DataFrame, config) -> np.ndarray | None:
    """Target for the second output: the contrastive target for the embedding head, else the
    auxiliary regression target (or None)."""
    if config["head"] == "embedding":
        t = config.get("contrastive_target", "label")
        if t == "label":
            return meta["label"].to_numpy(dtype=np.float32)
        return aux_targets(meta, {**config, "aux_target": t})
    return aux_targets(meta, config)


def aux_targets(meta: pd.DataFrame, config) -> np.ndarray | None:
    """Auxiliary regression target per sample, or None when the config has none."""
    target = config.get("aux_target")
    if not target:
        return None
    if target == "frame_fraction":
        return meta["frame_fraction"].to_numpy(dtype=np.float32)
    if target == "proximity":
        window = float(config["aux_window"])
        before = meta["minutes_to_start"].to_numpy(dtype=np.float64)
        after = meta["minutes_since_end"].to_numpy(dtype=np.float64)
        # distance to the tornado interval: positive before onset, positive after end, 0 during
        dist = np.maximum(np.nan_to_num(before, nan=window), 0) + np.maximum(
            np.nan_to_num(after, nan=window), 0
        )
        dist = np.where(np.isnan(before) | np.isnan(after), window, dist)
        return (-np.minimum(dist, window) / window).astype(np.float32)
    raise ValueError(f"unknown aux_target {target!r}")


class FeatureBatches(keras.utils.PyDataset):
    """Batches of (features float32, targets, sample weights) from in-memory float16 features.

    With `aux` given, targets and weights are dicts keyed "logit" and "aux"."""

    def __init__(
        self,
        features,
        labels,
        weights,
        batch_size,
        shuffle,
        seed=0,
        aux=None,
        aux_name="aux",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features, self.labels, self.weights, self.aux = features, labels, weights, aux
        self.aux_name = aux_name or "aux"
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(labels))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        sel = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = self.features[sel].astype(np.float32)
        if self.aux is None:
            return x, self.labels[sel], self.weights[sel]
        return (
            x,
            {"logit": self.labels[sel], self.aux_name: self.aux[sel]},
            {"logit": self.weights[sel], self.aux_name: np.ones_like(self.weights[sel])},
        )

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def sample_weights(meta: pd.DataFrame, config) -> np.ndarray:
    x = {
        "ef_number": meta["ef_number"].to_numpy(),
        "category": meta["category"].map(CATEGORY_CODES).to_numpy(),
    }
    y = meta["label"].to_numpy()
    _, _, w = pp.compute_sample_weight(
        x, y, wN=config["wN"], w0=config["w0"], w1=config["w1"], w2=config["w2"], wW=config["wW"]
    )
    return np.asarray(w, dtype=np.float32)


def predict_features(model, features, batch_size=512) -> np.ndarray:
    """Binary logits for cached features; takes the "logit" output of a multi-output head."""
    out = []
    for i in range(0, len(features), batch_size):
        p = model.predict_on_batch(features[i : i + batch_size].astype(np.float32))
        if isinstance(p, dict):
            p = p["logit"]
        out.append(np.asarray(p).reshape(-1))
    return np.concatenate(out)


def main(config):
    seed = config.get("seed")
    if seed is not None:
        keras.utils.set_random_seed(int(seed))
    features_dir = config["features_dir"] or os.path.join(os.environ["TORNET_ROOT"], "features")
    logging.info(f"features from {features_dir}")
    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(config)

    aux = config.get("aux_target")
    embedding = config["head"] == "embedding"
    outputs = ("logit", "embedding") if embedding else (("logit", "aux") if aux else ("logit",))
    second_name = outputs[1] if len(outputs) > 1 else None
    x_train, m_train = load_features(features_dir, "train", config["train_years"])
    y_train = m_train["label"].to_numpy(dtype=np.float32)
    w_train = sample_weights(m_train, config)
    ds_train = FeatureBatches(
        x_train,
        y_train,
        w_train,
        config["batch_size"],
        shuffle=True,
        seed=seed or 0,
        aux=second_target(m_train, config),
        aux_name=second_name,
    )
    ds_val = None
    if config["val_years"]:
        x_val, m_val = load_features(features_dir, "train", config["val_years"])
        ds_val = FeatureBatches(
            x_val,
            m_val["label"].to_numpy(dtype=np.float32),
            sample_weights(m_val, config),
            config["batch_size"],
            shuffle=False,
            aux=second_target(m_val, config),
            aux_name=second_name,
        )
    logging.info(f"train {x_train.shape}, val {None if ds_val is None else x_val.shape}")

    head_kwargs = dict(config["head_kwargs"])
    if embedding:
        head_kwargs.setdefault("temperature", config["temperature"])
    model = build_head_model(
        tuple(x_train.shape[1:]), config["head"], outputs=outputs, **head_kwargs
    )
    logging.info(f"head {config['head']} {config['head_kwargs']}: {model.count_params():,} params")

    decay_steps = config["decay_steps"] or len(ds_train)
    lr = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps, config["decay_rate"], staircase=False
    )
    bce = keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=config["label_smooth"])
    if embedding:
        model.compile(
            loss={
                "logit": bce,
                "embedding": SoftSupervisedContrastive(
                    temperature=config["temperature"], sigma=config["contrastive_sigma"]
                ),
            },
            loss_weights={"logit": 1.0, "embedding": float(config["contrastive_weight"])},
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics={"logit": build_metrics()},
            weighted_metrics=[],
        )
        metric_prefix = "logit_"
    elif aux:
        model.compile(
            loss={"logit": bce, "aux": keras.losses.Huber()},
            loss_weights={"logit": 1.0, "aux": float(config["aux_weight"])},
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics={"logit": build_metrics()},
            weighted_metrics=[],
        )
        metric_prefix = "logit_"
    else:
        model.compile(
            loss=bce,
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=build_metrics(),
            weighted_metrics=[],
        )
        metric_prefix = ""

    expdir = make_exp_dir(exp_dir=config["exp_dir"], prefix=config["exp_name"])
    git_commit, git_dirty = git_revision()
    logging.info(f"expdir={expdir} (git {git_commit}{'+dirty' if git_dirty else ''})")
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config, "params": model.count_params()}, f, indent=2)
    checkpoints_dir = make_callback_dirs(expdir)
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(expdir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    best_checkpoint = last_checkpoint = None
    if config.get("save_all_checkpoints"):
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoints_dir, "head_{epoch:03d}.keras"), save_best_only=False
            )
        )
    else:
        last_checkpoint = os.path.join(checkpoints_dir, "last.keras")
        callbacks.append(keras.callbacks.ModelCheckpoint(last_checkpoint, save_best_only=False))
        if ds_val is not None:
            best_checkpoint = os.path.join(checkpoints_dir, "best.keras")
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    best_checkpoint,
                    monitor=f"val_{metric_prefix}AUC",
                    mode="max",
                    save_best_only=True,
                )
            )

    history = model.fit(
        ds_train, epochs=config["epochs"], validation_data=ds_val, callbacks=callbacks, verbose=2
    )

    val_auc = history.history.get(f"val_{metric_prefix}AUC", [])
    if val_auc:
        best_epoch = int(np.argmax(val_auc)) + 1
        results = {
            "AUC": float(val_auc[best_epoch - 1]),
            "AUCPR": float(history.history[f"val_{metric_prefix}AUCPR"][best_epoch - 1]),
            "best_epoch": best_epoch,
        }
    else:
        results = {"AUC": None, "AUCPR": None, "best_epoch": None}
    results.update(
        {
            "epochs_run": len(history.history["loss"]),
            "exp_name": config["exp_name"],
            "head": config["head"],
            "head_kwargs": config["head_kwargs"],
            "aux_target": aux,
            "contrastive_target": config["contrastive_target"] if embedding else None,
            "aux_weight": float(config["aux_weight"]) if aux else None,
            "params": int(model.count_params()),
            "seed": seed,
            "train_years": list(config["train_years"]),
            "val_years": list(config["val_years"]),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "expdir": expdir,
        }
    )

    # Score the selected checkpoint on the test years and keep the predictions for tornet-score.
    if config["test_years"]:
        chosen = best_checkpoint if best_checkpoint and val_auc else last_checkpoint
        if chosen and os.path.exists(chosen):
            model = keras.saving.load_model(chosen, compile=False)
        x_test, m_test = load_features(features_dir, "test", config["test_years"])
        rows = m_test.drop(columns=["year"], errors="ignore").to_dict("records")
        logits = predict_features(model, x_test)
        preds = finish_predictions(
            [{**r, "logit": float(lg)} for r, lg in zip(rows, logits, strict=True)]
        )
        preds.to_csv(os.path.join(expdir, "predictions.csv"), index=False)
        scores = score_predictions(preds)
        scores["years"] = list(config["test_years"])
        scores["checkpoint"] = chosen
        with open(os.path.join(expdir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
        if embedding:
            embs = np.concatenate(
                [
                    np.asarray(
                        model.predict_on_batch(x_test[i : i + 512].astype(np.float32))["embedding"]
                    )
                    for i in range(0, len(x_test), 512)
                ]
            )
            np.save(os.path.join(expdir, "embeddings_test.npy"), embs.astype(np.float16))
        results["test_AUC"] = scores["AUC"]
        results["test_AUCPR"] = scores["AUCPR"]
        results["predictions"] = os.path.join(expdir, "predictions.csv")
        logging.info(f"test scores: {scores}")

    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"results: {results}")
    return results


def cli():
    config = copy.deepcopy(DEFAULT_CONFIG)
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config.update(json.load(f))
    main(config)


if __name__ == "__main__":
    cli()
