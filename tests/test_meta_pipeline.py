"""Cache round 2 and the meta model on the synthetic dataset: the new heads, the per-frame label
target, pooled embeddings, radar metrics, environment extraction and the gradient-boosting model."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from .conftest import FRAMES, SYNTHETIC_SAMPLES, TILT

keras = pytest.importorskip("keras")

from tornet.env import hrrr  # noqa: E402
from tornet.models.heads import build_head_model  # noqa: E402
from tornet.scripts import (  # noqa: E402
    cache_features,
    env_fetch,
    meta_model,
    pool_features,
    radar_metrics,
    train_head,
    train_keras,
)

TINY = {"epochs": 1, "batch_size": 2, "start_filters": 4, "dataloader_kwargs": {}}
N_TRAIN_2013 = sum(1 for rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "train" and "/2013/" in rel)


@pytest.fixture(scope="module")
def pipeline(tornet_root, tmp_path_factory):
    """A trained tiny model, a two-frame feature cache, radar metrics and pooled embeddings."""
    os.environ["TORNET_ROOT"] = str(tornet_root)
    tmp = tmp_path_factory.mktemp("pipeline")
    exp = tmp / "experiments"
    exp.mkdir()
    r = train_keras.main(
        {
            **train_keras.DEFAULT_CONFIG,
            **TINY,
            "exp_dir": str(exp),
            "train_years": [2013],
            "val_years": [2014],
        }
    )
    cache = tmp / "features2"
    argv = sys.argv
    sys.argv = ["tornet-cache-features", "--model_path", r["last_checkpoint"], "--years", "2013", "2014",
                "--out", str(cache), "--frames", str(FRAMES), "--batch_size", "2", "--workers", "1"]  # fmt: skip
    try:
        cache_features.main()
    finally:
        sys.argv = argv
    radar = tmp / "radar_metrics"
    for year in (2013, 2014):
        for partition in ("train", "test"):
            radar_metrics.run(str(tornet_root), partition, year, str(radar), workers=1)
    pooled = tmp / "pooled"
    sys.argv = ["tornet-pool-features", "--features_dir", str(cache), "--out", str(pooled)]
    try:
        pool_features.main()
    finally:
        sys.argv = argv
    return {"exp": exp, "cache": cache, "radar": radar, "pooled": pooled, "root": tornet_root}


def test_new_heads_build_and_round_trip(tmp_path):
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 5, 16), dtype=np.float32)
    for head, kw in [
        ("spacetime_transformer", {"depth": 1, "num_heads": 4}),
        ("spacetime_transformer", {"depth": 1, "num_heads": 4, "pool": "mean"}),
        ("temporal_diff_concat", {"hidden": (8,)}),
        ("temporal_concat", {"hidden": (8,)}),
    ]:
        m = build_head_model(
            (3, 4, 5, 16), head, outputs=("logit", "aux"), output_units={"aux": 3}, **kw
        )
        out = {k: keras.ops.convert_to_numpy(v) for k, v in m(x).items()}
        assert out["logit"].shape == (2, 1) and out["aux"].shape == (2, 3)
        path = tmp_path / f"{head}.keras"
        m.save(path)
        again = keras.saving.load_model(path, compile=False)
        out2 = {k: keras.ops.convert_to_numpy(v) for k, v in again(x).items()}
        assert all(np.allclose(out[k], out2[k], atol=1e-5) for k in out)
    with pytest.raises(ValueError, match="max or mean"):
        build_head_model((3, 4, 5, 16), "spacetime_transformer", pool="cls")
    # the difference head sees 2T-1 frames' worth of channels
    diff = build_head_model((3, 4, 5, 16), "temporal_diff_concat", hidden=(8,))
    first_conv = next(layer for layer in diff.layers if isinstance(layer, keras.layers.Conv2D))
    assert first_conv.kernel.shape[-2] == 5 * 16


def test_radar_metrics_rows(pipeline):
    df = pd.read_csv(pipeline["radar"] / "train_2013_radar.csv")
    assert len(df) == N_TRAIN_2013 and "error" not in df.columns
    assert {"site", "site_lat", "site_lon", "center_lat", "center_lon", "nyquist"} <= set(
        df.columns
    )
    # the chip centre lies downrange of the site, within the range limits
    row = df.iloc[0]
    assert row["site"] == "KTLX" and 20 < row["rng_center_km"] < 80
    assert (
        abs(row["center_lat"] - row["site_lat"]) < 1
        and abs(row["center_lon"] - row["site_lon"]) < 1
    )
    for k in range(FRAMES):
        assert f"frame_time_{k}" in df and f"frame_label_{k}" in df
        for s in range(TILT):
            for m in ("vrot", "azshear", "velmax", "dbzmax", "valid"):
                assert np.isfinite(df[f"f{k}_s{s}_{m}"]).all()
    assert (df[f"frame_time_{FRAMES - 1}"] > df["frame_time_0"]).all()
    tor = df[df["filename"].str.contains("TOR")]
    assert (tor[f"frame_label_{FRAMES - 1}"] == 1).all()
    # the metrics are what the formulas say on a hand-made sweep
    vel = np.zeros((12, 6), np.float32)
    vel[4, :], vel[6, :] = 20, -20  # a couplet two gates apart
    m = radar_metrics.sweep_metrics(vel, np.full((12, 6), 30.0, np.float32), np.full(6, 50_000.0))
    assert m["vrot"] == 20 and m["velmax"] == 20 and m["dbzmax"] == 30 and m["valid"] == 1
    assert m["azshear"] > 0


def test_pooled_embeddings(pipeline):
    pooled, cache = pipeline["pooled"], pipeline["cache"]
    x = np.load(pooled / "train_2013_pooled.npy")
    feats = np.load(cache / "train_2013_features.npy").astype(np.float32)
    assert x.dtype == np.float16 and x.shape == (len(feats), FRAMES, 2 * feats.shape[-1])
    c = feats.shape[-1]
    assert np.allclose(x[:, :, :c].astype(np.float32), feats.max(axis=(2, 3)), atol=1e-2)
    assert np.allclose(x[:, :, c:].astype(np.float32), feats.mean(axis=(2, 3)), atol=1e-2)
    assert (pooled / "train_2013_meta.csv").exists()
    info = json.loads((pooled / "pooled.json").read_text())
    assert info["cache"]["frames"] == FRAMES


def test_frame_label_target_trains_a_per_frame_output(pipeline):
    exp, cache, radar = pipeline["exp"], pipeline["cache"], pipeline["radar"]
    cfg = {
        **train_head.DEFAULT_CONFIG,
        "features_dir": str(cache),
        "meta_extra": str(radar),
        "train_years": [2013],
        "val_years": [2014],
        "test_years": [2014],
        "head": "temporal_concat",
        "head_kwargs": {"hidden": (8,)},
        "frames": [0],  # the head sees only the oldest scan but predicts every frame's label
        "aux_target": "frame_labels",
        "epochs": 1,
        "batch_size": 2,
        "exp_dir": str(exp),
        "exp_name": "onset_fold_x",
    }
    meta = train_head.join_extra(
        pd.read_csv(cache / "train_2013_meta.csv"), train_head.load_extra(cfg)
    )
    y = train_head.aux_targets(meta, cfg)
    assert y.shape == (N_TRAIN_2013, FRAMES) and set(np.unique(y)) <= {0.0, 1.0}
    cfg["head"] = "temporal_concat"
    cfg["frames"] = [0, 1]
    r = train_head.main(cfg)
    assert r["aux_target"] == "frame_labels" and 0.0 <= r["test_AUC"] <= 1.0
    model = keras.saving.load_model(r["best_checkpoint"], compile=False)
    assert tuple(model.output["aux"].shape[1:]) == (FRAMES,)
    with pytest.raises(ValueError, match="frame_label"):
        train_head.main({**cfg, "meta_extra": None, "exp_name": "onset_missing"})


def fetch_fake_environment(pipeline, tmp_path, monkeypatch):
    """Runs the fetch script against a fake 10x10 grid; returns (env csv path, fields, index, calls)."""
    lat, lon = np.meshgrid(np.linspace(35, 36, 10), np.linspace(-98, -97, 10), indexing="ij")
    grid_index = hrrr.GridIndex.from_grid(lat, lon)
    fields = {name: np.full((10, 10), i, np.float32) for i, name in enumerate(hrrr.FIELDS)}
    fields["sbcape"][5, 5] = 100.0
    calls = []

    class Fake:
        index = grid_index

        def fields(self, valid):
            calls.append(valid)
            return None if valid.year == 2013 else fields

    monkeypatch.setattr(env_fetch, "_FETCHER", Fake())
    monkeypatch.setattr(env_fetch, "_init", lambda *a: None)
    out_dir = tmp_path / "env"
    path = env_fetch.run(str(pipeline["radar"]), str(out_dir), fxx=2, halo=1, workers=1)
    return path, fields, grid_index, calls


def test_environment_extraction_and_fetch(pipeline, tmp_path, monkeypatch):
    # point, box max and box mean on a fake grid come out as defined
    path, fields, index, calls = fetch_fake_environment(pipeline, tmp_path, monkeypatch)
    r, c, dist = index.query([35.0, 36.0], [-98.0, -97.0])
    assert tuple(r) == (0, 9) and tuple(c) == (0, 9) and dist.max() < 1e-6
    out = hrrr.extract(fields, np.array([4, 0]), np.array([4, 0]), halo=1)
    assert out["env_sbcape_pt"][0] == 0 and out["env_sbcape_max"][0] == 100
    assert np.isclose(out["env_sbcape_mean"][0], 100 / 9)
    assert set(out) == set(hrrr.env_columns())

    # the fetch wrote a row per sample, pre-archive hours as unavailable, and resumes by hour
    df = pd.read_csv(path)
    radar = pd.concat(pd.read_csv(p) for p in pipeline["radar"].glob("*_radar.csv"))
    assert set(df["filename"]) == set(radar["filename"])
    assert (df.loc[df["filename"].str.contains("/2013/"), "env_available"] == 0).all()
    assert (df.loc[df["filename"].str.contains("/2014/"), "env_available"] == 1).all()
    n_hours = len(calls)
    env_fetch.run(str(pipeline["radar"]), os.path.dirname(path), fxx=2, halo=1, workers=1)
    assert len(calls) == n_hours  # every hour recorded as done, nothing refetched
    assert len(pd.read_csv(path)) == len(df)
    assert hrrr.nearest_hour(pd.Timestamp("2014-04-28 21:40")) == pd.Timestamp("2014-04-28 22:00")


def test_meta_model_end_to_end(pipeline, tmp_path, monkeypatch):
    env_path, *_ = fetch_fake_environment(pipeline, tmp_path, monkeypatch)
    base = {
        **meta_model.DEFAULT_CONFIG,
        "pooled_dir": str(pipeline["pooled"]),
        "radar_dir": str(pipeline["radar"]),
        "env_csv": env_path,
        "train_years": [2013],
        "val_years": [2014],
        "test_years": [2014],
        "params": {
            **meta_model.DEFAULT_CONFIG["params"],
            "min_child_samples": 1,
            "min_data_in_bin": 1,
            "feature_fraction": 1.0,
        },
        "num_rounds": 5,
        "early_stopping": 5,
        "exp_dir": str(pipeline["exp"]),
    }
    cache = {}
    table, meta = meta_model.build_table(
        {**base, "features": ["embedding", "meta", "radar", "env"], "frames": [0]},
        "train",
        [2013],
        cache,
    )
    groups = {c.split("_", 1)[0] for c in table.columns}
    assert groups == {"emb", "meta", "radar", "env"} and len(table) == len(meta) == N_TRAIN_2013
    assert "radar_r0_s0_vrot" in table and "radar_r1_s0_dvrot" not in table
    two, _ = meta_model.build_table(
        {**base, "features": ["radar"], "frames": [0, 1]}, "train", [2013], cache
    )
    assert "radar_r1_s0_dvrot" in two and "radar_r0_s0_vrot" in two
    assert "env_stp_pt" in table and "env_shr06_max" in table and "env_dist_km" not in table
    with pytest.raises(ValueError, match="unknown feature groups"):
        meta_model.build_table({**base, "features": ["bogus"]}, "train", [2013], cache)

    r = meta_model.main(
        {
            **base,
            "features": ["embedding", "meta", "radar", "env"],
            "frames": [0],
            "exp_name": "meta_fold_x",
        },
        cache,
    )
    assert 0.0 <= r["test_AUC"] <= 1.0 and r["n_features"] == table.shape[1]
    assert set(r["gain_by_group"]) <= {"emb", "meta", "radar", "env"}
    preds = pd.read_csv(os.path.join(r["expdir"], "predictions.csv"))
    assert {"filename", "label", "category", "ef_number", "year", "prob", "logit"} <= set(
        preds.columns
    )
    assert os.path.exists(os.path.join(r["expdir"], "feature_importance.csv"))
    assert os.path.exists(os.path.join(r["expdir"], "model.txt"))

    # the CLI merges params and shares the cache across configs
    cfg_path = tmp_path / "meta.json"
    cfg_path.write_text(
        json.dumps(
            {
                **base,
                "features": ["meta", "radar"],
                "params": {"num_leaves": 3},
                "exp_name": "meta_cli_fold_x",
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["tornet-meta-model", str(cfg_path)])
    meta_model.cli()
    run = sorted(pipeline["exp"].glob("meta_cli_fold_x_*"))[-1]
    params = json.loads((run / "params.json").read_text())["config"]["params"]
    assert params["num_leaves"] == 3 and params["objective"] == "binary"
