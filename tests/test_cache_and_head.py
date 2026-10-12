"""tornet-cache-features and tornet-train-head end to end on the synthetic dataset."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from .conftest import SYNTHETIC_SAMPLES

keras = pytest.importorskip("keras")

from tornet.scripts import cache_features, score, test_keras, train_head, train_keras  # noqa: E402

TINY = {"epochs": 1, "batch_size": 2, "start_filters": 4, "dataloader_kwargs": {}}


@pytest.fixture
def trained(tornet_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TORNET_ROOT", str(tornet_root))
    exp = tmp_path / "experiments"
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
    return exp, r["last_checkpoint"]


def test_cache_features_and_verify(trained, tmp_path, monkeypatch):
    exp, checkpoint = trained
    preds = tmp_path / "preds.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-test",
            "--model_path",
            checkpoint,
            "--years",
            "2014",
            "--batch_size",
            "2",
            "--workers",
            "1",
            "--output",
            str(tmp_path / "s.json"),
            "--predictions",
            str(preds),
        ],
    )
    test_keras.main()

    out = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-cache-features",
            "--model_path",
            checkpoint,
            "--years",
            "2013",
            "2014",
            "--out",
            str(out),
            "--batch_size",
            "2",
            "--workers",
            "1",
            "--verify",
            str(preds),
        ],
    )
    cache_features.main()

    info = json.loads((out / "cache.json").read_text())
    assert info["tap_layer"] == "features"
    assert len(info["feature_shape"]) == 3 and info["feature_shape"][-1] == 32  # 8 * start_filters
    assert info["verify"]["max_abs_logit_diff"] < 1e-2  # float16 storage only
    n_train_2013 = sum(
        1 for rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "train" and "/2013/" in rel
    )
    feats = np.load(out / "train_2013_features.npy")
    meta = pd.read_csv(out / "train_2013_meta.csv")
    assert feats.dtype == np.float16 and feats.shape[0] == n_train_2013 == len(meta)
    assert {"filename", "label", "category", "ef_number", "year"} <= set(meta.columns)
    assert not (out / "test_2013_features.npy").exists()  # no 2013 test files in the fixture
    return out


def test_train_head_on_cached_features(trained, tmp_path, monkeypatch, capsys):
    exp, checkpoint = trained
    out = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-cache-features",
            "--model_path",
            checkpoint,
            "--years",
            "2013",
            "2014",
            "--out",
            str(out),
            "--batch_size",
            "2",
            "--workers",
            "1",
        ],
    )
    cache_features.main()

    for head, kw in [
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "max"}),
        ("token_mlp", {"depth": 1}),
    ]:
        r = train_head.main(
            {
                **train_head.DEFAULT_CONFIG,
                "features_dir": str(out),
                "train_years": [2013],
                "val_years": [2014],
                "test_years": [2014],
                "head": head,
                "head_kwargs": kw,
                "epochs": 2,
                "batch_size": 2,
                "exp_dir": str(exp),
                "exp_name": f"head_{head}_fold_x",
            }
        )
        assert r["epochs_run"] == 2 and r["best_epoch"] in (1, 2)
        assert r["params"] > 0 and r["head"] == head
        expdir = r["expdir"]
        assert os.path.exists(r["best_checkpoint"]) and os.path.exists(r["last_checkpoint"])
        preds = pd.read_csv(os.path.join(expdir, "predictions.csv"))
        assert {"filename", "label", "category", "ef_number", "year", "logit", "prob"} <= set(
            preds.columns
        )
        assert set(preds["year"]) == {2014}
        with open(os.path.join(expdir, "scores.json")) as f:
            scores = json.load(f)
        assert 0.0 <= scores["AUC"] <= 1.0 and r["test_AUC"] == scores["AUC"]

    # the scorer accepts head predictions and tornet-results sees both runs
    monkeypatch.setattr(
        sys, "argv", ["tornet-score", os.path.join(expdir, "predictions.csv"), "--bootstrap", "0"]
    )
    score.main()
    from tornet.scripts import results

    df = results.collect(str(exp))
    assert {"head_transformer_fold_x", "head_token_mlp_fold_x"} <= set(df["exp_name"])


def test_train_head_without_validation_or_test(trained, tmp_path, monkeypatch):
    exp, checkpoint = trained
    out = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-cache-features",
            "--model_path",
            checkpoint,
            "--years",
            "2013",
            "--partitions",
            "train",
            "--out",
            str(out),
            "--batch_size",
            "2",
            "--workers",
            "1",
        ],
    )
    cache_features.main()
    r = train_head.main(
        {
            **train_head.DEFAULT_CONFIG,
            "features_dir": str(out),
            "train_years": [2013],
            "val_years": [],
            "test_years": [],
            "head": "maxpool",
            "head_kwargs": {},
            "epochs": 1,
            "batch_size": 2,
            "exp_dir": str(exp),
            "exp_name": "head_final",
        }
    )
    assert r["AUC"] is None and r["best_checkpoint"] is None and "test_AUC" not in r
    assert os.path.exists(r["last_checkpoint"])


def test_train_head_with_auxiliary_target(trained, tmp_path, monkeypatch):
    exp, checkpoint = trained
    out = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-cache-features",
            "--model_path",
            checkpoint,
            "--years",
            "2013",
            "2014",
            "--out",
            str(out),
            "--batch_size",
            "2",
            "--workers",
            "1",
        ],
    )
    cache_features.main()
    meta = pd.read_csv(out / "train_2013_meta.csv")
    assert {"minutes_to_start", "minutes_since_end", "frame_fraction"} <= set(meta.columns)
    tor = meta[meta["category"] == "TOR"].iloc[0]
    assert np.isfinite(tor["minutes_to_start"]) and 0 < tor["frame_fraction"] <= 1
    nul = meta[meta["category"] == "NUL"].iloc[0]
    assert np.isnan(nul["minutes_to_start"]) and nul["frame_fraction"] == 0

    cfg = {
        **train_head.DEFAULT_CONFIG,
        "features_dir": str(out),
        "train_years": [2013],
        "val_years": [2014],
        "test_years": [2014],
        "head": "token_mlp",
        "head_kwargs": {"depth": 1},
        "epochs": 2,
        "batch_size": 2,
        "exp_dir": str(exp),
    }
    prox = train_head.aux_targets(meta, {**cfg, "aux_target": "proximity"})
    assert prox.min() >= -1 and prox.max() <= 0 and prox[meta["category"] == "NUL"].min() == -1
    for target in ("proximity", "frame_fraction"):
        r = train_head.main({**cfg, "aux_target": target, "exp_name": f"head_aux_{target}_fold_x"})
        assert r["aux_target"] == target and r["best_epoch"] in (1, 2)
        assert 0.0 <= r["test_AUC"] <= 1.0
        preds = pd.read_csv(os.path.join(r["expdir"], "predictions.csv"))
        assert "logit" in preds.columns and np.isfinite(preds["logit"]).all()


def test_train_embedding_head(trained, tmp_path, monkeypatch):
    exp, checkpoint = trained
    out = tmp_path / "features"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-cache-features",
            "--model_path",
            checkpoint,
            "--years",
            "2013",
            "2014",
            "--out",
            str(out),
            "--batch_size",
            "2",
            "--workers",
            "1",
        ],
    )
    cache_features.main()
    for target in ("label", "proximity"):
        r = train_head.main(
            {
                **train_head.DEFAULT_CONFIG,
                "features_dir": str(out),
                "train_years": [2013],
                "val_years": [2014],
                "test_years": [2014],
                "head": "embedding",
                "head_kwargs": {"depth": 1, "num_heads": 4, "embed_dim": 8},
                "contrastive_target": target,
                "epochs": 2,
                "batch_size": 2,
                "exp_dir": str(exp),
                "exp_name": f"head_embedding_{target}_fold_x",
            }
        )
        assert r["contrastive_target"] == target and 0.0 <= r["test_AUC"] <= 1.0
        embs = np.load(os.path.join(r["expdir"], "embeddings_test.npy"))
        n_test = sum(1 for _rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "test")
        assert embs.shape == (n_test, 8)


def test_multi_frame_cache_and_heads(trained, tmp_path, monkeypatch):
    """A --frames cache holds every frame's features from the shared backbone; its last frame
    reproduces the single-frame model, temporal heads train on all frames, and single-frame heads
    train on the last frame as the control."""
    exp, checkpoint = trained
    preds = tmp_path / "preds.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tornet-test", "--model_path", checkpoint, "--years", "2014", "--batch_size", "2",
         "--workers", "1", "--predictions", str(preds)],
    )  # fmt: skip
    test_keras.main()

    out = tmp_path / "features2"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tornet-cache-features", "--model_path", checkpoint, "--years", "2013", "2014", "--out",
         str(out), "--frames", "2", "--batch_size", "2", "--workers", "1", "--verify", str(preds)],
    )  # fmt: skip
    cache_features.main()
    info = json.loads((out / "cache.json").read_text())
    assert info["frames"] == 2 and len(info["feature_shape"]) == 4 and info["feature_shape"][0] == 2
    assert info["verify"]["max_abs_logit_diff"] < 1e-2
    feats = np.load(out / "train_2013_features.npy")
    assert feats.ndim == 5 and feats.shape[1] == 2 and feats.dtype == np.float16
    # frames differ (the fixture's scans are not identical), so this is not a repeated last frame
    assert not np.allclose(feats[:, 0].astype(np.float32), feats[:, 1].astype(np.float32))

    for head, kw in [
        ("maxpool", {}),
        ("temporal_attention", {"num_heads": 4}),
        ("temporal_concat", {}),
    ]:
        r = train_head.main(
            {
                **train_head.DEFAULT_CONFIG,
                "features_dir": str(out),
                "train_years": [2013],
                "val_years": [2014],
                "test_years": [2014],
                "head": head,
                "head_kwargs": kw,
                "epochs": 1,
                "batch_size": 2,
                "exp_dir": str(exp),
                "exp_name": f"frozen_{head}_fold_x",
            }
        )
        assert r["frames"] == 2 and r["head"] == head and 0.0 <= r["test_AUC"] <= 1.0
        model = keras.saving.load_model(r["best_checkpoint"], compile=False)
        assert tuple(model.input.shape[1:]) == tuple(info["feature_shape"])


def test_frames_select_which_scans_the_head_sees(trained, tmp_path, monkeypatch):
    """`frames` picks scans from a multi-frame cache: one index gives a single-frame head that
    scan, several give a temporal head that window; configs in one invocation share features."""
    exp, checkpoint = trained
    out = tmp_path / "features2"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tornet-cache-features", "--model_path", checkpoint, "--years", "2013", "2014", "--out",
         str(out), "--frames", "2", "--batch_size", "2", "--workers", "1"],
    )  # fmt: skip
    cache_features.main()
    feats = np.load(out / "train_2013_features.npy")
    assert train_head.select_frames(feats, [0]).shape == feats[:, 0].shape
    assert train_head.select_frames(feats, [0, 1]).shape == feats.shape
    assert train_head.feature_shape(feats.shape, [1]) == feats.shape[2:]
    assert train_head.feature_shape(feats.shape, [0, 1]) == feats.shape[1:]

    base = {
        **train_head.DEFAULT_CONFIG,
        "features_dir": str(out),
        "train_years": [2013],
        "val_years": [2014],
        "test_years": [2014],
        "epochs": 1,
        "batch_size": 2,
        "exp_dir": str(exp),
    }
    cfgs = [
        {**base, "head": "maxpool", "frames": [0], "exp_name": "lead_maxpool_fold_x"},
        {**base, "head": "temporal_attention", "head_kwargs": {"num_heads": 4}, "frames": [0, 1],
         "exp_name": "lead_attention_fold_x"},
    ]  # fmt: skip
    paths = []
    for i, c in enumerate(cfgs):
        p = tmp_path / f"cfg{i}.json"
        p.write_text(json.dumps(c))
        paths.append(str(p))
    monkeypatch.setattr(sys, "argv", ["tornet-train-head", *paths])
    loads = []
    real = train_head.load_features

    def counting(*a, **k):
        loads.append(a[1:3])
        return real(*a, **k)

    monkeypatch.setattr(train_head, "load_features", counting)
    train_head.cli()
    # six loads requested (train/val/test per config), the second config served from the cache
    assert len(loads) == 6
    for name, shape in [
        ("lead_maxpool_fold_x", feats.shape[2:]),
        ("lead_attention_fold_x", feats.shape[1:]),
    ]:
        run = sorted((exp).glob(f"{name}_*"))[-1]
        r = json.loads((run / "results.json").read_text())
        assert r["frames_used"] in ([0], [0, 1]) and 0.0 <= r["test_AUC"] <= 1.0
        model = keras.saving.load_model(run / "checkpoints" / "best.keras", compile=False)
        assert tuple(model.input.shape[1:]) == tuple(shape)

    with pytest.raises(ValueError, match="multi-frame"):
        single = tmp_path / "features1"
        monkeypatch.setattr(
            sys,
            "argv",
            ["tornet-cache-features", "--model_path", checkpoint, "--years", "2013", "2014",
             "--out", str(single), "--batch_size", "2", "--workers", "1"],
        )  # fmt: skip
        cache_features.main()
        train_head.main({**base, "features_dir": str(single), "head": "maxpool", "frames": [0]})
