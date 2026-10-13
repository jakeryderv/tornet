"""Radar forecaster on the synthetic dataset: loader layout, model shapes, pretrained encoder
transfer, training in both modes, and scoring through the frozen detector."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from .conftest import AZ, FRAMES, RNG, TILT

keras = pytest.importorskip("keras")

from tornet.data.constants import ALL_VARIABLES  # noqa: E402
from tornet.data.forecast import (  # noqa: E402
    BACKGROUND,
    ForecastLoader,
    denormalize_fields,
    normalize_fields,
)
from tornet.models.cnn_baseline import build_model  # noqa: E402
from tornet.models.forecaster import (  # noqa: E402
    build_forecaster,
    load_encoder_weights,
    persistence_fields,
    pooled_sizes,
)
from tornet.scripts import score_forecast, train_forecaster, train_keras  # noqa: E402

SHAPE = (AZ, RNG, TILT)
N_IN, HORIZONS = 1, FRAMES - 1


@pytest.fixture(scope="module")
def detector(tornet_root, tmp_path_factory):
    """A tiny trained single-frame detector standing in for the published weights."""
    os.environ["TORNET_ROOT"] = str(tornet_root)
    exp = tmp_path_factory.mktemp("det")
    r = train_keras.main(
        {
            **train_keras.DEFAULT_CONFIG,
            "epochs": 1,
            "batch_size": 2,
            "start_filters": 4,
            "dataloader_kwargs": {},
            "exp_dir": str(exp),
            "train_years": [2013],
            "val_years": [2014],
        }
    )
    return r["last_checkpoint"]


def test_loader_layout_and_normalisation(tornet_root):
    ds = ForecastLoader(
        str(tornet_root), "train", [2013], batch_size=4, n_in=N_IN, horizons=HORIZONS
    )
    x, y = ds[0]
    n = len(x["category"])
    for v in ALL_VARIABLES:
        assert x[v].shape == (n, N_IN, *SHAPE)
    assert x["coordinates"].shape == (n, AZ, RNG, 2) and x["last_fields"].shape == (
        n,
        AZ,
        RNG,
        2 * 6,
    )
    assert y["fields"].shape == (n, HORIZONS, AZ, RNG, 12) and y["logits"].shape == (n, HORIZONS)
    assert y["minutes"].shape == (n, HORIZONS) and np.allclose(y["minutes"], 5.0)
    assert np.isfinite(y["fields"]).all()  # NaN gates became the background value
    assert (y["fields"] == BACKGROUND).any()
    # denormalising the last input scan reproduces its raw values where they were finite
    raw = denormalize_fields(x["last_fields"])
    for v in ALL_VARIABLES:
        orig = x[v][:, -1]
        finite = np.isfinite(orig)
        assert np.allclose(raw[v][finite], orig[finite], atol=1e-4)
    # key selection for training
    ds2 = ForecastLoader(
        str(tornet_root), "train", [2013], batch_size=4, n_in=N_IN, horizons=HORIZONS,
        select_keys=ForecastLoader.model_input_keys(), target_keys=["fields"],
    )  # fmt: skip
    x2, y2 = ds2[0]
    assert set(x2) == set(ForecastLoader.model_input_keys()) and set(y2) == {"fields"}
    assert np.array_equal(
        normalize_fields({v: x[v][:, 0] for v in ALL_VARIABLES}), x["last_fields"]
    )


def test_forecaster_shapes_and_encoder_transfer(detector):
    assert pooled_sizes((120, 240)) == [(60, 120), (30, 60), (15, 30), (8, 15)]
    det = keras.saving.load_model(detector, compile=False)
    m = build_forecaster(
        shape=SHAPE,
        n_in=N_IN,
        horizons=HORIZONS,
        start_filters=4,
        decoder_filters=(8, 8, 8, 8),
        detector=det,
    )
    assert load_encoder_weights(m, det) == 10
    rng = np.random.default_rng(0)
    x = {v: rng.standard_normal((2, N_IN, *SHAPE), dtype=np.float32) for v in ALL_VARIABLES}
    x["range_folded_mask"] = np.zeros((2, N_IN, *SHAPE), np.float32)
    x["coordinates"] = rng.standard_normal((2, AZ, RNG, 2), dtype=np.float32)
    out = {k: keras.ops.convert_to_numpy(v) for k, v in m(x).items()}
    assert out["fields"].shape == (2, HORIZONS, AZ, RNG, 12) and out["logits"].shape == (
        2,
        HORIZONS,
    )
    # the frozen detector inside agrees with the detector called directly on the forecast
    direct = score_forecast.detector_logits(
        det, out["fields"][:, 0], x["range_folded_mask"][:, -1], x["coordinates"]
    )
    assert np.allclose(direct, out["logits"][:, 0], atol=1e-4)
    assert not any(w.trainable for w in det.weights) or not det.trainable
    assert persistence_fields(np.zeros((2, AZ, RNG, 12)), HORIZONS).shape == (
        2,
        HORIZONS,
        AZ,
        RNG,
        12,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_encoder_weights(m, build_model(shape=SHAPE, c_shape=(AZ, RNG, 2), start_filters=8))


def test_train_and_score_both_modes(detector, tornet_root, tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    exp.mkdir()
    base = {
        **train_forecaster.DEFAULT_CONFIG,
        "train_years": [2013],
        "val_years": [2014],
        "epochs": 1,
        "batch_size": 2,
        "n_in": N_IN,
        "horizons": HORIZONS,
        "start_filters": 4,
        "decoder_filters": [8, 8, 8, 8],
        "init_from": detector,
        "detector": detector,
        "exp_dir": str(exp),
        "dataloader_kwargs": {},
    }
    r0 = train_forecaster.main({**base, "detector_weight": 0.0, "exp_name": "fc_mae"})
    assert (
        r0["epochs_run"] == 1 and r0["val_fields_mae"] is not None and r0["val_logits_AUC"] is None
    )
    r1 = train_forecaster.main({**base, "detector_weight": 0.5, "exp_name": "fc_det"})
    assert r1["val_logits_AUC"] is not None and os.path.exists(r1["best_checkpoint"])
    frozen = keras.saving.load_model(r1["best_checkpoint"], compile=False)
    assert "logits" in frozen.output

    out = tmp_path / "scores"
    monkeypatch.setattr(
        sys, "argv",
        ["tornet-score-forecast", "--model_path", r1["best_checkpoint"], "--years", "2014", "--out", str(out),
         "--batch_size", "2", "--workers", "1", "--bootstrap", "5", "--detector", detector],
    )  # fmt: skip
    score_forecast.main()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["horizons"]) == HORIZONS
    h0 = summary["horizons"][0]
    assert h0["lead_minutes_median"] == 5.0 and h0["mae_forecast"] > 0 and h0["mae_persistence"] > 0
    for w in ("persistence", "forecast", "truth"):
        df = pd.read_csv(out / f"predictions_{w}_h0.csv")
        assert {
            "filename",
            "label",
            "category",
            "ef_number",
            "year",
            "prob",
            "logit",
            "minutes",
        } <= set(df.columns)
        assert len(df) == h0["n"]
    assert "NC" in h0["forecast_minus_persistence"] and "NC" in h0["scores"]["truth"]
