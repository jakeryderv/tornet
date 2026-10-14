"""The torch port of the TorNet CNN, the torch forecasting dataset, and the SMP forecaster script
end to end on the synthetic dataset."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from .conftest import AZ, FRAMES, RNG, TILT

keras = pytest.importorskip("keras")
pytest.importorskip("segmentation_models_pytorch")

from tornet.data import preprocess as pp  # noqa: E402
from tornet.data.constants import ALL_VARIABLES  # noqa: E402
from tornet.data.forecast_torch import ForecastDataset, channel_indices, collate  # noqa: E402
from tornet.data.loader import read_file  # noqa: E402
from tornet.models.cnn_baseline import build_model  # noqa: E402
from tornet.models.torch_backbone import TorNetDetector, load_keras_weights  # noqa: E402
from tornet.scripts import forecast_smp  # noqa: E402

N_IN, HORIZONS = 1, FRAMES - 1


@pytest.fixture(scope="module")
def keras_detector(tornet_root):
    """An untrained but deterministic Keras detector at width 4 with the published layout."""
    keras.utils.set_random_seed(0)
    return build_model(shape=(AZ, RNG, TILT), c_shape=(AZ, RNG, 2), start_filters=4)


def test_torch_port_matches_keras(keras_detector, tornet_root):
    det = TorNetDetector(start_filters=4)
    assert load_keras_weights(det, keras_detector) == 13  # 10 CoordConv + 3 head convolutions
    det.eval()
    ds = ForecastDataset(str(tornet_root), "train", [2013, 2014], n_in=N_IN, horizons=HORIZONS)
    batch = collate([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        t_logit = det(batch["x"][:, -1], batch["coords"]).squeeze(1).numpy()
        feats = det.encoder(batch["x"][:, -1], batch["coords"])
    assert [f.shape[1] for f in feats] == det.encoder.out_channels == [4, 8, 16, 32, 32]
    assert feats[0].shape[-2:] == (AZ, RNG) and feats[-1].shape[-2:] == (
        -(-AZ // 16),
        -(-RNG // 16),
    )
    k_logits = []
    for f in ds.files:
        d = read_file(f, variables=ALL_VARIABLES, n_frames=FRAMES)
        pp.add_coordinates(d, include_az=False)
        x = {v: d[v][:1] for v in ALL_VARIABLES}
        x["range_folded_mask"] = d["range_folded_mask"][:1]
        x["coordinates"] = d["coordinates"][None]
        k_logits.append(float(np.asarray(keras_detector.predict_on_batch(x)).reshape(-1)[0]))
    assert np.allclose(
        t_logit, np.array(k_logits), atol=2e-3
    )  # TF32-level numerics on the GPU side
    with pytest.raises(ValueError, match="kernel shape"):  # a wider backbone does not fit
        load_keras_weights(TorNetDetector(start_filters=4).encoder, build_model(start_filters=8))


def test_dataset_layout(tornet_root):
    ds = ForecastDataset(
        str(tornet_root),
        "train",
        [2013],
        n_in=N_IN,
        horizons=HORIZONS,
        weights={"wW": 0.5, "w2": 2.0},
    )
    item = ds[0]
    assert item["x"].shape == (N_IN, 14, AZ, RNG) and item["coords"].shape == (2, AZ, RNG)
    assert item["future"].shape == (HORIZONS, 14, AZ, RNG) and item["labels"].shape == (HORIZONS,)
    assert item["frame_labels"].shape == (FRAMES,) and torch.allclose(
        item["minutes"], torch.full((HORIZONS,), 5.0)
    )
    assert torch.isfinite(item["x"]).all() and torch.isfinite(item["future"]).all()
    assert channel_indices(["VEL", "DBZ"]) == [2, 3, 0, 1]
    batch = collate([ds[i] for i in range(len(ds))])
    assert batch["x"].shape[0] == len(ds) and len(batch["meta"]) == len(ds)
    weights = {m["category"]: float(w) for m, w in zip(batch["meta"], batch["weight"], strict=True)}
    assert weights.get("TOR", 1.0) in (1.0, 2.0) and weights.get("NUL", 1.0) == 1.0


def test_pattern_flags():
    fl = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    pre, ongoing, patterns = forecast_smp.pattern_flags(fl, n_in=2)
    assert pre.tolist() == [True, True, False, False, False]
    assert ongoing.tolist() == [False, False, True, True, False]
    assert patterns[1] == "0011"


@pytest.mark.parametrize(
    "bottleneck,target_products",
    [("cnn", ["VEL", "DBZ"]), ("transformer", ["VEL", "DBZ"]), ("cnn", ALL_VARIABLES)],
)
def test_forecastnet_shapes(bottleneck, target_products):
    cfg = {
        **forecast_smp.DEFAULT_CONFIG,
        "n_in": N_IN,
        "horizons": HORIZONS,
        "encoder_weights": None,
        "start_filters": 4,
        "bottleneck": bottleneck,
        "target_products": target_products,
        "detector_weight": 0.5,
        "transformer": {"depth": 1, "heads": 2, "dropout": 0.0},
        "decoder_channels": [8, 8, 8, 8],
    }
    forecast_smp.check_config(cfg)
    n_ch = 2 * len(target_products)
    model = forecast_smp.ForecastNet(cfg, grid=(-(-AZ // 16), -(-RNG // 16)))
    x = torch.randn(2, N_IN, 14, AZ, RNG)
    coords = torch.randn(2, 2, AZ, RNG)
    out = model(x, coords)
    assert out["fields"].shape == (2, HORIZONS, n_ch, AZ, RNG)
    assert out["logits"].shape == (2, HORIZONS)
    # the likelihood map, and the logit is its maximum
    assert out["cls_map"].shape == (2, HORIZONS, -(-AZ // 16), -(-RNG // 16))
    assert torch.allclose(out["cls_map"].amax(dim=(2, 3)), out["logits"])
    # the forecast chip: drawn products in, everything else persisted from the last scan
    chip = forecast_smp.full_chip(out["fields"][:, 0], x[:, -1], model.target_channels)
    assert chip.shape == (2, 14, AZ, RNG) and torch.equal(chip[:, 12:], x[:, -1, 12:])
    assert torch.equal(chip[:, model.target_channels], out["fields"][:, 0])
    persisted = [i for i in range(12) if i not in model.target_channels]
    assert len(persisted) == 12 - n_ch and torch.equal(chip[:, persisted], x[:, -1, persisted])
    # L1 weights: 1 on the focus products (two sweeps each), the anchor elsewhere
    w = forecast_smp.channel_weights(cfg)
    assert len(w) == n_ch and w.count(1.0) == 4 and all(v == 0.1 for v in w if v != 1.0)


def test_check_config_rejects_untaught_products():
    cfg = {**forecast_smp.DEFAULT_CONFIG, "target_products": ALL_VARIABLES}
    with pytest.raises(ValueError, match="detector_weight"):
        forecast_smp.check_config(cfg)  # 12 channels with no detector loss: 8 would get no gradient
    forecast_smp.check_config({**cfg, "detector_weight": 0.5})
    with pytest.raises(ValueError, match="subset"):
        forecast_smp.check_config({**forecast_smp.DEFAULT_CONFIG, "focus_products": ["KDP"]})
    with pytest.raises(ValueError, match="unknown"):
        forecast_smp.check_config({**forecast_smp.DEFAULT_CONFIG, "target_products": ["VEL", "X"]})


def test_script_end_to_end(tornet_root, keras_detector, tmp_path, monkeypatch):
    monkeypatch.setenv("TORNET_ROOT", str(tornet_root))
    det_path = tmp_path / "det.keras"
    keras_detector.save(det_path)
    cfg = {
        **forecast_smp.DEFAULT_CONFIG,
        "train_years": [2013],
        "val_years": [2014],
        "test_years": [2014],
        "n_in": N_IN,
        "horizons": HORIZONS,
        "encoder_weights": str(det_path),
        "detector": str(det_path),
        "start_filters": 4,
        "decoder_channels": [8, 8, 8, 8],
        "target_products": ALL_VARIABLES,  # the 12-channel variant: focus + anchored products
        "heatmap_weight": 0.5,
        "detector_weight": 0.5,
        "epochs": 1,
        "batch_size": 2,
        "workers": 0,
        "amp": False,
        "bootstrap": 5,
        "exp_dir": str(tmp_path / "experiments"),
        "exp_name": "fc_smp",
    }
    r = forecast_smp.main(cfg)
    assert r["epochs_run"] == 1 and os.path.exists(r["best_checkpoint"])
    assert set(r["mae"]) == {f"h{h}" for h in range(HORIZONS)}
    scores = os.path.join(r["expdir"], "forecast_scores")
    summary = json.loads(open(os.path.join(scores, "summary.json")).read())
    assert len(summary["horizons"]) == HORIZONS and summary["horizons"][0]["n"] == 1
    h0 = summary["horizons"][0]
    assert set(h0["mae_by_product"]["forecast"]) == set(ALL_VARIABLES)
    focus = [h0["mae_by_product"]["forecast"][p] for p in ("VEL", "DBZ")]
    assert np.isclose(h0["mae_forecast"], np.mean(focus))  # headline MAE is over the focus
    for w in ("head", "persistence", "forecast", "truth"):
        df = pd.read_csv(os.path.join(scores, f"predictions_{w}_h0.csv"))
        assert {
            "filename",
            "label",
            "category",
            "ef_number",
            "year",
            "prob",
            "logit",
            "pattern",
            "pre_onset",
        } <= set(df.columns)
    maps = np.load(os.path.join(scores, "maps_h0.npz"))
    assert set(maps.files) == {"filenames", "head", "persistence", "forecast", "truth"}
    assert (
        maps["head"].shape
        == maps["truth"].shape
        == (summary["horizons"][0]["n"], -(-AZ // 16), -(-RNG // 16))
    )
    hist = pd.read_csv(os.path.join(r["expdir"], "history.csv"))
    assert {"train_loss", "train_field", "train_cls", "train_detector", "val_loss"} <= set(
        hist.columns
    )
    # the CLI merges nested dicts
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                **cfg,
                "epochs": 1,
                "transformer": {"depth": 1},
                "target_products": ["VEL", "DBZ"],
                "exp_name": "fc_cli",
                "test_years": [],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["tornet-forecast", str(cfg_path)])
    forecast_smp.cli()
