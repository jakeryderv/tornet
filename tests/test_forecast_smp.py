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
    "encoder,bottleneck", [("tornet", "cnn"), ("tornet", "transformer"), ("resnet18", "cnn")]
)
def test_forecastnet_shapes(encoder, bottleneck):
    cfg = {
        **forecast_smp.DEFAULT_CONFIG,
        "n_in": N_IN,
        "horizons": HORIZONS,
        "encoder": encoder,
        "encoder_weights": None,
        "start_filters": 4,
        "bottleneck": bottleneck,
        "transformer": {"depth": 1, "heads": 2, "dropout": 0.0},
        "decoder_channels": [8, 8, 8, 8],
    }
    model = forecast_smp.ForecastNet(cfg, grid=(-(-AZ // 16), -(-RNG // 16)))
    x = torch.randn(2, N_IN, 14, AZ, RNG)
    coords = torch.randn(2, 2, AZ, RNG)
    out = model(x, coords)
    assert out["fields"].shape == (2, HORIZONS, 4, AZ, RNG) and out["logits"].shape == (2, HORIZONS)
    chip = forecast_smp.full_chip(out["fields"][:, 0], x[:, -1], model.target_channels)
    assert chip.shape == (2, 14, AZ, RNG) and torch.equal(chip[:, 12:], x[:, -1, 12:])


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
                "exp_name": "fc_cli",
                "test_years": [],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["tornet-forecast", str(cfg_path)])
    forecast_smp.cli()
