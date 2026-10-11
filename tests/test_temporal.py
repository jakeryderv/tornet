"""Multi-frame loading and the temporal heads."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from tornet.data.constants import ALL_VARIABLES  # noqa: E402
from tornet.data.loader import get_dataloader  # noqa: E402
from tornet.models.cnn_baseline import backbone_layers, build_model  # noqa: E402
from tornet.models.temporal import TEMPORAL_HEADS  # noqa: E402
from tornet.scripts import test_keras, train_keras  # noqa: E402

from .conftest import AZ, FRAMES, RNG, SYNTHETIC_SAMPLES, TILT  # noqa: E402

SHAPE = (24, 48, 2)


def inputs(frames, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    shape = (frames, *SHAPE) if frames > 1 else SHAPE
    x = {v: rng.standard_normal((batch, *shape), dtype=np.float32) for v in ALL_VARIABLES}
    x["range_folded_mask"] = np.zeros((batch, *shape), dtype=np.float32)
    x["coordinates"] = rng.standard_normal((batch, *SHAPE), dtype=np.float32)
    return x


def test_loader_stacks_frames_behind_the_batch_axis(tornet_root):
    ds = get_dataloader(str(tornet_root), [2013], "train", 2, n_frames=FRAMES)
    x, y = ds[0]
    n = len(y)
    assert x["DBZ"].shape == (n, FRAMES, AZ, RNG, TILT)
    assert x["range_folded_mask"].shape == (n, FRAMES, AZ, RNG, TILT)
    assert x["coordinates"].shape == (n, AZ, RNG, 2)  # shared by every frame
    assert x["time"].shape == (n,) and y.shape == (n,)
    single = get_dataloader(str(tornet_root), [2013], "train", 2, n_frames=1)[0]
    np.testing.assert_array_equal(single[1], y)  # the label is the last frame's
    np.testing.assert_array_equal(single[0]["DBZ"], x["DBZ"][:, -1])


def test_loader_rejects_bad_frame_counts(tornet_root):
    with pytest.raises(ValueError):
        get_dataloader(str(tornet_root), [2013], "train", 2, n_frames=5)


@pytest.mark.parametrize("head", sorted(TEMPORAL_HEADS))
def test_temporal_model_runs_and_shares_the_backbone(head):
    model = build_model(shape=SHAPE, c_shape=SHAPE, start_filters=4, head=head, frames=3)
    assert sum(1 for la in backbone_layers(model) if la.weights) == 10  # one backbone, not three
    out = model.predict(inputs(3), verbose=0)
    assert out.shape == (2, 1) and np.isfinite(out).all()


def test_temporal_heads_accept_any_map_size():
    model = build_model(
        shape=(None, None, 2),
        c_shape=(None, None, 2),
        start_filters=4,
        head="temporal_attention",
        frames=2,
    )
    x = {k: v for k, v in inputs(2).items()}
    bigger = {k: np.concatenate([v, v], axis=-3) for k, v in x.items()}  # double the range axis
    assert model.predict(bigger, verbose=0).shape == (2, 1)


def test_temporal_heads_are_parameter_matched_at_full_width():
    att = build_model(
        shape=SHAPE, c_shape=SHAPE, start_filters=48, head="temporal_attention", frames=4
    )
    cat = build_model(
        shape=SHAPE, c_shape=SHAPE, start_filters=48, head="temporal_concat", frames=4
    )
    base = build_model(shape=SHAPE, c_shape=SHAPE, start_filters=48)
    extra_att, extra_cat = (
        att.count_params() - base.count_params(),
        cat.count_params() - base.count_params(),
    )
    assert abs(extra_att - extra_cat) / extra_cat < 0.05


def test_frames_and_head_must_agree():
    with pytest.raises(ValueError, match="temporal head"):
        build_model(shape=SHAPE, c_shape=SHAPE, start_filters=4, head="maxpool", frames=2)
    with pytest.raises(ValueError, match="frames > 1"):
        build_model(shape=SHAPE, c_shape=SHAPE, start_filters=4, head="temporal_concat")


def test_temporal_model_round_trips_through_save(tmp_path):
    model = build_model(
        shape=SHAPE, c_shape=SHAPE, start_filters=4, head="temporal_attention", frames=2
    )
    x = inputs(2)
    before = model.predict(x, verbose=0)
    path = str(tmp_path / "t.keras")
    model.save(path)
    loaded = keras.saving.load_model(path, compile=False)
    assert test_keras.model_frames(loaded) == 2
    np.testing.assert_allclose(loaded.predict(x, verbose=0), before, atol=1e-5)


def test_train_and_test_temporal_model(tornet_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TORNET_ROOT", str(tornet_root))
    exp = tmp_path / "experiments"
    exp.mkdir()
    config = {
        **train_keras.DEFAULT_CONFIG,
        "epochs": 1,
        "batch_size": 2,
        "start_filters": 4,
        "dataloader_kwargs": {},
        "frames": FRAMES,
        "head": "temporal_concat",
        "exp_dir": str(exp),
        "exp_name": "temporal",
        "train_years": [2013],
        "val_years": [2014],
    }
    r = train_keras.main(config)
    assert r["frames"] == FRAMES and r["head"] == "temporal_concat"

    model = keras.saving.load_model(r["best_checkpoint"], compile=False)
    ds = get_dataloader(
        str(tornet_root),
        [2014],
        "test",
        2,
        select_keys=list(model.input.keys()) + test_keras.META_KEYS,
        n_frames=test_keras.model_frames(model),
    )
    preds = test_keras.predict_dataset(model, ds)
    assert len(preds) == sum(1 for s in SYNTHETIC_SAMPLES if s[1] == "test" and "2014" in s[0])
    assert np.isfinite(preds["logit"]).all()
