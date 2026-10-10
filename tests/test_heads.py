"""Classification heads: shapes, parameter matching, serialization, and build_model integration."""

import os
import tempfile

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from tornet.models import heads  # noqa: E402
from tornet.models.cnn_baseline import build_model  # noqa: E402

C = 64
SHAPE = (7, 15, C)


@pytest.mark.parametrize(
    ("head", "kwargs"),
    [
        ("maxpool", {}),
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "max"}),
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "mean"}),
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "cls"}),
        ("token_mlp", {"depth": 1, "pool": "max"}),
        ("token_mlp", {"depth": 1, "pool": "mean"}),
        ("transformer", {"depth": 1, "num_heads": 4, "dim": 32}),
    ],
)
def test_heads_produce_one_logit(head, kwargs):
    m = heads.build_head_model(SHAPE, head, **kwargs)
    out = m.predict(np.random.randn(3, *SHAPE).astype(np.float32), verbose=0)
    assert out.shape == (3, 1)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_control_matches_transformer_parameter_count(depth):
    t = heads.build_head_model(SHAPE, "transformer", depth=depth, num_heads=4)
    c = heads.build_head_model(SHAPE, "token_mlp", depth=depth)
    assert abs(t.count_params() - c.count_params()) / t.count_params() < 0.03


def test_matched_hidden_formula():
    assert heads.matched_hidden(384) == 2304
    assert heads.matched_hidden(64, mlp_ratio=2) == 256


def test_token_mlp_rejects_cls():
    with pytest.raises(ValueError, match="class token"):
        heads.build_head_model(SHAPE, "token_mlp", pool="cls")


def test_heads_round_trip_through_keras_save():
    for head, kw in [
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "cls"}),
        ("token_mlp", {"depth": 1}),
    ]:
        m = heads.build_head_model(SHAPE, head, **kw)
        x = np.random.randn(2, *SHAPE).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.keras")
            m.save(path)
            m2 = keras.saving.load_model(path, compile=False)
        np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-5)


def test_max_pooled_heads_accept_any_map_size():
    for head in ("transformer", "token_mlp"):
        kw = {"depth": 1, "num_heads": 4} if head == "transformer" else {"depth": 1}
        m = heads.build_head_model((None, None, C), head, pool="max", **kw)
        assert m.predict(np.random.randn(1, 9, 20, C).astype(np.float32), verbose=0).shape == (1, 1)


def test_build_model_head_option():
    base = build_model(start_filters=4)
    hybrid = build_model(
        start_filters=4, head="transformer", head_kwargs={"depth": 1, "num_heads": 4}
    )
    control = build_model(start_filters=4, head="token_mlp", head_kwargs={"depth": 1})
    assert abs(hybrid.count_params() - control.count_params()) / hybrid.count_params() < 0.03
    # backbone identical across heads: same number of CoordConv layers
    n = lambda m: sum(type(x).__name__ == "CoordConv2D" for x in m.layers)  # noqa: E731
    assert n(base) == n(hybrid) == n(control) == 10
    with pytest.raises(ValueError, match="unknown head"):
        build_model(start_filters=4, head="mlp")


def test_heads_can_emit_named_extra_outputs():
    x = np.random.randn(2, *SHAPE).astype(np.float32)
    for head, kw in [
        ("maxpool", {}),
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "cls"}),
        ("transformer", {"depth": 1, "num_heads": 4, "pool": "max"}),
        ("token_mlp", {"depth": 1}),
    ]:
        m = heads.build_head_model(SHAPE, head, outputs=("logit", "aux"), **kw)
        out = m.predict(x, verbose=0)
        assert set(out) == {"logit", "aux"} and out["logit"].shape == out["aux"].shape == (2, 1)
    # the single-output maxpool head keeps upstream's "heatmap" layer name
    single = heads.build_head_model(SHAPE, "maxpool")
    assert "heatmap" in [layer.name for layer in single.layers]
