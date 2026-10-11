"""Backbone initialisation from a saved model and the per-layer-scaled optimizer."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from tornet.data.constants import ALL_VARIABLES  # noqa: E402
from tornet.models.cnn_baseline import (  # noqa: E402
    FEATURES_LAYER,
    backbone_end,
    backbone_layers,
    build_model,
    load_backbone_weights,
)
from tornet.models.optimizers import ScaledAdam, scales_for  # noqa: E402

SHAPE = (24, 48, 2)


def tiny(head="maxpool", **head_kwargs):
    return build_model(
        shape=SHAPE, c_shape=SHAPE, start_filters=4, head=head, head_kwargs=head_kwargs
    )


def inputs(batch=2, seed=0):
    rng = np.random.default_rng(seed)
    x = {v: rng.standard_normal((batch, *SHAPE), dtype=np.float32) for v in ALL_VARIABLES}
    x["range_folded_mask"] = np.zeros((batch, *SHAPE), dtype=np.float32)
    x["coordinates"] = rng.standard_normal((batch, *SHAPE), dtype=np.float32)
    return x


def test_backbone_end_is_the_named_features_layer():
    model = tiny("transformer", depth=1, num_heads=2, pool="cls")
    end = backbone_end(model)
    assert model.layers[end].name == FEATURES_LAYER
    # the transformer head has its own dropouts after the features layer
    assert any(isinstance(layer, keras.layers.Dropout) for layer in model.layers[end + 1 :])
    assert sum(1 for layer in backbone_layers(model) if layer.weights) == 10  # 2+2+3+3 convs


def test_backbone_end_falls_back_to_last_dropout_for_upstream_models():
    inp = keras.Input((8,))
    x = keras.layers.Dense(4)(inp)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dropout(0.1, name="tap")(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inp, out)
    assert model.layers[backbone_end(model)].name == "tap"


def test_load_backbone_weights_copies_backbone_only(tmp_path):
    source = tiny("maxpool")
    path = str(tmp_path / "source.keras")
    source.save(path)

    target = tiny("transformer", depth=1, num_heads=2, pool="cls")
    head_before = [w.copy() for w in target.layers[-1].get_weights()]
    n = load_backbone_weights(target, path)
    assert n == 10

    for a, b in zip(
        [la for la in backbone_layers(source) if la.weights],
        [lb for lb in backbone_layers(target) if lb.weights],
        strict=True,
    ):
        for wa, wb in zip(a.get_weights(), b.get_weights(), strict=True):
            np.testing.assert_array_equal(wa, wb)
    for w0, w1 in zip(head_before, target.layers[-1].get_weights(), strict=True):
        np.testing.assert_array_equal(w0, w1)

    # the backbones now compute the same features
    feats = lambda m: keras.Model(m.input, m.layers[backbone_end(m)].output)  # noqa: E731
    x = inputs()
    np.testing.assert_allclose(
        feats(source).predict(x, verbose=0), feats(target).predict(x, verbose=0), atol=1e-5
    )


def test_load_backbone_weights_rejects_a_different_backbone(tmp_path):
    source = build_model(shape=SHAPE, c_shape=SHAPE, start_filters=8)
    target = tiny()
    with pytest.raises(ValueError, match="shape mismatch"):
        load_backbone_weights(target, source)


def test_scaled_adam_scales_the_step_per_variable():
    model = tiny("maxpool")
    backbone = backbone_layers(model)
    scales = scales_for(backbone, 0.0)
    assert len(scales) == 20  # kernel + bias for each of the 10 convs
    before = {v.path: np.array(v) for v in model.trainable_variables}

    opt = ScaledAdam(learning_rate=1e-2, scales=scales)
    model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(from_logits=True))
    model.train_on_batch(inputs(), np.array([[1.0], [0.0]], dtype=np.float32))

    for v in model.trainable_variables:
        moved = not np.array_equal(before[v.path], np.array(v))
        assert moved == (v.path not in scales), v.path


def test_scaled_adam_round_trips_through_save(tmp_path):
    model = tiny("maxpool")
    model.compile(
        optimizer=ScaledAdam(learning_rate=1e-3, scales=scales_for(backbone_layers(model), 0.1)),
        loss="mse",
    )
    path = str(tmp_path / "m.keras")
    model.save(path)
    loaded = keras.saving.load_model(path)
    assert isinstance(loaded.optimizer, ScaledAdam)
    assert loaded.optimizer.scales == model.optimizer.scales
