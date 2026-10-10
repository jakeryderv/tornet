"""Smoke tests that do not require the TorNet dataset."""

import numpy as np
import pytest

from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX


def test_constants_consistent():
    assert set(ALL_VARIABLES) == set(CHANNEL_MIN_MAX)


def test_build_and_run_keras_baseline():
    keras = pytest.importorskip("keras")
    from tornet.models.cnn_baseline import build_model

    shape = (120, 240, 2)
    model = build_model(shape=shape, c_shape=shape, start_filters=8)

    batch = 2
    rng = np.random.default_rng(0)
    x = {v: rng.standard_normal((batch, *shape), dtype=np.float32) for v in ALL_VARIABLES}
    x["range_folded_mask"] = np.zeros((batch, *shape), dtype=np.float32)
    x["coordinates"] = rng.standard_normal((batch, *shape), dtype=np.float32)

    out = model.predict(x, verbose=0)
    assert out.shape[0] == batch
    assert np.isfinite(out).all()
    assert keras.config.backend() in {"torch", "tensorflow", "jax"}
