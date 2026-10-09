"""Pure numpy preprocessing helpers."""

import numpy as np
import pytest

from tornet.data import preprocess as pp
from tornet.data.constants import ALL_VARIABLES

FRAMES, AZ, RNG, TILT = 2, 6, 8, 2


def make_sample():
    rng = np.random.default_rng(0)
    shape = (FRAMES, AZ, RNG, TILT)
    d = {v: rng.standard_normal(shape).astype(np.float32) for v in ALL_VARIABLES}
    d["range_folded_mask"] = np.zeros(shape, dtype=np.float32)
    d["label"] = np.array([0, 1])
    d["ef_number"] = np.array([1])
    d["category"] = np.array([0])
    d["az_lower"] = np.array([10.0])
    d["az_upper"] = np.array([40.0])
    d["rng_lower"] = np.array([20000.0])
    d["rng_upper"] = np.array([80000.0])
    return d


def test_get_shape_uses_any_radar_variable():
    d = make_sample()
    assert pp.get_shape(d) == (FRAMES, AZ, RNG, TILT)
    assert pp.get_shape({"KDP": d["KDP"]}) == (FRAMES, AZ, RNG, TILT)


@pytest.mark.parametrize(
    ("include_az", "tilt_last", "expected"),
    [
        (True, True, (AZ, RNG, 3)),
        (False, True, (AZ, RNG, 2)),
        (True, False, (3, AZ, RNG)),
        (False, False, (2, AZ, RNG)),
    ],
)
def test_compute_coordinates_shape(include_az, tilt_last, expected):
    d = make_sample()
    if not tilt_last:
        d = pp.permute_dims(d, (0, 3, 1, 2))
    c = pp.compute_coordinates(d, include_az=include_az, tilt_last=tilt_last)
    assert c.shape == expected


def test_compute_coordinates_values():
    d = make_sample()
    min_range_m = 2125.0
    c = pp.compute_coordinates(d, min_range_m=min_range_m, include_az=True)
    r, theta, rinv = c[..., 0], c[..., 1], c[..., 2]

    scale = 1e-5
    assert np.all(r >= min_range_m * scale)
    np.testing.assert_allclose(rinv, 1 / r)
    # Range grows along the range axis and is constant along azimuth.
    assert np.all(np.diff(r, axis=1) >= 0)
    np.testing.assert_allclose(r[0], r[-1])
    # Azimuth is converted to math convention (90 - az) in radians.
    np.testing.assert_allclose(theta[0, 0], np.deg2rad(90 - 10.0))
    np.testing.assert_allclose(theta[-1, 0], np.deg2rad(90 - 40.0))


def test_add_coordinates_mutates_and_returns():
    d = make_sample()
    out = pp.add_coordinates(d)
    assert out is d
    assert d["coordinates"].shape == (AZ, RNG, 3)


def test_remove_time_dim_keeps_last_frame():
    d = make_sample()
    last = d["DBZ"][-1].copy()
    pp.remove_time_dim(d)
    assert d["DBZ"].shape == (AZ, RNG, TILT)
    np.testing.assert_array_equal(d["DBZ"], last)
    assert d["label"] == 1


def test_add_batch_dim():
    d = pp.add_batch_dim(make_sample())
    assert d["DBZ"].shape == (1, FRAMES, AZ, RNG, TILT)
    assert d["label"].shape == (1, FRAMES)


def test_select_keys():
    d = make_sample()
    assert pp.select_keys(d) is d
    assert pp.select_keys(d, keys=None) is d
    assert set(pp.select_keys(d, keys=["DBZ", "label"])) == {"DBZ", "label"}


def test_permute_dims_only_touches_radar_arrays():
    d = make_sample()
    pp.permute_dims(d, (0, 3, 1, 2))
    for v in ALL_VARIABLES + ["range_folded_mask"]:
        assert d[v].shape == (FRAMES, TILT, AZ, RNG)
    assert d["label"].shape == (FRAMES,)


def test_split_x_y():
    d = make_sample()
    x, y = pp.split_x_y(d)
    assert x is d
    np.testing.assert_array_equal(y, d["label"])


def test_compute_sample_weight_by_category_and_ef():
    x = {
        "ef_number": np.array([-1, -1, 0, 1, 2, 3]),
        "category": np.array([1, 2, 0, 0, 0, 0]),  # NUL, WRN, TOR...
    }
    y = np.zeros(6)
    _, _, w = pp.compute_sample_weight(x, y, wN=1.0, w0=2.0, w1=3.0, w2=4.0, wW=0.5)
    np.testing.assert_array_equal(w, [1.0, 0.5, 2.0, 3.0, 4.0, 4.0])
