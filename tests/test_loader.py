"""Data loading against the synthetic dataset from conftest.py."""

import datetime

import numpy as np
import pytest

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import TornadoDataLoader, query_catalog, read_file

from .conftest import AZ, AZ_LIMITS, FRAMES, RANGE_LIMITS, RNG, SYNTHETIC_SAMPLES, TILT

TOR_FILE = SYNTHETIC_SAMPLES[0][0]
NUL_FILE = SYNTHETIC_SAMPLES[1][0]


def test_read_file_default_last_frame(tornet_root):
    data = read_file(str(tornet_root / TOR_FILE))

    for v in ALL_VARIABLES:
        assert data[v].shape == (1, AZ, RNG, TILT)
        assert data[v].dtype == np.float32
    assert data["range_folded_mask"].shape == (1, AZ, RNG, TILT)
    assert data["range_folded_mask"].dtype == np.float32

    assert data["label"].tolist() == [1]
    assert data["category"].tolist() == [0]  # TOR
    assert data["ef_number"].tolist() == [1]
    assert data["event_id"].tolist() == [1]
    assert data["az_lower"].tolist() == [AZ_LIMITS[0]]
    assert data["az_upper"].tolist() == [AZ_LIMITS[1]]
    assert data["rng_lower"].tolist() == [RANGE_LIMITS[0]]
    assert data["rng_upper"].tolist() == [RANGE_LIMITS[1]]

    expected_start = int(
        (datetime.datetime(2013, 5, 20, 19, 30) - datetime.datetime(1970, 1, 1)).total_seconds()
    )
    assert data["tornado_start_time"].tolist() == [expected_start]
    assert data["tornado_end_time"][0] > expected_start
    assert data["time"].dtype == np.int64
    assert data["time"][0] == expected_start + 5 * 60  # last of two 5-minute frames


def test_read_file_null_sample_has_no_tornado_times(tornet_root):
    data = read_file(str(tornet_root / NUL_FILE))
    assert data["label"].tolist() == [0]
    assert data["category"].tolist() == [1]  # NUL
    assert data["ef_number"].tolist() == [-1]
    assert data["tornado_start_time"].tolist() == [0]
    assert data["tornado_end_time"].tolist() == [0]


def test_read_file_frames_variables_and_tilt_order(tornet_root):
    data = read_file(
        str(tornet_root / TOR_FILE), variables=["DBZ", "VEL"], n_frames=FRAMES, tilt_last=False
    )
    assert set(ALL_VARIABLES) & set(data) == {"DBZ", "VEL"}
    assert data["DBZ"].shape == (FRAMES, TILT, AZ, RNG)
    assert data["range_folded_mask"].shape == (FRAMES, TILT, AZ, RNG)
    assert data["label"].shape == (FRAMES,)


def test_read_file_label_follows_ef_number(tornet_root):
    """read_file overrides the last frame label with ef_number >= 0 (a v1.0 data fix)."""
    tor = read_file(str(tornet_root / TOR_FILE))
    nul = read_file(str(tornet_root / NUL_FILE))
    assert tor["label"][-1] == (tor["ef_number"][0] >= 0) == 1
    assert nul["label"][-1] == (nul["ef_number"][0] >= 0) == 0


def test_query_catalog_filters_by_type_and_year(tornet_root):
    root = str(tornet_root)
    train_2013 = query_catalog(root, "train", [2013], random_state=0)
    assert len(train_2013) == 2
    assert all(f.startswith(root) and f.endswith(".nc") for f in train_2013)

    assert len(query_catalog(root, "train", [2013, 2014], random_state=0)) == 3
    assert len(query_catalog(root, "train", (2013, 2014), random_state=0)) == 3  # tuple years
    assert len(query_catalog(root, "test", [2014], random_state=0)) == 1
    assert query_catalog(root, "test", [2013], random_state=0) == []


def test_query_catalog_shuffle_is_seeded(tornet_root):
    root = str(tornet_root)
    years = [2013, 2014]
    a = query_catalog(root, "train", years, random_state=1)
    b = query_catalog(root, "train", years, random_state=1)
    c = query_catalog(root, "train", years, random_state=2)
    assert a == b
    assert sorted(a) == sorted(c)


def test_query_catalog_missing_catalog_raises(tmp_path):
    with pytest.raises(RuntimeError, match="catalog.csv"):
        query_catalog(str(tmp_path), "train", [2013], random_state=0)


def test_tornado_data_loader_indexes_iterates_and_transforms(tornet_root):
    files = query_catalog(str(tornet_root), "train", [2013, 2014], random_state=0)

    def tag(d):
        d["tagged"] = True
        return d

    loader = TornadoDataLoader(files, variables=["DBZ"], transform=tag)
    assert len(loader) == 3
    assert loader[0]["tagged"] is True
    assert "VEL" not in loader[0]

    seen = list(loader)
    assert len(seen) == 3
    assert all(s["DBZ"].shape == (1, AZ, RNG, TILT) for s in seen)
    assert len(list(loader)) == 3  # a second pass restarts the iterator


def test_keras_dataloader_batches(tornet_root):
    pytest.importorskip("keras")
    from tornet.data.keras_loader import KerasDataLoader

    weights = {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5}
    ds = KerasDataLoader(
        data_root=str(tornet_root),
        data_type="train",
        years=[2013, 2014],
        batch_size=2,
        weights=weights,
        select_keys=["DBZ", "VEL", "range_folded_mask", "coordinates"],
    )
    assert len(ds) == 2  # 3 files, batch size 2

    x, y, w = ds[0]
    assert set(x) == {"DBZ", "VEL", "range_folded_mask", "coordinates"}
    assert x["DBZ"].shape == (2, AZ, RNG, TILT)
    assert x["coordinates"].shape == (2, AZ, RNG, 2)  # R and 1/R, no azimuth channel
    assert y.shape == (2,)
    assert w.shape == (2,)
    assert set(np.unique(w)).issubset({0.5, 1.0, 2.0})

    x_last, y_last, w_last = ds[1]
    assert x_last["DBZ"].shape[0] == 1
    assert y_last.shape == (1,)
