"""Shared fixtures.

Builds a tiny synthetic TorNet directory so the data path can be tested without the real dataset.
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Keras 3 defaults to the tensorflow backend, which this fork does not lock.
os.environ.setdefault("KERAS_BACKEND", "torch")

from tornet.data.constants import ALL_VARIABLES  # noqa: E402

# Small but structurally faithful: (time, azimuth, range, tilt)
FRAMES, AZ, RNG, TILT = 2, 12, 24, 2
AZ_LIMITS = (10.0, 40.0)
RANGE_LIMITS = (20000.0, 80000.0)

# (relative path, catalog type, category, ef_number, start time)
SYNTHETIC_SAMPLES = [
    ("train/2013/TOR_130520_193000_KTLX.nc", "train", "TOR", 1, "2013-05-20 19:30:00"),
    ("train/2013/NUL_130601_120000_KFWS.nc", "train", "NUL", -1, "2013-06-01 12:00:00"),
    ("train/2014/WRN_140428_210000_KBMX.nc", "train", "WRN", -1, "2014-04-28 21:00:00"),
    ("test/2014/TOR_140610_000000_KOAX.nc", "test", "TOR", 2, "2014-06-10 00:00:00"),
]


def write_sample(path, *, category, ef_number, start, event_id=1):
    """Write one synthetic TorNet sample with the layout `tornet.data.loader.read_file` expects."""
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    shape = (FRAMES, AZ, RNG, TILT)
    dims = ("time", "azimuth", "range", "tilt")

    data_vars = {}
    for v in ALL_VARIABLES:
        arr = rng.standard_normal(shape).astype(np.float32)
        arr[:, 0, 0, :] = np.nan  # real files contain NaN gates
        data_vars[v] = (dims, arr)
    data_vars["range_folded_mask"] = (dims, np.zeros(shape, dtype=bool))
    labels = np.zeros(FRAMES, dtype=np.int64)
    labels[-1] = int(ef_number >= 0)
    data_vars["frame_labels"] = ("time", labels)
    data_vars["azimuth_limits"] = ("limits", np.array(AZ_LIMITS, dtype=np.float32))
    data_vars["range_limits"] = ("limits", np.array(RANGE_LIMITS, dtype=np.float32))
    data_vars["nyquist_velocity"] = (("time", "tilt"), np.full((FRAMES, TILT), 30.0, np.float32))

    times = pd.date_range(start, periods=FRAMES, freq="5min")
    attrs = {
        "category": category,
        "event_id": event_id,
        "ef_number": ef_number,
        "site_name": "KTLX",
        "site_lat": 35.333,
        "site_lon": -97.278,
    }
    if ef_number >= 0:
        attrs["tornado_start_time"] = start
        attrs["tornado_end_time"] = str(times[-1])

    path.parent.mkdir(parents=True, exist_ok=True)
    coords = {
        "time": times,
        "azimuth": np.linspace(AZ_LIMITS[0], AZ_LIMITS[1], AZ, dtype=np.float32),
        "range": np.linspace(RANGE_LIMITS[0], RANGE_LIMITS[1], RNG, dtype=np.float32),
    }
    xr.Dataset(data_vars, coords=coords, attrs=attrs).to_netcdf(path)


@pytest.fixture(scope="session")
def tornet_root(tmp_path_factory):
    """A directory shaped like TORNET_ROOT: catalog.csv plus train/ and test/ NetCDF files."""
    root = tmp_path_factory.mktemp("tornet")
    rows = []
    for i, (rel, kind, category, ef, start) in enumerate(SYNTHETIC_SAMPLES, start=1):
        write_sample(root / rel, category=category, ef_number=ef, start=start, event_id=i)
        rows.append(
            {
                "filename": rel,
                "type": kind,
                "category": category,
                "event_id": i,
                "ef_number": ef,
                "start_time": start,
                "end_time": str(pd.Timestamp(start) + pd.Timedelta(minutes=5 * (FRAMES - 1))),
            }
        )
    pd.DataFrame(rows).to_csv(root / "catalog.csv", index=False)
    return root
