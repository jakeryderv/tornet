"""Point extraction of HRRR forecast fields at storm locations.

The HRRR (3 km, hourly, CONUS) is on the public AWS archive from 2014-07-30. Fields are read by
HTTP byte range through Herbie, one GRIB message at a time, so an hour costs a few megabytes.

Leakage: the HRRR assimilates radar, so a field valid at the scan time from a run initialised at
that hour has seen the storm. Every sample therefore takes the run initialised `fxx` hours before
the hour nearest the scan, i.e. a forecast issued at least fxx - 0.5 h before the newest scan.

`FIELDS` names what is fetched; `extract` turns one hour's fields into rows for the samples in
that hour: the value at the nearest grid point and the max and mean over a box of
(2 * `halo` + 1) cells around it, since the storm's inflow is not exactly at the chip centre.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

HRRR_START = pd.Timestamp("2014-07-30")

# (search regex for Herbie, mapping from (cfgrib variable, level value) to our column name).
# Shear layers are fetched separately: cfgrib keys both 0-1000 m and 0-6000 m layers as level 0.
FIELD_GROUPS = [
    (
        ":(CAPE|CIN):(surface|90-0 mb|255-0 mb)|:HLCY:(1000|3000)-0 m"
        "|:HGT:level of adiabatic|:(TMP|DPT):2 m",
        {
            ("cape", "surface"): "sbcape",
            ("cin", "surface"): "sbcin",
            ("cape", 9000.0): "mlcape",
            ("cin", 9000.0): "mlcin",
            ("cape", 25500.0): "mucape",
            ("cin", 25500.0): "mucin",
            ("hlcy", 1000.0): "srh01",
            ("hlcy", 3000.0): "srh03",
            ("gh", "adiabaticCondensation"): "lcl",
            ("t2m", "heightAboveGround"): "t2m",
            ("d2m", "heightAboveGround"): "d2m",
        },
    ),
    (":V[UV]CSH:0-1000 m", {("vucsh", 0.0): "ushr01", ("vvcsh", 0.0): "vshr01"}),
    (":V[UV]CSH:0-6000 m", {("vucsh", 0.0): "ushr06", ("vvcsh", 0.0): "vshr06"}),
]
FIELDS = [name for _, mapping in FIELD_GROUPS for name in mapping.values()]
STATS = ("pt", "max", "mean")


def env_columns() -> list[str]:
    return [f"env_{f}_{s}" for f in FIELDS for s in STATS]


def nearest_hour(t: pd.Timestamp) -> pd.Timestamp:
    return (t + pd.Timedelta(minutes=30)).floor("h")


@dataclass
class GridIndex:
    """Nearest HRRR grid cell for a set of lat/lon points (the grid is fixed, so built once)."""

    shape: tuple
    tree: object

    @staticmethod
    def xyz(lat, lon):
        la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    @classmethod
    def from_grid(cls, lat: np.ndarray, lon: np.ndarray) -> GridIndex:
        from scipy.spatial import cKDTree

        return cls(lat.shape, cKDTree(cls.xyz(lat, lon).reshape(-1, 3)))

    def query(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, distance_km) of the nearest cell for each point."""
        dist, idx = self.tree.query(self.xyz(lat, lon).reshape(-1, 3))
        rows, cols = np.unravel_index(idx, self.shape)
        return rows, cols, dist * 6371.0


def flatten_fields(datasets, mapping: dict) -> dict[str, np.ndarray]:
    """Map cfgrib datasets (as Herbie returns them) to {column name: 2-D array}."""
    out = {}
    if not isinstance(datasets, list):
        datasets = [datasets]
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            extra = [d for d in da.dims if d not in ("y", "x")]
            if extra:
                for i, value in enumerate(da[extra[0]].values):
                    key = (var, float(value))
                    if key in mapping:
                        out[mapping[key]] = np.asarray(
                            da.isel({extra[0]: i}).values, dtype=np.float32
                        )
                continue
            level_type = da.attrs.get("GRIB_typeOfLevel")
            candidates = [(var, level_type)]
            for coord in ds.coords:
                if coord in ("latitude", "longitude", "time", "step", "valid_time", "y", "x"):
                    continue
                try:
                    candidates.append((var, float(ds[coord].values)))
                except (TypeError, ValueError):
                    continue
            for key in candidates:
                if key in mapping:
                    out[mapping[key]] = np.asarray(da.values, dtype=np.float32)
                    break
    return out


def extract(fields: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray, halo: int) -> dict:
    """Point, box max and box mean of every field at the given cells -> {column: (n,) array}."""
    out = {}
    n = len(rows)
    for name in FIELDS:
        arr = fields.get(name)
        if arr is None:
            for s in STATS:
                out[f"env_{name}_{s}"] = np.full(n, np.nan, dtype=np.float32)
            continue
        pt = arr[rows, cols]
        mx, mn = np.empty(n, np.float32), np.empty(n, np.float32)
        for i, (r, c) in enumerate(zip(rows, cols, strict=True)):
            box = arr[max(r - halo, 0) : r + halo + 1, max(c - halo, 0) : c + halo + 1]
            mx[i], mn[i] = np.nanmax(box), np.nanmean(box)
        out[f"env_{name}_pt"], out[f"env_{name}_max"], out[f"env_{name}_mean"] = pt, mx, mn
    return out


class HRRRFetcher:
    """Fetches one valid hour's fields (all groups) through Herbie. One instance per process."""

    def __init__(self, fxx: int = 2, save_dir: str | None = None, product: str = "sfc"):
        self.fxx, self.product = fxx, product
        self.save_dir = save_dir or os.path.join(tempfile.gettempdir(), "herbie_tornet")
        self.index: GridIndex | None = None

    def herbie(self, init: pd.Timestamp):
        from herbie import Herbie

        return Herbie(
            init.to_pydatetime(),
            model="hrrr",
            product=self.product,
            fxx=self.fxx,
            save_dir=self.save_dir,
            verbose=False,
        )

    def fields(self, valid: pd.Timestamp) -> dict[str, np.ndarray] | None:
        """All FIELDS for the run initialised `fxx` hours before `valid`, or None when the
        archive has no file for that run."""
        if valid < HRRR_START:
            return None
        init = valid - pd.Timedelta(hours=self.fxx)
        h = self.herbie(init)
        if h.grib is None:
            return None
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for search, mapping in FIELD_GROUPS:
                try:
                    datasets = h.xarray(search, remove_grib=True)
                except Exception as e:  # one missing message should not sink the hour
                    logging.warning(f"{init} fxx={self.fxx} {search}: {e!r}")
                    continue
                if self.index is None:
                    ds0 = datasets[0] if isinstance(datasets, list) else datasets
                    self.index = GridIndex.from_grid(ds0.latitude.values, ds0.longitude.values)
                out.update(flatten_fields(datasets, mapping))
        return out
