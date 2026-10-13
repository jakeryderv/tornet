"""Per-sample radar metadata and hand-computed rotation metrics from the raw TorNet chips.

One pass over the NetCDF files writes <out>/<partition>_<year>_radar.csv with, per sample:

- filename, site, site_lat/site_lon, the chip centre (az_center_deg, rng_center_km,
  center_lat/center_lon: where the storm is, for joining environment data), nyquist velocity
- per frame k (oldest first): frame_time_k (epoch seconds) and frame_label_k, the file's own
  per-frame tornado label
- per frame k and sweep s, from the velocity and reflectivity fields:
    f{k}_s{s}_vrot      rotational velocity proxy: max over gates of (max - min) / 2 of the
                        velocity inside a 2.5 degree azimuth window at one range
    f{k}_s{s}_azshear   max azimuthal shear (1/s) of the 3-gate-smoothed velocity
    f{k}_s{s}_velmax    max |velocity| (m/s)
    f{k}_s{s}_dbzmax    max reflectivity (dBZ)
    f{k}_s{s}_valid     fraction of velocity gates that are valid

These are the classic precursors forecasters watch (a tightening, strengthening couplet), so they
are both inputs to the meta model and an interpretable yardstick for what the backbone sees.

Usage: tornet-radar-metrics [--years 2013 ...] [--partitions train test] [--out radar_metrics]
                            [--workers 8]
"""

import argparse
import logging
import math
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from tornet.data.loader import query_catalog

logging.basicConfig(level=logging.INFO)

EARTH_RADIUS_KM = 6371.0
AZ_WINDOW = 5  # gates of 0.5 degrees
AZ_STEP_RAD = math.radians(0.5)


def destination(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple:
    """Great-circle destination point from (lat, lon) along a bearing."""
    lat1, lon1, brg = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    d = distance_km / EARTH_RADIUS_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg))
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def sweep_metrics(vel: np.ndarray, dbz: np.ndarray, range_m: np.ndarray) -> dict:
    """Metrics for one (azimuth, range) sweep of velocity and reflectivity."""
    out = {"valid": float(np.isfinite(vel).mean()) if vel.size else float("nan")}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected
        out["velmax"] = float(np.nanmax(np.abs(vel))) if np.isfinite(vel).any() else float("nan")
        out["dbzmax"] = float(np.nanmax(dbz)) if np.isfinite(dbz).any() else float("nan")
        if vel.shape[0] >= AZ_WINDOW:
            win = np.lib.stride_tricks.sliding_window_view(vel, AZ_WINDOW, axis=0)
            valid = np.isfinite(win).sum(axis=-1) >= 2
            vrot = (np.nanmax(win, axis=-1) - np.nanmin(win, axis=-1)) / 2
            out["vrot"] = (
                float(np.nanmax(np.where(valid, vrot, np.nan))) if valid.any() else float("nan")
            )
        else:
            out["vrot"] = float("nan")
        if vel.shape[0] >= 4:
            # 3-gate azimuth smoothing (nan-aware), then gate-to-gate difference over arc length
            win3 = np.lib.stride_tricks.sliding_window_view(vel, 3, axis=0)
            smooth = np.nanmean(win3, axis=-1)
            diff = np.abs(np.diff(smooth, axis=0))
            arc = np.broadcast_to(range_m[None, :] * AZ_STEP_RAD, diff.shape)
            shear = diff / arc
            out["azshear"] = float(np.nanmax(shear)) if np.isfinite(shear).any() else float("nan")
        else:
            out["azshear"] = float("nan")
    return out


def file_metrics(path: str, data_root: str) -> dict:
    with xr.open_dataset(path) as ds:
        attrs = ds.attrs
        az = ds["azimuth_limits"].values.astype(float)
        rng = ds["range_limits"].values.astype(float)
        az_center = float((az[0] + az[1]) / 2 % 360)
        rng_center_km = float((rng[0] + rng[1]) / 2e3)
        site_lat = float(attrs.get("site_lat", float("nan")))
        site_lon = float(attrs.get("site_lon", float("nan")))
        row = {
            "filename": os.path.relpath(path, data_root),
            "site": str(attrs.get("site_name", "")),
            "site_lat": site_lat,
            "site_lon": site_lon,
            "az_center_deg": az_center,
            "rng_center_km": rng_center_km,
            "n_frames": int(ds.sizes["time"]),
        }
        if np.isfinite(site_lat) and np.isfinite(site_lon):
            row["center_lat"], row["center_lon"] = destination(
                site_lat, site_lon, az_center, rng_center_km
            )
        else:
            row["center_lat"] = row["center_lon"] = float("nan")
        nyq = ds["nyquist_velocity"].values if "nyquist_velocity" in ds else np.array([np.nan])
        row["nyquist"] = float(np.nanmax(nyq))
        times = (ds["time"].values.astype("datetime64[s]").astype(np.int64)).tolist()
        labels = ds["frame_labels"].values.astype(int).tolist()
        vel_all = ds["VEL"].values  # (time, azimuth, range, sweep)
        dbz_all = ds["DBZ"].values
        if "range" in ds.coords or "range" in ds.variables:
            range_m = ds["range"].values.astype(float)
        else:  # synthetic files carry only the limits
            range_m = np.linspace(rng[0], rng[1], vel_all.shape[2])
        for k in range(vel_all.shape[0]):
            row[f"frame_time_{k}"] = times[k]
            row[f"frame_label_{k}"] = labels[k]
            for s in range(vel_all.shape[-1]):
                m = sweep_metrics(vel_all[k, :, :, s], dbz_all[k, :, :, s], range_m)
                for name, value in m.items():
                    row[f"f{k}_s{s}_{name}"] = value
    return row


def _worker(args):
    path, data_root = args
    try:
        return file_metrics(path, data_root)
    except Exception as e:  # keep the pass going; the row records the failure
        return {"filename": os.path.relpath(path, data_root), "error": repr(e)}


def run(
    data_root: str, partition: str, year: int, out: str, workers: int, limit: int | None = None
):
    dst = os.path.join(out, f"{partition}_{year}_radar.csv")
    if os.path.exists(dst):
        logging.info(f"{dst} exists, skipping")
        return dst
    files = sorted(query_catalog(data_root, partition, [year], random_state=0))
    if limit:
        files = files[:limit]
    if not files:
        logging.info(f"no {partition} files for {year}")
        return None
    jobs = [(f, data_root) for f in files]
    rows = []
    if workers > 1:
        with Pool(workers) as pool:
            for row in tqdm(
                pool.imap(_worker, jobs, chunksize=16), total=len(jobs), desc=f"{partition} {year}"
            ):
                rows.append(row)
    else:
        rows = [_worker(j) for j in tqdm(jobs, desc=f"{partition} {year}")]
    df = pd.DataFrame(rows)
    errors = int(df["error"].notna().sum()) if "error" in df else 0
    if errors:
        logging.warning(f"{errors} files failed; see the error column")
    os.makedirs(out, exist_ok=True)
    df.to_csv(dst + ".tmp", index=False)
    os.replace(dst + ".tmp", dst)
    logging.info(f"wrote {dst}: {len(df)} rows")
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", default=os.environ.get("TORNET_ROOT"))
    parser.add_argument(
        "--years", type=int, nargs="*", default=None, help="default: all in catalog"
    )
    parser.add_argument("--partitions", nargs="*", default=["train", "test"])
    parser.add_argument(
        "--out", default="radar_metrics", help="relative paths are under TORNET_ROOT"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="files per year (smoke tests)")
    args = parser.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(args.data_root, args.out)
    years = args.years
    if not years:
        catalog = pd.read_csv(
            os.path.join(args.data_root, "catalog.csv"), parse_dates=["start_time"]
        )
        years = sorted(catalog["start_time"].dt.year.unique().tolist())
    for year in years:
        for partition in args.partitions:
            run(args.data_root, partition, year, out, args.workers, args.limit)


if __name__ == "__main__":
    main()
