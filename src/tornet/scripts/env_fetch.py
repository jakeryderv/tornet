"""Fetch near-storm HRRR environment fields for every sample in the radar-metrics tables.

Reads <radar_dir>/*_radar.csv (tornet-radar-metrics) for each sample's storm location and
newest scan time, groups samples by the hour nearest that time, and for each hour fetches the
HRRR run initialised `fxx` hours earlier (see tornet.env.hrrr for the fields and the leakage
argument). Appends one row per sample to <out>/env_hrrr.csv as hours complete and records the
hours done, so an interrupted run resumes. Hours before the archive begins (2014-07-30) and hours
whose file is missing produce rows with env_available = 0 and NaN fields.

Usage: tornet-env-fetch [--radar_dir radar_metrics] [--out env_hrrr] [--fxx 2] [--halo 4]
                        [--workers 8] [--years ...] [--limit N hours]
"""

import argparse
import glob
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tornet.env.hrrr import FIELDS, HRRRFetcher, env_columns, extract, nearest_hour

logging.basicConfig(level=logging.INFO)

_FETCHER: HRRRFetcher | None = None


def _init(fxx: int, save_dir: str):
    global _FETCHER
    _FETCHER = HRRRFetcher(fxx=fxx, save_dir=save_dir)


def _hour(job):
    """(valid hour, filenames, lats, lons, halo) -> (valid hour, DataFrame of rows)."""
    valid, filenames, lats, lons, halo = job
    n = len(filenames)
    rows = {"filename": filenames, "env_valid": [valid.isoformat()] * n}
    fields = None
    try:
        fields = _FETCHER.fields(valid)
    except Exception as e:  # noqa: BLE001 - a bad hour is recorded, not fatal
        logging.warning(f"{valid}: {e!r}")
    if not fields:
        rows["env_available"] = np.zeros(n, dtype=int)
        rows["env_dist_km"] = np.full(n, np.nan)
        for c in env_columns():
            rows[c] = np.full(n, np.nan, dtype=np.float32)
        return valid, pd.DataFrame(rows)
    r, c, dist = _FETCHER.index.query(lats, lons)
    rows["env_available"] = np.full(n, int(all(f in fields for f in FIELDS)))
    rows["env_dist_km"] = dist
    rows.update(extract(fields, r, c, halo))
    return valid, pd.DataFrame(rows)


def load_samples(radar_dir: str, years: list[int] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(radar_dir, "*_radar.csv"))):
        year = int(os.path.basename(path).split("_")[1])
        if years and year not in years:
            continue
        df = pd.read_csv(path)
        tcols = [c for c in df.columns if c.startswith("frame_time_")]
        df["time"] = df[tcols].max(axis=1)
        frames.append(df[["filename", "center_lat", "center_lon", "time"]])
    if not frames:
        raise FileNotFoundError(f"no *_radar.csv under {radar_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["center_lat"]) & np.isfinite(df["center_lon"])]
    df["valid"] = pd.to_datetime(df["time"], unit="s").map(nearest_hour)
    return df


def run(radar_dir, out, fxx, halo, workers, years=None, limit=None, save_dir=None):
    os.makedirs(out, exist_ok=True)
    rows_path, done_path = os.path.join(out, "env_hrrr.csv"), os.path.join(out, "hours_done.txt")
    samples = load_samples(radar_dir, years)
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = {line.strip() for line in f if line.strip()}
    jobs = []
    for valid, g in samples.groupby("valid"):
        if valid.isoformat() in done:
            continue
        jobs.append(
            (
                valid,
                g["filename"].tolist(),
                g["center_lat"].to_numpy(float),
                g["center_lon"].to_numpy(float),
                halo,
            )
        )
    jobs.sort(key=lambda j: j[0])
    if limit:
        jobs = jobs[:limit]
    logging.info(
        f"{len(samples)} samples in {samples['valid'].nunique()} hours; {len(done)} hours done, "
        f"{len(jobs)} to fetch with {workers} workers (fxx={fxx}, halo={halo})"
    )
    if not jobs:
        return rows_path
    header = not os.path.exists(rows_path)
    t0, n_done, n_missing = time.time(), 0, 0
    save_dir = save_dir or os.path.join(out, "grib_tmp")
    with (
        Pool(workers, initializer=_init, initargs=(fxx, save_dir)) as pool,
        open(done_path, "a") as done_f,
    ):
        for valid, df in pool.imap_unordered(_hour, jobs, chunksize=1):
            df.to_csv(rows_path, mode="a", header=header, index=False)
            header = False
            done_f.write(valid.isoformat() + "\n")
            done_f.flush()
            n_done += 1
            n_missing += int(df["env_available"].iloc[0] == 0)
            if n_done % 50 == 0 or n_done == len(jobs):
                rate = n_done / (time.time() - t0)
                logging.info(
                    f"{n_done}/{len(jobs)} hours ({n_missing} without data), {rate * 3600:.0f}/h, "
                    f"eta {(len(jobs) - n_done) / max(rate, 1e-9) / 60:.0f} min"
                )
    return rows_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", default=os.environ.get("TORNET_ROOT"))
    parser.add_argument("--radar_dir", default="radar_metrics")
    parser.add_argument("--out", default="env_hrrr")
    parser.add_argument("--fxx", type=int, default=2, help="forecast hour (run age at the sample)")
    parser.add_argument("--halo", type=int, default=4, help="box half-width in grid cells (3 km)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="hours to fetch (smoke tests)")
    args = parser.parse_args()
    rel = lambda p: p if os.path.isabs(p) else os.path.join(args.data_root, p)  # noqa: E731
    run(
        rel(args.radar_dir),
        rel(args.out),
        args.fxx,
        args.halo,
        args.workers,
        args.years,
        args.limit,
    )


if __name__ == "__main__":
    main()
