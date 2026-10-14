"""Convert TorNet netCDF samples to Parquet shards for publishing on the Hugging Face Hub.

Layout under --out:
    data/<partition>/<year>/part-NNNNN-of-MMMMM.parquet   the samples, `shard_size` per file
                                                  (256 samples is ~170 MB with zstd)
    metadata/<partition>-<year>.parquet           one row per sample, scalar fields only
    catalog.parquet                               the metadata files merged (`catalog` command)

Each sample row keeps every field of the netCDF: the six products as (4, 120, 240, 2) float16
arrays in physical units (scan, azimuth, range, tilt), the range-folded mask as uint8 of the
same shape, the four per-scan labels and times, event metadata, azimuth and range limits, the
tornado start and end times (v1.1), plus the catalog's site, latitude and longitude. Arrays are
`datasets.Array4D` features, so `load_dataset("parquet", data_files=...)` restores them typed.

Usage:
    tornet-to-parquet convert --out DIR [--partitions train test] [--years 2013 ...]
                              [--shard-size 256] [--workers N]   (one shard per worker at a
                              time; a worker holds ~1.5 GB per 256 samples)
    tornet-to-parquet verify --out DIR [--years ...] [--samples N]
    tornet-to-parquet catalog --out DIR
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import read_file

SHAPE = (4, 120, 240, 2)
SCALARS = [
    "filename",
    "partition",
    "year",
    "category",
    "ef_number",
    "event_id",
    "episode_id",
    "site",
    "lat",
    "lon",
    "label",
    "frame_labels",
    "time",
    "az_lower",
    "az_upper",
    "rng_lower",
    "rng_upper",
    "tornado_start_time",
    "tornado_end_time",
]


def features():
    from datasets import Array4D, Features, Sequence, Value

    return Features(
        {
            **{v: Array4D(shape=SHAPE, dtype="float16") for v in ALL_VARIABLES},
            "range_folded_mask": Array4D(shape=SHAPE, dtype="uint8"),
            "filename": Value("string"),
            "partition": Value("string"),
            "year": Value("int16"),
            "category": Value("string"),
            "ef_number": Value("int8"),
            "event_id": Value("int64"),
            "episode_id": Value("int64"),
            "site": Value("string"),
            "lat": Value("float32"),
            "lon": Value("float32"),
            "label": Value("int8"),
            "frame_labels": Sequence(Value("int8"), length=4),
            "time": Sequence(Value("int64"), length=4),
            "az_lower": Value("float32"),
            "az_upper": Value("float32"),
            "rng_lower": Value("float32"),
            "rng_upper": Value("float32"),
            "tornado_start_time": Value("int64"),
            "tornado_end_time": Value("int64"),
        }
    )


def read_one(args: tuple[str, str, dict]) -> dict:
    root, filename, cat = args
    d = read_file(os.path.join(root, filename), variables=ALL_VARIABLES, n_frames=4)
    row = {v: d[v].astype(np.float16) for v in ALL_VARIABLES}
    for v in ALL_VARIABLES:
        if row[v].shape != SHAPE:
            raise ValueError(f"{filename}: {v} has shape {row[v].shape}, expected {SHAPE}")
    row["range_folded_mask"] = d["range_folded_mask"].astype(np.uint8)
    labels = d["frame_labels"][0].astype(np.int8)
    row.update(
        {
            "filename": filename,
            "partition": cat["type"],
            "year": int(cat["year"]),
            "category": {0: "TOR", 1: "NUL", 2: "WRN"}[int(d["category"][0])],
            "ef_number": int(d["ef_number"][0]),
            "event_id": int(d["event_id"][0]),
            "episode_id": int(cat["episode_id"]),
            "site": str(cat["site_name"]),
            "lat": float(cat["lat"]),
            "lon": float(cat["lon"]),
            "label": int(labels[-1]),
            "frame_labels": labels.tolist(),
            "time": d["time"].astype(np.int64).tolist(),
            "az_lower": float(d["az_lower"][0]),
            "az_upper": float(d["az_upper"][0]),
            "rng_lower": float(d["rng_lower"][0]),
            "rng_upper": float(d["rng_upper"][0]),
            "tornado_start_time": int(d["tornado_start_time"][0]),
            "tornado_end_time": int(d["tornado_end_time"][0]),
        }
    )
    return row


def _array4d(arrays: list[np.ndarray], feat) -> pa.ExtensionArray:
    """(N, *SHAPE) numpy -> the `datasets` ArrayXD extension array, built from flat buffers
    (nested ListArrays), about 10x faster than Dataset.from_dict."""
    x = np.ascontiguousarray(np.stack(arrays))
    storage = pa.array(x.reshape(-1))
    for dim in reversed(x.shape[1:]):
        count = len(storage) // dim
        offsets = pa.array(np.arange(0, count * dim + 1, dim, dtype=np.int32))
        storage = pa.ListArray.from_arrays(offsets, storage)
    return pa.ExtensionArray.from_storage(feat(), storage)


def build_table(rows: list[dict], feats) -> pa.Table:
    schema = feats.arrow_schema
    cols = []
    for name, feat in feats.items():
        if name in ALL_VARIABLES or name == "range_folded_mask":
            cols.append(_array4d([r[name] for r in rows], feat))
        else:
            cols.append(pa.array([r[name] for r in rows], type=schema.field(name).type))
    return pa.Table.from_arrays(cols, schema=schema)


def convert_shard(job: dict) -> list[dict]:
    """One worker, one shard: read its files, build the table, write it; returns metadata."""
    feats = features()
    rows = [read_one((job["root"], f, c)) for f, c in job["files"]]
    pq.write_table(build_table(rows, feats), job["path"], compression="zstd")
    meta = [{k: r[k] for k in SCALARS} | {"shard": job["shard"]} for r in rows]
    return meta


def convert(
    root: str, out: str, partitions: list[str], years: list[int], shard_size: int, workers: int
) -> None:
    catalog = pd.read_csv(os.path.join(root, "catalog.csv"), parse_dates=["start_time", "end_time"])
    catalog["year"] = catalog["start_time"].dt.year
    os.makedirs(os.path.join(out, "metadata"), exist_ok=True)
    jobs, groups = [], []
    for partition in partitions:
        for year in years:
            rows_cat = catalog[(catalog["type"] == partition) & (catalog["year"] == year)]
            rows_cat = rows_cat.sort_values("filename").reset_index(drop=True)
            if rows_cat.empty:
                logging.info("%s/%d: no files in the catalog, skipped", partition, year)
                continue
            data_dir = os.path.join(out, "data", partition, str(year))
            os.makedirs(data_dir, exist_ok=True)
            files = [(r["filename"], r.to_dict()) for _, r in rows_cat.iterrows()]
            n_shards = -(-len(files) // shard_size)
            groups.append((partition, year, len(files), n_shards, data_dir))
            for shard in range(n_shards):
                jobs.append(
                    {
                        "root": root,
                        "partition": partition,
                        "year": year,
                        "shard": shard,
                        "files": files[shard * shard_size : (shard + 1) * shard_size],
                        "path": os.path.join(
                            data_dir, f"part-{shard:05d}-of-{n_shards:05d}.parquet"
                        ),
                    }
                )
    logging.info("%d shards over %d partition-years, %d workers", len(jobs), len(groups), workers)
    meta: dict[tuple[str, int], list[dict]] = {}
    t0, done, n_files = time.time(), 0, 0
    with Pool(workers) as pool:
        for job, rows in zip(jobs, pool.imap(convert_shard, jobs), strict=True):
            meta.setdefault((job["partition"], job["year"]), []).extend(rows)
            done += 1
            n_files += len(rows)
            if done % 10 == 0 or done == len(jobs):
                logging.info(
                    "%d/%d shards, %d files, %.0f files/s",
                    done,
                    len(jobs),
                    n_files,
                    n_files / (time.time() - t0),
                )
    for partition, year, n, n_shards, data_dir in groups:
        pd.DataFrame(meta[(partition, year)]).to_parquet(
            os.path.join(out, "metadata", f"{partition}-{year}.parquet"), index=False
        )
        size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
        logging.info(
            "%s/%d: %d files, %d shards, %.2f GB", partition, year, n, n_shards, size / 1e9
        )


def verify(root: str, out: str, years: list[int], samples: int, seed: int = 0) -> None:
    """Every converted partition-year: row count equals the catalog's, shard files are all
    there, and `samples` random rows read back through `datasets` equal the netCDF exactly."""
    from datasets import load_dataset

    catalog = pd.read_csv(os.path.join(root, "catalog.csv"), parse_dates=["start_time", "end_time"])
    catalog["year"] = catalog["start_time"].dt.year
    rng = np.random.default_rng(seed)
    checked, problems = 0, []
    for partition in ("train", "test"):
        for year in years:
            meta_path = os.path.join(out, "metadata", f"{partition}-{year}.parquet")
            expected = int(((catalog["type"] == partition) & (catalog["year"] == year)).sum())
            if not os.path.exists(meta_path):
                if expected:
                    problems.append(
                        f"{partition}/{year}: not converted ({expected} files in catalog)"
                    )
                continue
            meta = pd.read_parquet(meta_path)
            data_dir = os.path.join(out, "data", partition, str(year))
            shards = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
            n_shards = int(meta["shard"].max()) + 1
            if len(meta) != expected or len(shards) != n_shards:
                problems.append(
                    f"{partition}/{year}: {len(meta)} rows vs {expected} in catalog, {len(shards)} shards vs {n_shards}"
                )
                continue
            for idx in rng.choice(len(meta), size=min(samples, len(meta)), replace=False):
                row = meta.iloc[int(idx)]
                shard = os.path.join(data_dir, shards[int(row["shard"])])
                ds = load_dataset("parquet", data_files=shard, split="train").with_format("numpy")
                i = list(ds["filename"]).index(row["filename"])
                got = ds[i]
                ref = read_file(
                    os.path.join(root, row["filename"]), variables=ALL_VARIABLES, n_frames=4
                )
                ok = all(
                    np.array_equal(
                        got[v].astype(np.float16), ref[v].astype(np.float16), equal_nan=True
                    )
                    for v in ALL_VARIABLES
                )
                ok = ok and np.array_equal(
                    got["range_folded_mask"], ref["range_folded_mask"].astype(np.uint8)
                )
                ok = ok and list(got["frame_labels"]) == ref["frame_labels"][0].tolist()
                ok = ok and list(got["time"]) == ref["time"].tolist()
                checked += 1
                if not ok:
                    problems.append(
                        f"{partition}/{year}: {row['filename']} differs from the netCDF"
                    )
            logging.info("%s/%d ok: %d rows, %d shards", partition, year, len(meta), n_shards)
    if problems:
        for p in problems:
            logging.error(p)
        raise SystemExit(f"verify failed: {len(problems)} problems")
    logging.info("verify ok: %d samples compared exactly", checked)


def merge_catalog(out: str) -> None:
    files = sorted(os.listdir(os.path.join(out, "metadata")))
    df = pd.concat(
        [pd.read_parquet(os.path.join(out, "metadata", f)) for f in files], ignore_index=True
    )
    df.to_parquet(os.path.join(out, "catalog.parquet"), index=False)
    logging.info("catalog.parquet: %d rows from %d files", len(df), len(files))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--root", default=os.environ.get("TORNET_ROOT"))
    c.add_argument("--out", required=True)
    c.add_argument("--partitions", nargs="+", default=["train", "test"])
    c.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2023)))
    c.add_argument("--shard-size", type=int, default=256)
    c.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    v = sub.add_parser("verify")
    v.add_argument("--root", default=os.environ.get("TORNET_ROOT"))
    v.add_argument("--out", required=True)
    v.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2023)))
    v.add_argument(
        "--samples", type=int, default=5, help="random rows per partition-year to compare"
    )
    m = sub.add_parser("catalog")
    m.add_argument("--out", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.command == "convert":
        if not a.root:
            raise SystemExit("set --root or TORNET_ROOT")
        convert(a.root, a.out, a.partitions, a.years, a.shard_size, a.workers)
    elif a.command == "verify":
        if not a.root:
            raise SystemExit("set --root or TORNET_ROOT")
        verify(a.root, a.out, a.years, a.samples)
    else:
        merge_catalog(a.out)


if __name__ == "__main__":
    main()
