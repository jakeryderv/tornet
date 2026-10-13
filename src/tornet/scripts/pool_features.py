"""Pool a feature cache to one short vector per scan, for models that take the backbone's
features as plain inputs (tornet-meta-model) rather than through a head.

For every <partition>_<year>_features.npy written by tornet-cache-features (rank 4 or 5), writes
<out>/<partition>_<year>_pooled.npy of shape (N, 2C) or (N, T, 2C): the global max over the
feature map followed by the global mean, float16, plus a copy of the metadata CSV. The whole
dataset pools to a few hundred megabytes, so the export moves off the pod and the rest of the
meta-model work runs locally.

Usage: tornet-pool-features [--features_dir features_4f] [--out pooled_4f] [--chunk 2048]
"""

import argparse
import glob
import json
import logging
import os
import shutil

import numpy as np

logging.basicConfig(level=logging.INFO)


def pool_chunk(x: np.ndarray) -> np.ndarray:
    """(n, [T,] H, W, C) -> (n, [T,] 2C): max over H, W, then mean."""
    x = x.astype(np.float32)
    axes = (-3, -2)
    return np.concatenate([x.max(axis=axes), x.mean(axis=axes)], axis=-1)


def pool_file(src: str, dst: str, chunk: int) -> tuple:
    feats = np.load(src, mmap_mode="r")
    out = None
    for i in range(0, len(feats), chunk):
        p = pool_chunk(np.asarray(feats[i : i + chunk])).astype(np.float16)
        if out is None:
            out = np.empty((len(feats),) + p.shape[1:], dtype=np.float16)
        out[i : i + len(p)] = p
    np.save(dst, out)
    return out.shape


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.environ["TORNET_ROOT"], path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features_dir", default="features_4f")
    parser.add_argument("--out", default="pooled_4f")
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    src_dir, out = resolve(args.features_dir), resolve(args.out)
    os.makedirs(out, exist_ok=True)
    shapes = {}
    for src in sorted(glob.glob(os.path.join(src_dir, "*_features.npy"))):
        stem = os.path.basename(src)[: -len("_features.npy")]
        dst = os.path.join(out, f"{stem}_pooled.npy")
        meta_src, meta_dst = (
            os.path.join(src_dir, f"{stem}_meta.csv"),
            os.path.join(out, f"{stem}_meta.csv"),
        )
        if os.path.exists(dst) and os.path.exists(meta_dst):
            logging.info(f"{stem}: exists, skipping")
            shapes[stem] = list(np.load(dst, mmap_mode="r").shape)
            continue
        shapes[stem] = list(pool_file(src, dst, args.chunk))
        shutil.copyfile(meta_src, meta_dst)
        logging.info(f"{stem}: {shapes[stem]}")
    info = {"features_dir": src_dir, "pooling": ["max", "mean"], "shapes": shapes}
    cache_json = os.path.join(src_dir, "cache.json")
    if os.path.exists(cache_json):
        with open(cache_json) as f:
            info["cache"] = json.load(f)
    with open(os.path.join(out, "pooled.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
