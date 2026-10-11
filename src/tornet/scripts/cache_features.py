"""Cache backbone feature maps so classification heads can be trained in seconds per epoch.

Cuts a trained model at the last dropout layer of its convolutional backbone (block 4), runs every
sample of the requested years and partitions through it, and writes one float16 .npy of feature
maps plus one metadata CSV per (partition, year) into the output directory:

    <out>/<partition>_<year>_features.npy   (N, H, W, C) float16
    <out>/<partition>_<year>_meta.csv       filename, label, category, ef_number, event_id, time, ...
    <out>/cache.json                        model path, tap layer, feature shape

With --verify <predictions.csv>, the model's own head is applied to the cached features and the
resulting logits are compared with those in the CSV (from `tornet-test --predictions`), which
checks that the cache reproduces the full model.

Usage: tornet-cache-features --years 2013 2014 2015 [--partitions train test] [--out DIR]
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402

import tornet.models.heads  # noqa: E402, F401  (registers custom layers for load_model)
import tornet.models.layers  # noqa: E402, F401
import tornet.models.optimizers  # noqa: E402, F401
import tornet.models.temporal  # noqa: E402, F401
from tornet.data.keras_loader import iterate_batches  # noqa: E402
from tornet.data.loader import get_dataloader  # noqa: E402
from tornet.models.cnn_baseline import backbone_end  # noqa: E402
from tornet.scripts.score import roc_auc  # noqa: E402
from tornet.scripts.test_keras import META_KEYS, batch_metadata  # noqa: E402

logging.basicConfig(level=logging.INFO)


def split_backbone(model: keras.Model) -> tuple[keras.Model, keras.Model]:
    """Return (backbone, head) models: the backbone ends at its `features` layer (the last Dropout
    for upstream checkpoints); the head re-applies every layer after it to an input of the
    backbone's output shape. Rebuilding the head this way only works for the sequential `maxpool`
    head, which is all `--verify` needs."""
    end = backbone_end(model)
    tap = model.layers[end]
    backbone = keras.Model(inputs=model.input, outputs=tap.output, name="backbone")
    feature_shape = tuple(tap.output.shape[1:])
    head_input = keras.Input(feature_shape, name="features")
    y = head_input
    for layer in model.layers[end + 1 :]:
        y = layer(y)
    head = keras.Model(inputs=head_input, outputs=y, name="head")
    return backbone, head


def cache_partition(backbone, data_root, year, partition, out, batch_size, workers):
    ds = get_dataloader(
        data_root,
        [year],
        partition,
        batch_size,
        select_keys=list(backbone.input.keys()) + META_KEYS,
        workers=workers,
        use_multiprocessing=workers > 1,
    )
    if len(ds) == 0:
        logging.info(f"{partition} {year}: no files, skipping")
        return None
    input_keys = list(backbone.input.keys())
    feats, rows = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(iterate_batches(ds)):
        f = np.asarray(backbone.predict_on_batch({k: x[k] for k in input_keys}))
        feats.append(f.astype(np.float16))
        files = ds.file_list[i * ds.batch_size : i * ds.batch_size + len(f)]
        rows.extend(batch_metadata(x, y, files, ds.data_root))
        if (i + 1) % 20 == 0 or i + 1 == len(ds):
            logging.info(f"{partition} {year}: {i + 1}/{len(ds)} batches, {time.time() - t0:.0f}s")
    features = np.concatenate(feats)
    meta = pd.DataFrame(rows)
    meta["year"] = year
    np.save(os.path.join(out, f"{partition}_{year}_features.npy"), features)
    meta.to_csv(os.path.join(out, f"{partition}_{year}_meta.csv"), index=False)
    logging.info(f"{partition} {year}: wrote {features.shape} ({features.nbytes / 1e9:.2f} GB)")
    return features.shape[1:]


def verify(head, out, years, predictions_csv):
    """Apply the model's own head to cached test features and compare with saved full-model logits."""
    ref = pd.read_csv(predictions_csv).set_index("filename")
    diffs, y_all, p_all = [], [], []
    for year in years:
        fpath = os.path.join(out, f"test_{year}_features.npy")
        if not os.path.exists(fpath):
            continue
        feats = np.load(fpath, mmap_mode="r")
        meta = pd.read_csv(os.path.join(out, f"test_{year}_meta.csv"))
        logits = np.concatenate(
            [
                np.asarray(
                    head.predict_on_batch(np.asarray(feats[i : i + 256], dtype=np.float32))
                ).reshape(-1)
                for i in range(0, len(meta), 256)
            ]
        )
        common = meta["filename"].isin(ref.index)
        diffs.append(np.abs(logits[common] - ref.loc[meta["filename"][common], "logit"].to_numpy()))
        y_all.append(meta["label"].to_numpy()[common])
        p_all.append(logits[common])
    diffs = np.concatenate(diffs)
    auc = roc_auc(np.concatenate(y_all), np.concatenate(p_all))
    logging.info(
        f"verify: {len(diffs)} samples, max |logit diff| {diffs.max():.4f}, "
        f"mean {diffs.mean():.5f}, AUC from cached features {auc:.4f}"
    )
    return {"n": int(len(diffs)), "max_abs_logit_diff": float(diffs.max()), "auc": float(auc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_path", default=None, help="Model to cut (default: pretrained)")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2013, 2023)))
    parser.add_argument("--partitions", nargs="+", default=["train", "test"])
    parser.add_argument("--out", default=None, help="Output dir (default: $TORNET_ROOT/features)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", default=None, help="predictions CSV from tornet-test")
    args = parser.parse_args()

    data_root = os.environ["TORNET_ROOT"]
    out = args.out or os.path.join(data_root, "features")
    os.makedirs(out, exist_ok=True)

    model_path = args.model_path
    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="tornet-ml/tornado_detector_baseline_v1",
            filename="tornado_detector_baseline.keras",
        )
    model = keras.saving.load_model(model_path, compile=False)
    backbone, head = split_backbone(model)
    logging.info(
        f"tap layer: {backbone.layers[-1].name}, feature shape {backbone.output.shape[1:]}"
    )

    feature_shape = None
    for partition in args.partitions:
        for year in args.years:
            shape = cache_partition(
                backbone, data_root, year, partition, out, args.batch_size, args.workers
            )
            feature_shape = feature_shape or shape

    info = {
        "model_path": str(model_path),
        "tap_layer": backbone.layers[-1].name,
        "feature_shape": list(feature_shape) if feature_shape else None,
        "years": args.years,
        "partitions": args.partitions,
    }
    if args.verify:
        info["verify"] = verify(head, out, args.years, args.verify)
    with open(os.path.join(out, "cache.json"), "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"wrote {os.path.join(out, 'cache.json')}")


if __name__ == "__main__":
    main()
