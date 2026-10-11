"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

# Keras must see the backend before it is imported anywhere, including via tornet.metrics.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402

import tornet.models.heads  # noqa: E402, F401  (registers the head layers for load_model)
import tornet.models.layers  # noqa: E402, F401  (registers CoordConv2D etc. for load_model)
import tornet.models.optimizers  # noqa: E402, F401  (registers ScaledAdam saved in checkpoints)
from tornet import metrics as tfm  # noqa: E402
from tornet.data.keras_loader import iterate_batches  # noqa: E402
from tornet.data.loader import get_dataloader  # noqa: E402

logging.basicConfig(level=logging.INFO)


# Per-sample metadata carried alongside the model inputs so predictions can be scored offline.
META_KEYS = [
    "category",
    "ef_number",
    "event_id",
    "time",
    "rng_lower",
    "rng_upper",
    "tornado_start_time",
    "tornado_end_time",
    "frame_labels",
]
CATEGORY_NAMES = {0: "TOR", 1: "NUL", 2: "WRN"}


def build_metrics(from_logits=True):
    return [
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        keras.metrics.AUC(from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits=from_logits, name="BinaryAccuracy"),
        tfm.Precision(from_logits=from_logits, name="Precision"),
        tfm.Recall(from_logits=from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]


def batch_metadata(x, y, files, data_root) -> list[dict]:
    """One metadata row per sample in a loader batch, for predictions and feature caches."""
    n = len(files)
    meta = {k: np.asarray(x[k]).reshape(n, -1)[:, 0] for k in META_KEYS if k != "frame_labels"}
    frame_labels = np.asarray(x["frame_labels"]).reshape(n, -1)
    labels = np.asarray(y).reshape(-1)
    rows = []
    for j, f in enumerate(files):
        rows.append(
            {
                "filename": os.path.relpath(f, data_root),
                "label": int(labels[j]),
                "category": CATEGORY_NAMES[int(meta["category"][j])],
                "ef_number": int(meta["ef_number"][j]),
                "event_id": int(meta["event_id"][j]),
                "time": int(meta["time"][j]),
                "rng_center_km": float((meta["rng_lower"][j] + meta["rng_upper"][j]) / 2e3),
                # Minutes from this frame to tornado onset / since tornado end (v1.1 fields);
                # NaN for samples without a tornado.
                "minutes_to_start": (float(meta["tornado_start_time"][j] - meta["time"][j]) / 60)
                if meta["tornado_start_time"][j] > 0
                else float("nan"),
                "minutes_since_end": (float(meta["time"][j] - meta["tornado_end_time"][j]) / 60)
                if meta["tornado_end_time"][j] > 0
                else float("nan"),
                "frame_fraction": float(frame_labels[j].mean()),
            }
        )
    return rows


def finish_predictions(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["prob"] = 1 / (1 + np.exp(-df["logit"]))
    df["year"] = pd.to_datetime(df["time"], unit="s").dt.year
    return df


def predict_dataset(model, ds) -> pd.DataFrame:
    """Run the model over a KerasDataLoader and return one row per sample with its logit."""
    input_keys = list(model.input.keys())
    rows = []
    for i, (x, y) in enumerate(iterate_batches(ds)):
        logits = np.asarray(model.predict_on_batch({k: x[k] for k in input_keys})).reshape(-1)
        files = ds.file_list[i * ds.batch_size : i * ds.batch_size + len(logits)]
        for row, logit in zip(batch_metadata(x, y, files, ds.data_root), logits, strict=True):
            rows.append({**row, "logit": float(logit)})
        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            logging.info(f"predicted {i + 1}/{len(ds)} batches")
    return finish_predictions(rows)


def score_predictions(df: pd.DataFrame) -> dict[str, float]:
    """The same six metrics model.evaluate reported, computed from saved logits."""
    y = df["label"].to_numpy(dtype=np.float32)[:, None]
    logits = df["logit"].to_numpy(dtype=np.float32)[:, None]
    scores = {}
    for m in build_metrics():
        m.update_state(y, logits)
        scores[m.name] = float(m.result())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Pretrained model to test (.keras)", default=None)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2013, 2023)),
        help="Test years to evaluate on (default: all, 2013-2022)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8, help="Loader worker processes")
    parser.add_argument("--output", help="Optional path to write scores as JSON", default=None)
    parser.add_argument(
        "--predictions",
        help="Optional CSV path for per-sample logits and metadata; score it with tornet-score",
        default=None,
    )
    args = parser.parse_args()

    data_root = os.environ["TORNET_ROOT"]
    logging.info("TORNET_ROOT=" + data_root)

    trained_model = args.model_path
    if trained_model is None:
        # download model from hugging face
        # alternatively, you can manually download the file
        # from https://huggingface.co/tornet-ml/tornado_detector_baseline_v1
        # and point to it using --model_path
        from huggingface_hub import hf_hub_download

        trained_model = hf_hub_download(
            repo_id="tornet-ml/tornado_detector_baseline_v1",
            filename="tornado_detector_baseline.keras",
        )

    logging.info(f"Using {keras.config.backend()} backend")

    model = keras.saving.load_model(trained_model, compile=False)
    input_keys = list(model.input.keys())
    logging.info(f"Evaluating on test years {args.years}")

    # Predict once, keep the logits, derive every metric from them.
    ds = get_dataloader(
        data_root,
        args.years,
        "test",
        args.batch_size,
        select_keys=input_keys + META_KEYS,
        workers=args.workers,
        use_multiprocessing=args.workers > 1,
    )
    preds = predict_dataset(model, ds)
    scores = score_predictions(preds)
    if args.predictions:
        preds.to_csv(args.predictions, index=False)
        logging.info(f"wrote {len(preds)} predictions to {args.predictions}")

    scores["model_path"] = str(trained_model)
    scores["years"] = list(args.years)
    logging.info(scores)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2)
        logging.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
