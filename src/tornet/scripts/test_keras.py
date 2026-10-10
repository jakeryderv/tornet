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

import tornet.models.layers  # noqa: E402, F401  (registers CoordConv2D etc. for load_model)
from tornet import metrics as tfm  # noqa: E402
from tornet.data.loader import get_dataloader  # noqa: E402

logging.basicConfig(level=logging.INFO)


# Per-sample metadata carried alongside the model inputs so predictions can be scored offline.
META_KEYS = ["category", "ef_number", "event_id", "time", "rng_lower", "rng_upper"]
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


def predict_dataset(model, ds) -> pd.DataFrame:
    """Run the model over a KerasDataLoader and return one row per sample with its logit."""
    input_keys = list(model.input.keys())
    rows = []
    for i in range(len(ds)):
        x, y = ds[i]
        logits = keras.ops.convert_to_numpy(
            model({k: x[k] for k in input_keys}, training=False)
        ).reshape(-1)
        files = ds.file_list[i * ds.batch_size : i * ds.batch_size + len(logits)]
        meta = {k: np.asarray(x[k]).reshape(len(logits), -1)[:, 0] for k in META_KEYS}
        labels = np.asarray(y).reshape(-1)
        for j, f in enumerate(files):
            rows.append(
                {
                    "filename": os.path.relpath(f, ds.data_root),
                    "label": int(labels[j]),
                    "category": CATEGORY_NAMES[int(meta["category"][j])],
                    "ef_number": int(meta["ef_number"][j]),
                    "event_id": int(meta["event_id"][j]),
                    "time": int(meta["time"][j]),
                    "rng_center_km": float((meta["rng_lower"][j] + meta["rng_upper"][j]) / 2e3),
                    "logit": float(logits[j]),
                }
            )
        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            logging.info(f"predicted {i + 1}/{len(ds)} batches")
    df = pd.DataFrame(rows)
    df["prob"] = 1 / (1 + np.exp(-df["logit"]))
    df["year"] = pd.to_datetime(df["time"], unit="s").dt.year
    return df


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
        data_root, args.years, "test", args.batch_size, select_keys=input_keys + META_KEYS
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
