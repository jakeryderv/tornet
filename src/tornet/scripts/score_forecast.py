"""Score a trained forecaster through the frozen detector, horizon by horizon.

For every test sample and horizon the frozen detector sees three scans: the last observed one
(persistence, the floor), the forecast (the result), and the real scan at that horizon (the
ceiling). Each gives a tornado logit, written to predictions_<which>_h<k>.csv in the format
tornet-score takes, with the label of that horizon's scan. summary.json carries, per horizon,
the forecast and persistence error against the real scan (MAE in the detector's normalised
units), the detector's scores on each of the three, and paired bootstraps of forecast minus
persistence and truth minus persistence. Lead per horizon is the median minutes from the last
observed scan.

Usage: tornet-score-forecast --model_path experiments/<run>/checkpoints/best.keras --years 2014
                             [--out <dir>] [--batch_size 8] [--workers 8] [--bootstrap 1000]
"""

import argparse
import gc
import json
import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from tornet.data.forecast import ForecastLoader, denormalize_fields  # noqa: E402
from tornet.data.keras_loader import iterate_batches  # noqa: E402
from tornet.models import forecaster as _forecaster  # noqa: E402, F401  (registers custom layers)
from tornet.scripts.score import paired_difference, score_file  # noqa: E402
from tornet.scripts.test_keras import CATEGORY_NAMES, finish_predictions  # noqa: E402
from tornet.scripts.train_keras import resolve_init_from  # noqa: E402

logging.basicConfig(level=logging.INFO)
WHICH = ("persistence", "forecast", "truth")


def detector_logits(
    detector, fields: np.ndarray, rfm: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """Tornado logits for scans given in the normalised representation."""
    x = denormalize_fields(fields)
    x["range_folded_mask"], x["coordinates"] = rfm, coords
    out = detector.predict_on_batch(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return np.asarray(out).reshape(-1)


def run(
    model_path,
    years,
    partition,
    out,
    batch_size,
    workers,
    bootstrap,
    seed,
    detector_path,
    data_root,
    max_files=None,
):
    forecaster = keras.saving.load_model(model_path, compile=False)
    detector = keras.saving.load_model(resolve_init_from(detector_path), compile=False)
    n_in = int(forecaster.input["DBZ"].shape[1])
    horizons = int(forecaster.output["fields"].shape[1])
    ds = ForecastLoader(
        data_root,
        partition,
        years,
        batch_size=batch_size,
        n_in=n_in,
        horizons=horizons,
        workers=workers,
        use_multiprocessing=workers > 1,
    )
    if max_files:
        ds.file_list = ds.file_list[:max_files]
    rows = {w: [[] for _ in range(horizons)] for w in WHICH}
    err = {"forecast": [[] for _ in range(horizons)], "persistence": [[] for _ in range(horizons)]}
    minutes = [[] for _ in range(horizons)]
    files_seen = 0
    for i, (x, y) in enumerate(iterate_batches(ds)):
        model_x = {k: x[k] for k in ForecastLoader.model_input_keys()}
        pred = forecaster.predict_on_batch(model_x)
        fields = np.asarray(pred["fields"] if isinstance(pred, dict) else pred)
        rfm, coords = x["range_folded_mask"][:, -1], x["coordinates"]
        n = len(x["category"])
        files = ds.file_list[files_seen : files_seen + n]
        files_seen += n
        for h in range(horizons):
            scans = {
                "persistence": x["last_fields"],
                "forecast": fields[:, h],
                "truth": y["fields"][:, h],
            }
            logits = {w: detector_logits(detector, s, rfm, coords) for w, s in scans.items()}
            err["forecast"][h].append(
                np.abs(scans["forecast"] - scans["truth"]).mean(axis=(1, 2, 3))
            )
            err["persistence"][h].append(
                np.abs(scans["persistence"] - scans["truth"]).mean(axis=(1, 2, 3))
            )
            minutes[h].append(y["minutes"][:, h])
            for j, f in enumerate(files):
                base = {
                    "filename": os.path.relpath(f, data_root),
                    "label": int(y["logits"][j, h]),
                    "category": CATEGORY_NAMES[int(x["category"][j])],
                    "ef_number": int(x["ef_number"][j]),
                    "event_id": int(x["event_id"][j]),
                    "time": int(y["time"][j, h]),
                    "rng_center_km": float((x["rng_lower"][j] + x["rng_upper"][j]) / 2e3),
                    "minutes": float(y["minutes"][j, h]),
                }
                for w in WHICH:
                    rows[w][h].append({**base, "logit": float(logits[w][j])})
        if i % 50 == 49:
            gc.collect()
            logging.info(f"{files_seen} files")

    os.makedirs(out, exist_ok=True)
    summary = {"model_path": model_path, "years": list(years), "n_in": n_in, "horizons": []}
    for h in range(horizons):
        preds = {w: finish_predictions(rows[w][h]) for w in WHICH}
        for w in WHICH:
            preds[w].to_csv(os.path.join(out, f"predictions_{w}_h{h}.csv"), index=False)
        entry = {
            "horizon": h,
            "lead_minutes_median": float(np.median(np.concatenate(minutes[h]))),
            "mae_forecast": float(np.concatenate(err["forecast"][h]).mean()),
            "mae_persistence": float(np.concatenate(err["persistence"][h]).mean()),
            "n": int(len(preds["forecast"])),
            "positives": int(preds["forecast"]["label"].sum()),
            "scores": {w: score_file(preds[w], bootstrap, seed)["views"] for w in WHICH},
            "forecast_minus_persistence": paired_difference(
                preds["persistence"], preds["forecast"], bootstrap, seed
            ),
            "truth_minus_persistence": paired_difference(
                preds["persistence"], preds["truth"], bootstrap, seed
            ),
        }
        summary["horizons"].append(entry)
        auc = {w: entry["scores"][w]["NC"]["auc"] for w in WHICH}
        logging.info(
            f"h{h} lead {entry['lead_minutes_median']:.0f} min: MAE forecast {entry['mae_forecast']:.4f} "
            f"vs persistence {entry['mae_persistence']:.4f}; detector NC AUC "
            f"persistence {auc['persistence']:.4f} forecast {auc['forecast']:.4f} truth {auc['truth']:.4f}"
        )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--out", default=None, help="default: <run dir>/forecast_scores")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detector", default="pretrained")
    parser.add_argument("--max_files", type=int, default=None, help="smoke tests")
    args = parser.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(args.model_path)), "forecast_scores"
    )
    run(
        args.model_path,
        args.years,
        args.partition,
        out,
        args.batch_size,
        args.workers,
        args.bootstrap,
        args.seed,
        args.detector,
        os.environ["TORNET_ROOT"],
        args.max_files,
    )


if __name__ == "__main__":
    main()
