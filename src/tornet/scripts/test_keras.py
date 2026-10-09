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

# Keras must see the backend before it is imported anywhere, including via tornet.metrics.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402

from tornet.data.loader import get_dataloader  # noqa: E402
from tornet.metrics.keras import metrics as tfm  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Pretrained model to test (.keras)", default=None)
    parser.add_argument(
        "--dataloader",
        help="Which data loader to use for loading test data",
        default="keras",
        choices=["keras", "tensorflow", "tensorflow-tfds", "torch", "torch-tfds"],
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2013, 2023)),
        help="Test years to evaluate on (default: all, 2013-2022)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", help="Optional path to write scores as JSON", default=None)
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

    dataloader = args.dataloader

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f"Using {dataloader} dataloader")

    if ("tfds" in dataloader) and ("TFDS_DATA_DIR" in os.environ):
        logging.info("Using TFDS dataset location at " + os.environ["TFDS_DATA_DIR"])

    # load model
    model = keras.saving.load_model(trained_model, compile=False)

    ## Set up data loader
    logging.info(f"Evaluating on test years {args.years}")
    ds_test = get_dataloader(
        dataloader,
        data_root,
        args.years,
        "test",
        args.batch_size,
        select_keys=list(model.input.keys()),
    )

    # Compute various metrics
    from_logits = True
    metrics = [
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        keras.metrics.AUC(from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits=from_logits, name="BinaryAccuracy"),
        tfm.Precision(from_logits=from_logits, name="Precision"),
        tfm.Recall(from_logits=from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]
    model.compile(metrics=metrics)

    scores = model.evaluate(ds_test)
    scores = {m.name: float(scores[k + 1]) for k, m in enumerate(metrics)}
    scores["model_path"] = str(trained_model)
    scores["years"] = list(args.years)

    logging.info(scores)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2)
        logging.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
