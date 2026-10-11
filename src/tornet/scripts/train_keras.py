"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import copy
import json
import logging
import os
import shutil
import sys

import numpy as np

# Keras must see the backend before it is imported anywhere, including via tornet.metrics.
os.environ.setdefault("KERAS_BACKEND", "torch")
# Let the CUDA caching allocator grow segments instead of fragmenting: the last, smaller batch of
# an epoch changes tensor shapes, and without this a 12 GB GPU ran out of memory at 7.75 GB live.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import keras  # noqa: E402

from tornet import metrics as tfm  # noqa: E402
from tornet.data.constants import ALL_VARIABLES  # noqa: E402
from tornet.data.loader import get_dataloader  # noqa: E402
from tornet.data.preprocess import get_shape  # noqa: E402
from tornet.models.cnn_baseline import (  # noqa: E402
    backbone_layers,
    build_model,
    load_backbone_weights,
)
from tornet.models.losses import mae_loss  # noqa: E402
from tornet.models.optimizers import ScaledAdam, scales_for  # noqa: E402
from tornet.utils.general import (  # noqa: E402
    CollectGarbage,
    git_revision,
    make_callback_dirs,
    make_exp_dir,
)

logging.basicConfig(level=logging.INFO)

EXP_DIR = os.environ.get("EXP_DIR", ".")

DEFAULT_CONFIG = {
    "epochs": 10,
    "input_variables": ALL_VARIABLES,
    "train_years": list(range(2013, 2021)),
    "val_years": list(range(2021, 2023)),
    "batch_size": 128,
    "model": "vgg",
    "start_filters": 48,
    "learning_rate": 1e-4,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-5,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0,
    "loss": "cce",
    "head": "maxpool",
    "head_kwargs": {},
    # Trailing radar scans per sample. 1 is the baseline; more needs a temporal head.
    "frames": 1,
    # Path to a .keras model, or "pretrained" for the authors' Hugging Face weights, whose backbone
    # initialises this model's backbone. The head always starts fresh. None trains from scratch.
    "init_from": None,
    # Learning-rate multiplier for the backbone (1.0: same rate as the head; 0: frozen backbone).
    "backbone_lr_scale": 1.0,
    "exp_name": "tornet_baseline",
    "exp_dir": EXP_DIR,
    "dataloader_kwargs": {},
    # False keeps only checkpoints/best.keras (by val AUC) and checkpoints/last.keras
    "save_all_checkpoints": False,
    # Seeds python, numpy and the backend. GPU kernels may still be nondeterministic.
    "seed": 1234,
}


def resolve_init_from(init_from: str) -> str:
    """A .keras path, or "pretrained" for the authors' published weights."""
    if init_from != "pretrained":
        return init_from
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="tornet-ml/tornado_detector_baseline_v1",
        filename="tornado_detector_baseline.keras",
    )


def main(config):
    DATA_ROOT = os.environ["TORNET_ROOT"]
    logging.info("TORNET_ROOT=" + DATA_ROOT)

    # Gather all hyperparams
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    start_filters = config.get("start_filters")
    learning_rate = config.get("learning_rate")
    decay_steps = config.get("decay_steps")
    decay_rate = config.get("decay_rate")
    l2_reg = config.get("l2_reg")
    wN = config.get("wN")
    w0 = config.get("w0")
    w1 = config.get("w1")
    w2 = config.get("w2")
    wW = config.get("wW")
    head = config.get("head")
    head_kwargs = copy.deepcopy(config.get("head_kwargs") or {})
    frames = int(config.get("frames") or 1)
    init_from = config.get("init_from")
    backbone_lr_scale = float(config.get("backbone_lr_scale", 1.0))
    label_smooth = config.get("label_smooth")
    loss_fn = config.get("loss")
    input_variables = config.get("input_variables")
    exp_name = config.get("exp_name")
    exp_dir = config.get("exp_dir")
    train_years = config.get("train_years")
    val_years = config.get("val_years")
    dataloader_kwargs = copy.deepcopy(config.get("dataloader_kwargs") or {})

    seed = config.get("seed")
    if seed is not None:
        keras.utils.set_random_seed(int(seed))

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info("Running with config:")
    logging.info(config)

    weights = {"wN": wN, "w0": w0, "w1": w1, "w2": w2, "wW": wW}

    # Create data laoders
    dataloader_kwargs.update(
        {"select_keys": input_variables + ["range_folded_mask", "coordinates"], "n_frames": frames}
    )
    ds_train = get_dataloader(
        DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs
    )
    ds_val = None
    if val_years:
        ds_val = get_dataloader(
            DATA_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs
        )
    else:
        logging.info("No val_years given: training without a validation set")

    x, _, _ = next(iter(ds_train))
    in_shapes = (None, None, get_shape(x)[-1])
    c_shapes = (None, None, x["coordinates"].shape[-1])

    nn = build_model(
        shape=in_shapes,
        c_shape=c_shapes,
        start_filters=start_filters,
        l2_reg=l2_reg,
        input_variables=input_variables,
        head=head,
        head_kwargs=head_kwargs,
        frames=frames,
    )

    if init_from:
        init_path = resolve_init_from(init_from)
        n_layers = load_backbone_weights(nn, init_path)
        logging.info(f"initialised {n_layers} backbone layers from {init_path}")

    # model setup
    lr = keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay"
    )
    backbone = backbone_layers(nn)
    if backbone_lr_scale == 0:
        for layer in backbone:
            layer.trainable = False
        logging.info("backbone frozen")
    elif backbone_lr_scale != 1.0:
        logging.info(f"backbone learning rate scaled by {backbone_lr_scale}")

    from_logits = True
    if loss_fn.lower() == "cce":
        loss = keras.losses.BinaryCrossentropy(
            from_logits=from_logits, label_smoothing=label_smooth
        )
    elif loss_fn.lower() == "hinge":
        loss = keras.losses.Hinge()  # automatically converts labels to -1,1
    elif loss_fn.lower() == "mae":
        loss = mae_loss
    else:
        raise RuntimeError(f"unknown loss {loss_fn}")

    if backbone_lr_scale in (0, 1.0):
        opt = keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = ScaledAdam(learning_rate=lr, scales=scales_for(backbone, backbone_lr_scale))

    # Compute various metrics while training
    metrics = [
        keras.metrics.AUC(from_logits=from_logits, name="AUC", num_thresholds=2000),
        keras.metrics.AUC(from_logits=from_logits, curve="PR", name="AUCPR", num_thresholds=2000),
        tfm.BinaryAccuracy(from_logits, name="BinaryAccuracy"),
        tfm.TruePositives(from_logits, name="TruePositives"),
        tfm.FalsePositives(from_logits, name="FalsePositives"),
        tfm.TrueNegatives(from_logits, name="TrueNegatives"),
        tfm.FalseNegatives(from_logits, name="FalseNegatives"),
        tfm.Precision(from_logits, name="Precision"),
        tfm.Recall(from_logits, name="Recall"),
        tfm.F1Score(from_logits=from_logits, name="F1"),
    ]

    nn.compile(loss=loss, metrics=metrics, optimizer=opt, weighted_metrics=[])

    ## Setup experiment directory and model callbacks
    expdir = make_exp_dir(exp_dir=exp_dir, prefix=exp_name)
    git_commit, git_dirty = git_revision()
    logging.info(f"expdir={expdir} (git {git_commit}{'+dirty' if git_dirty else ''})")

    # Copy the properties that were used
    with open(os.path.join(expdir, "data.json"), "w") as f:
        json.dump(
            {
                "data_root": DATA_ROOT,
                "train_data": list(train_years),
                "val_data": list(val_years),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
            f,
        )
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config}, f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir, "train.py"))

    # Callbacks
    checkpoints_dir = make_callback_dirs(expdir)
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(expdir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
        CollectGarbage(),
    ]
    best_checkpoint = last_checkpoint = None
    if config.get("save_all_checkpoints"):
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoints_dir, "tornadoDetector_{epoch:03d}.keras"),
                save_best_only=False,
            )
        )
    else:
        last_checkpoint = os.path.join(checkpoints_dir, "last.keras")
        callbacks.append(keras.callbacks.ModelCheckpoint(last_checkpoint, save_best_only=False))
        if ds_val is not None:
            best_checkpoint = os.path.join(checkpoints_dir, "best.keras")
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    best_checkpoint, monitor="val_AUC", mode="max", save_best_only=True
                )
            )

    ## FIT
    history = nn.fit(
        ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks, verbose=1
    )

    # At the end, report the best validation score observed over all epochs
    val_auc = history.history.get("val_AUC", [])
    if val_auc:
        best_epoch = int(np.argmax(val_auc)) + 1
        results = {
            "AUC": float(val_auc[best_epoch - 1]),
            "AUCPR": float(history.history["val_AUCPR"][best_epoch - 1]),
            "best_epoch": best_epoch,
        }
    else:
        results = {"AUC": None, "AUCPR": None, "best_epoch": None}
    results.update(
        {
            "epochs_run": len(history.history["loss"]),
            "exp_name": exp_name,
            "seed": seed,
            "head": head,
            "head_kwargs": head_kwargs,
            "frames": frames,
            "init_from": init_from,
            "backbone_lr_scale": backbone_lr_scale,
            "params": int(nn.count_params()),
            "train_years": list(train_years),
            "val_years": list(val_years),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "expdir": expdir,
        }
    )
    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"results: {results}")

    return results


def cli():
    """Entry point for `tornet-train [params.json]`."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    # Load param file if given
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config.update(json.load(f))
    main(config)


if __name__ == "__main__":
    cli()
