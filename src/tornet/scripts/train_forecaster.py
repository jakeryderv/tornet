"""Train the radar forecaster (tornet.models.forecaster) on TorNet: the oldest `n_in` scans in,
the next `horizons` scans out, scored later by the frozen detector (tornet-score-forecast).

Two training modes, same model:
- `detector_weight` 0: masked-by-construction MAE on the future scans in the detector's
  normalised, background-filled representation (the control).
- `detector_weight` > 0: the same, plus the frozen detector's tornado cross-entropy on each
  forecast scan against that horizon's label (the detector as a loss).

Usage: tornet-train-forecaster <config.json>
"""

import copy
import json
import logging
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import keras  # noqa: E402

from tornet.data.forecast import ForecastLoader  # noqa: E402
from tornet.models.forecaster import (  # noqa: E402
    build_forecaster,
    load_encoder_weights,
    set_encoder_trainable,
)
from tornet.scripts.train_keras import resolve_init_from  # noqa: E402
from tornet.utils.general import (  # noqa: E402
    CollectGarbage,
    git_revision,
    make_callback_dirs,
    make_exp_dir,
)

logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG = {
    "train_years": [2013, 2015],
    "val_years": [2014],
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "decay_steps": None,  # default: one epoch
    "decay_rate": 0.958,
    "n_in": 2,
    "horizons": 2,
    "start_filters": 48,
    "decoder_filters": [256, 128, 64, 32],
    "l2_reg": 1e-5,
    "init_from": "pretrained",  # encoder initialisation: a .keras path or the published weights
    "encoder_trainable": True,
    "detector": "pretrained",  # the frozen scorer inside the model when detector_weight > 0
    "detector_weight": 0.0,
    "seed": 1234,
    "exp_name": "forecast",
    "exp_dir": os.environ.get("EXP_DIR", "experiments"),
    "dataloader_kwargs": {"workers": 8, "use_multiprocessing": True, "max_queue_size": 8},
    "max_files": None,  # cap on train/val files (smoke tests)
    "save_all_checkpoints": False,
}


def main(config):
    data_root = os.environ["TORNET_ROOT"]
    seed = config.get("seed")
    if seed is not None:
        keras.utils.set_random_seed(int(seed))
    logging.info(f"Using {keras.config.backend()} backend; TORNET_ROOT={data_root}")
    logging.info(config)
    n_in, horizons = int(config["n_in"]), int(config["horizons"])
    with_detector = float(config["detector_weight"]) > 0
    target_keys = ["fields", "logits"] if with_detector else ["fields"]
    loader_kwargs = dict(config["dataloader_kwargs"])
    ds_train = ForecastLoader(
        data_root,
        "train",
        config["train_years"],
        batch_size=config["batch_size"],
        n_in=n_in,
        horizons=horizons,
        select_keys=ForecastLoader.model_input_keys(),
        target_keys=target_keys,
        random_state=seed or 0,
        **loader_kwargs,
    )
    ds_val = None
    if config["val_years"]:
        ds_val = ForecastLoader(
            data_root,
            "train",
            config["val_years"],
            batch_size=config["batch_size"],
            n_in=n_in,
            horizons=horizons,
            select_keys=ForecastLoader.model_input_keys(),
            target_keys=target_keys,
            **loader_kwargs,
        )
    if config.get("max_files"):
        ds_train.file_list = ds_train.file_list[: int(config["max_files"])]
        if ds_val is not None:
            ds_val.file_list = ds_val.file_list[: int(config["max_files"])]
    sample_x, _ = ds_train[0]
    shape = tuple(sample_x["DBZ"].shape[2:])

    detector = None
    if with_detector:
        detector = keras.saving.load_model(resolve_init_from(config["detector"]), compile=False)
        detector.trainable = False
    model = build_forecaster(
        shape=shape,
        n_in=n_in,
        horizons=horizons,
        start_filters=int(config["start_filters"]),
        decoder_filters=tuple(config["decoder_filters"]),
        l2_reg=float(config["l2_reg"]),
        detector=detector,
    )
    if config.get("init_from"):
        source = keras.saving.load_model(resolve_init_from(config["init_from"]), compile=False)
        n_set = load_encoder_weights(model, source)
        logging.info(f"encoder initialised from {config['init_from']}: {n_set} layers")
    set_encoder_trainable(model, bool(config.get("encoder_trainable", True)))
    logging.info(
        f"forecaster {model.count_params():,} params, "
        f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,} trainable"
    )

    decay_steps = config["decay_steps"] or len(ds_train)
    lr = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps, config["decay_rate"], staircase=False
    )
    losses = {"fields": keras.losses.MeanAbsoluteError()}
    weights = {"fields": 1.0}
    metrics = {}
    if with_detector:
        losses["logits"] = keras.losses.BinaryCrossentropy(from_logits=True)
        weights["logits"] = float(config["detector_weight"])
        metrics["logits"] = [keras.metrics.AUC(from_logits=True, name="AUC")]
    model.compile(
        loss=losses,
        loss_weights=weights,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=metrics,
    )

    expdir = make_exp_dir(exp_dir=config["exp_dir"], prefix=config["exp_name"])
    git_commit, git_dirty = git_revision()
    logging.info(f"expdir={expdir} (git {git_commit}{'+dirty' if git_dirty else ''})")
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": config, "params": model.count_params()}, f, indent=2)
    checkpoints_dir = make_callback_dirs(expdir)
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(expdir, "history.csv")),
        keras.callbacks.TerminateOnNaN(),
        CollectGarbage(),
    ]
    last_checkpoint = os.path.join(checkpoints_dir, "last.keras")
    callbacks.append(keras.callbacks.ModelCheckpoint(last_checkpoint, save_best_only=False))
    best_checkpoint = None
    if ds_val is not None:
        best_checkpoint = os.path.join(checkpoints_dir, "best.keras")
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                best_checkpoint, monitor="val_loss", mode="min", save_best_only=True
            )
        )
    history = model.fit(
        ds_train, epochs=config["epochs"], validation_data=ds_val, callbacks=callbacks, verbose=2
    )
    h = history.history
    val_loss = h.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else None
    results = {
        "exp_name": config["exp_name"],
        "epochs_run": len(h["loss"]),
        "best_epoch": best_epoch,
        "val_loss": float(val_loss[best_epoch - 1]) if best_epoch else None,
        "val_fields_mae": float(h["val_fields_loss"][best_epoch - 1])
        if best_epoch and "val_fields_loss" in h
        else (float(val_loss[best_epoch - 1]) if best_epoch else None),
        "val_logits_AUC": float(h["val_logits_AUC"][best_epoch - 1])
        if best_epoch and "val_logits_AUC" in h
        else None,
        "n_in": n_in,
        "horizons": horizons,
        "detector_weight": float(config["detector_weight"]),
        "encoder_trainable": bool(config.get("encoder_trainable", True)),
        "init_from": config.get("init_from"),
        "params": int(model.count_params()),
        "seed": seed,
        "train_years": list(config["train_years"]),
        "val_years": list(config["val_years"]),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "expdir": expdir,
    }
    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"results: {results}")
    return results


def cli():
    config = copy.deepcopy(DEFAULT_CONFIG)
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config.update(json.load(f))
    main(config)


if __name__ == "__main__":
    cli()
