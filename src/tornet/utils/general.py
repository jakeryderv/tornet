"""
Utils for importing modules from string

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import datetime
import os
import subprocess

import keras


def make_exp_dir(
    exp_dir="../experiments", prefix="", symlink_name="latest", task_type=None, task_id=0
):
    """
    Creates a run directory named <prefix>_<yymmdd_HHMMSS>, with SLURM job and array task
    ids appended when those variables are set, and points a symlink at it.
    """
    parts = [datetime.datetime.now().strftime("%y%m%d_%H%M%S")]
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        if os.getenv(var):
            parts.append(os.environ[var])
    dated_dir = "_".join(([prefix] if prefix else []) + parts)
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if array_job_id:
        dated_dir = os.path.join(array_job_id, dated_dir)
    os.makedirs(os.path.join(exp_dir, dated_dir))
    linked_dir = os.path.join(exp_dir, symlink_name)
    if os.path.islink(linked_dir):
        os.unlink(linked_dir)
    os.symlink(dated_dir, linked_dir)
    return os.path.join(exp_dir, dated_dir)


def git_revision(path: str | None = None) -> tuple[str | None, bool | None]:
    """
    Short commit hash and dirty flag for the git checkout containing `path` (default: this
    package). Returns (None, None) when not running from a git checkout.
    """
    path = path or os.path.dirname(os.path.abspath(__file__))
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", "-C", path, *args], capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = run("rev-parse", "--short", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def make_callback_dirs(logdir):
    checkpoints_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


class CollectGarbage(keras.callbacks.Callback):
    """Keras on the torch backend can leave reference cycles holding GPU tensors between epochs,
    which grows memory until an OOM (seen with multi-output heads). Collect at each epoch end."""

    def on_epoch_end(self, epoch, logs=None):
        import gc

        gc.collect()
        if keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
