"""End-to-end runs of the train, test and results commands on the synthetic dataset."""

import json
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

from .conftest import SYNTHETIC_SAMPLES

keras = pytest.importorskip("keras")

from tornet.scripts import results, score, test_keras, train_keras  # noqa: E402

TINY = {"epochs": 2, "batch_size": 2, "start_filters": 4, "dataloader_kwargs": {}}


@pytest.fixture
def env(tornet_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TORNET_ROOT", str(tornet_root))
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


def train(env, **overrides):
    config = {**train_keras.DEFAULT_CONFIG, **TINY, "exp_dir": str(env), **overrides}
    return train_keras.main(config)


def test_train_with_validation_writes_results_and_checkpoints(env):
    r = train(env, exp_name="baseline_fold_a", train_years=[2013], val_years=[2014])

    assert r["epochs_run"] == 2
    assert r["best_epoch"] in (1, 2)
    assert 0.0 <= r["AUC"] <= 1.0
    assert r["train_years"] == [2013] and r["val_years"] == [2014]
    assert "git_commit" in r and "git_dirty" in r

    expdir = env / os.path.basename(r["expdir"])
    assert re.fullmatch(r"baseline_fold_a_\d{6}_\d{6}", expdir.name)
    assert json.loads((expdir / "results.json").read_text()) == r
    assert json.loads((expdir / "data.json").read_text())["git_commit"] == r["git_commit"]
    assert sorted(p.name for p in (expdir / "checkpoints").iterdir()) == [
        "best.keras",
        "last.keras",
    ]
    assert r["best_checkpoint"] == str(expdir / "checkpoints" / "best.keras")
    assert (expdir / "history.csv").exists()
    assert (env / "latest").is_symlink()


def test_train_init_from_checkpoint_with_transformer_head(env):
    base = train(env, exp_name="base", train_years=[2013], val_years=[2014])
    r = train(
        env,
        exp_name="finetune",
        train_years=[2013],
        val_years=[2014],
        head="transformer",
        head_kwargs={"depth": 1, "num_heads": 2, "pool": "cls"},
        init_from=base["last_checkpoint"],
        backbone_lr_scale=0.1,
    )
    assert r["epochs_run"] == 2
    assert r["head"] == "transformer" and r["init_from"] == base["last_checkpoint"]
    assert r["backbone_lr_scale"] == 0.1
    assert r["params"] > 0
    # the checkpoint (with its ScaledAdam) loads back in a fresh process via tornet-test
    assert os.path.exists(r["best_checkpoint"])


def test_train_without_validation_keeps_only_last(env):
    r = train(env, exp_name="baseline_final", train_years=[2013, 2014], val_years=[])
    assert r["AUC"] is None and r["best_epoch"] is None
    assert r["epochs_run"] == 2
    assert r["best_checkpoint"] is None
    expdir = env / os.path.basename(r["expdir"])
    assert [p.name for p in (expdir / "checkpoints").iterdir()] == ["last.keras"]


def test_save_all_checkpoints_keeps_every_epoch(env):
    r = train(env, train_years=[2013], val_years=[2014], save_all_checkpoints=True)
    expdir = env / os.path.basename(r["expdir"])
    assert sorted(p.name for p in (expdir / "checkpoints").iterdir()) == [
        "tornadoDetector_001.keras",
        "tornadoDetector_002.keras",
    ]


def test_main_does_not_mutate_defaults(env):
    before = json.dumps(train_keras.DEFAULT_CONFIG, sort_keys=True)
    train(env, epochs=1, train_years=[2013], val_years=[2014])
    assert json.dumps(train_keras.DEFAULT_CONFIG, sort_keys=True) == before


def test_evaluate_checkpoint_on_subset_years(env, tmp_path, monkeypatch):
    r = train(env, epochs=1, train_years=[2013], val_years=[2014])
    out = tmp_path / "scores.json"
    test_years = sorted(
        {int(rel.split("/")[1]) for rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "test"}
    )
    preds = tmp_path / "preds.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-test",
            "--model_path",
            r["best_checkpoint"],
            "--years",
            *map(str, test_years),
            "--batch_size",
            "2",
            "--output",
            str(out),
            "--predictions",
            str(preds),
        ],
    )
    test_keras.main()

    scores = json.loads(out.read_text())
    assert scores["years"] == test_years
    assert scores["model_path"] == r["best_checkpoint"]
    for name in ("AUC", "AUCPR", "BinaryAccuracy", "Precision", "Recall", "F1"):
        assert 0.0 <= scores[name] <= 1.0

    df = pd.read_csv(preds)
    n_test = sum(1 for _rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "test")
    assert len(df) == n_test
    assert set(df.columns) >= {
        "filename",
        "label",
        "category",
        "ef_number",
        "event_id",
        "year",
        "logit",
        "prob",
    }
    assert df["filename"].str.startswith("test/").all()
    assert set(df["category"]) <= {"TOR", "NUL", "WRN"}
    assert set(df["year"]) <= set(test_years)
    # tornet-score accepts the file as-is
    monkeypatch.setattr(sys, "argv", ["tornet-score", str(preds), "--bootstrap", "0"])
    score.main()


def test_tornet_test_cli_loads_checkpoint_in_fresh_process(env, tmp_path):
    """Custom layers must be registered by the command itself, not by whatever imported them earlier."""
    r = train(env, epochs=1, train_years=[2013], val_years=[2014])
    out = tmp_path / "scores.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tornet.scripts.test_keras",
            "--model_path",
            r["last_checkpoint"],
            "--years",
            "2014",
            "--batch_size",
            "2",
            "--output",
            str(out),
        ],
        env={**os.environ, "TORNET_ROOT": os.environ["TORNET_ROOT"]},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 0.0 <= json.loads(out.read_text())["AUC"] <= 1.0


def test_results_table_and_fold_grouping(env, monkeypatch, capsys):
    train(env, epochs=1, exp_name="baseline_fold_a", train_years=[2013], val_years=[2014])
    train(env, epochs=1, exp_name="baseline_fold_b", train_years=[2014], val_years=[2013])

    df = results.collect(str(env))
    assert len(df) == 2  # the `latest` symlink is not double counted
    assert set(df["exp_name"]) == {"baseline_fold_a", "baseline_fold_b"}
    assert results.fold_group("baseline_fold_a") == "baseline"

    monkeypatch.setattr(sys, "argv", ["tornet-results", str(env), "--group"])
    results.main()
    out = capsys.readouterr().out
    assert "baseline_fold_a_" in out and "baseline_fold_b_" in out
    assert "baseline" in out.splitlines()[-1] or "mean" in out
