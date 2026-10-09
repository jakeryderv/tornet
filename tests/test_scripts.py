"""End-to-end runs of the train and test commands on the synthetic dataset."""

import json
import sys

import pytest

from .conftest import SYNTHETIC_SAMPLES

keras = pytest.importorskip("keras")

from tornet.scripts import test_keras, train_keras  # noqa: E402

TINY = {
    "epochs": 2,
    "batch_size": 2,
    "start_filters": 4,
    "dataloader_kwargs": {},
}


@pytest.fixture
def env(tornet_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TORNET_ROOT", str(tornet_root))
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


def test_train_with_validation_writes_results_and_checkpoints(env):
    config = dict(
        train_keras.DEFAULT_CONFIG,
        **TINY,
        exp_name="fold",
        exp_dir=str(env),
        train_years=[2013],
        val_years=[2014],
    )
    results = train_keras.main(config)

    assert results["epochs_run"] == 2
    assert results["best_epoch"] in (1, 2)
    assert 0.0 <= results["AUC"] <= 1.0

    expdir = env / results["expdir"].split("/")[-1]
    assert json.loads((expdir / "results.json").read_text()) == results
    assert json.loads((expdir / "data.json").read_text())["train_data"] == [2013]
    assert sorted(p.name for p in (expdir / "checkpoints").iterdir()) == [
        "tornadoDetector_001.keras",
        "tornadoDetector_002.keras",
    ]
    assert (expdir / "history.csv").exists()
    assert (env / "latest").is_symlink()


def test_train_without_validation_years(env):
    config = dict(
        train_keras.DEFAULT_CONFIG,
        **TINY,
        exp_name="final",
        exp_dir=str(env),
        train_years=[2013, 2014],
        val_years=[],
    )
    results = train_keras.main(config)
    assert results["AUC"] is None
    assert results["best_epoch"] is None
    assert results["epochs_run"] == 2
    assert (
        env / results["expdir"].split("/")[-1] / "checkpoints" / "tornadoDetector_002.keras"
    ).exists()


def test_main_does_not_mutate_defaults(env):
    before = json.dumps(train_keras.DEFAULT_CONFIG, sort_keys=True)
    config = {
        **train_keras.DEFAULT_CONFIG,
        **TINY,
        "epochs": 1,
        "exp_dir": str(env),
        "train_years": [2013],
        "val_years": [2014],
    }
    train_keras.main(config)
    assert json.dumps(train_keras.DEFAULT_CONFIG, sort_keys=True) == before


def test_evaluate_checkpoint_on_subset_years(env, tmp_path, monkeypatch):
    config = {
        **train_keras.DEFAULT_CONFIG,
        **TINY,
        "epochs": 1,
        "exp_dir": str(env),
        "train_years": [2013],
        "val_years": [2014],
    }
    results = train_keras.main(config)
    checkpoint = (
        env / results["expdir"].split("/")[-1] / "checkpoints" / "tornadoDetector_001.keras"
    )
    out = tmp_path / "scores.json"

    test_years = sorted(
        {int(rel.split("/")[1]) for rel, kind, *_ in SYNTHETIC_SAMPLES if kind == "test"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornet-test",
            "--model_path",
            str(checkpoint),
            "--years",
            *map(str, test_years),
            "--batch_size",
            "2",
            "--output",
            str(out),
        ],
    )
    test_keras.main()

    scores = json.loads(out.read_text())
    assert scores["years"] == test_years
    assert scores["model_path"] == str(checkpoint)
    for name in ("AUC", "AUCPR", "BinaryAccuracy", "Precision", "Recall", "F1"):
        assert 0.0 <= scores[name] <= 1.0
