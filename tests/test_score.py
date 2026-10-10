"""Offline scoring helpers and the tornet-score command."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from tornet.scripts import score


def test_roc_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert score.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert score.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert score.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # ties -> chance
    # one negative outranks one positive: 3 of 4 pairs correct
    assert score.roc_auc(y, np.array([0.1, 0.7, 0.6, 0.9])) == pytest.approx(0.75)
    assert np.isnan(score.roc_auc(np.array([1, 1]), np.array([0.1, 0.9])))


def test_average_precision_known_values():
    y = np.array([0, 0, 1, 1])
    assert score.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # ranking: pos, neg, pos, neg -> precisions at hits 1/1 and 2/3
    assert score.average_precision(y, np.array([0.4, 0.2, 0.9, 0.6])) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert np.isnan(score.average_precision(np.zeros(3), np.array([0.1, 0.2, 0.3])))


def test_best_threshold_maximises_csi():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
    best = score.best_threshold(y, p)
    assert best["threshold"] == pytest.approx(0.7)
    assert best["csi"] == 1.0 and best["precision"] == 1.0 and best["recall"] == 1.0


def synthetic_predictions(n=400, seed=0, skill=2.0):
    rng = np.random.default_rng(seed)
    category = rng.choice(["TOR", "NUL", "WRN"], size=n, p=[0.15, 0.55, 0.30])
    label = (category == "TOR").astype(int)
    ef = np.where(label == 1, rng.integers(0, 4, n), -1)
    logit = rng.normal(skill * (label - 0.5), 1.0)
    return pd.DataFrame(
        {
            "filename": [f"test/{2013 + i % 2}/f{i}.nc" for i in range(n)],
            "label": label,
            "category": category,
            "ef_number": ef,
            "year": 2013 + np.arange(n) % 2,
            "prob": 1 / (1 + np.exp(-logit)),
        }
    )


def test_score_file_views_and_breakdowns():
    df = synthetic_predictions()
    r = score.score_file(df, n_boot=50, seed=0)

    assert set(r["views"]) == {"NC", "RC", "WC"}
    nc, rc, wc = (r["views"][k] for k in ("NC", "RC", "WC"))
    assert nc["n"] == len(df) and rc["n"] + wc["n"] - nc["n_confirmed"] == nc["n"]
    assert nc["n_confirmed"] == rc["n_confirmed"] == wc["n_confirmed"]
    assert 0.8 < nc["auc"] <= 1.0
    assert nc["auc_ci95"][0] <= nc["auc"] <= nc["auc_ci95"][1]
    assert 0 < nc["best_csi"] <= 1

    assert set(r["by_ef"]) <= {"EF0", "EF1", "EF2", "EF3"}
    assert set(r["by_year"]) == {"2013", "2014"}
    assert len(r["reliability"]) == 10


def test_paired_difference_detects_better_model():
    a = synthetic_predictions(skill=1.0)
    b = synthetic_predictions(skill=3.0)
    d = score.paired_difference(a, b, n_boot=100, seed=0)
    assert d["n_common"] == len(a)
    assert d["NC"]["delta_auc"] > 0
    assert d["NC"]["delta_auc_ci95"][0] > 0  # confidently better


def test_cli_writes_report(tmp_path, monkeypatch, capsys):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    synthetic_predictions(skill=1.0).to_csv(a, index=False)
    synthetic_predictions(skill=3.0).to_csv(b, index=False)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["tornet-score", str(a), str(b), "--bootstrap", "20", "--output", str(out)]
    )
    score.main()
    printed = capsys.readouterr().out
    assert "=== " + str(a) in printed and "paired bootstrap" in printed
    report = json.loads(out.read_text())
    assert str(b) in report["paired_vs_first"]
