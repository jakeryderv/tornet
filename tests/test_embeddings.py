"""tornet-embeddings report and plots on a synthetic embedding file."""

import json
import sys

import numpy as np
import pandas as pd

from tornet.scripts import embeddings


def test_embeddings_report_and_plots(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    n = 300
    cats = rng.choice(["TOR", "NUL", "WRN"], size=n, p=[0.2, 0.5, 0.3])
    centres = {
        "TOR": np.array([1.0, 0, 0]),
        "NUL": np.array([0, 1.0, 0]),
        "WRN": np.array([0, 0, 1.0]),
    }
    embs = np.stack([centres[c] + 0.3 * rng.standard_normal(3) for c in cats]).astype(np.float16)
    np.save(tmp_path / "embeddings_test.npy", embs)
    pd.DataFrame(
        {
            "filename": [f"f{i}" for i in range(n)],
            "category": cats,
            "label": (cats == "TOR").astype(int),
            "ef_number": np.where(cats == "TOR", rng.integers(0, 4, n), -1),
            "year": 2013 + np.arange(n) % 3,
            "prob": rng.random(n),
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    monkeypatch.setattr(sys, "argv", ["tornet-embeddings", str(tmp_path), "--clusters", "3"])
    embeddings.main()
    out = capsys.readouterr().out
    assert "centroid cosine similarity" in out

    rep = json.loads((tmp_path / "embedding_report.json").read_text())
    assert rep["n"] == n
    # well-separated synthetic clusters: each category lands on its own centroid
    for c in ("TOR", "NUL", "WRN"):
        assert rep["nearest_centroid_rate"][c][c] > 0.9
    assert rep["within_category_cosine"] > rep["between_category_cosine"]
    for name in ("embeddings_category.png", "embeddings_year.png", "embeddings_ef.png"):
        assert (tmp_path / name).stat().st_size > 0
