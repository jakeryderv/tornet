"""Inspect the test-set embeddings written by a tornet-train-head run with head "embedding".

Writes PCA scatter plots coloured by category, EF rating and year, and an embedding_report.json
with centroid cosine similarities, nearest-centroid assignment rates per category, and a k-means
clustering cross-tabulated against the categories (does the warned-null population form its own
cluster?).

Usage: tornet-embeddings <run directory> [--clusters 3]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

CATEGORIES = ["TOR", "NUL", "WRN"]


def pca2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        sims = x @ centers.T  # cosine on unit vectors
        new = sims.argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                m = members.mean(axis=0)
                centers[c] = m / (np.linalg.norm(m) + 1e-8)
    return labels


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def report(embs: np.ndarray, meta: pd.DataFrame, k: int) -> dict:
    cats = meta["category"].to_numpy()
    centroids = {c: unit(embs[cats == c].mean(axis=0)) for c in CATEGORIES if (cats == c).any()}
    names = list(centroids)
    cmat = np.array([[float(centroids[a] @ centroids[b]) for b in names] for a in names])
    nearest = np.array(names)[np.stack([embs @ centroids[n] for n in names], axis=1).argmax(axis=1)]
    assignment = {c: {n: float((nearest[cats == c] == n).mean()) for n in names} for c in names}
    # within- vs between-category cosine (sampled for speed)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(embs), min(len(embs), 3000), replace=False)
    sub, subcats = embs[idx], cats[idx]
    sims = sub @ sub.T
    same = subcats[:, None] == subcats[None, :]
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    cohesion = {
        "within_category_cosine": float(sims[same].mean()),
        "between_category_cosine": float(sims[diff].mean()),
    }
    clusters = kmeans(embs, k)
    crosstab = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(cats, name="category"))
    return {
        "n": int(len(embs)),
        "centroid_cosine": {
            a: {b: cmat[i, j] for j, b in enumerate(names)} for i, a in enumerate(names)
        },
        "nearest_centroid_rate": assignment,
        **cohesion,
        "kmeans": {"k": k, "cluster_by_category": crosstab.to_dict()},
    }


def plots(xy: np.ndarray, meta: pd.DataFrame, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    colourings = {
        "category": (meta["category"], {"NUL": "#9aa0a6", "WRN": "#f4a261", "TOR": "#d62828"}),
        "year": (meta["year"].astype(str), None),
    }
    ef = meta["ef_number"].where(meta["ef_number"] >= 0)
    for name, (series, palette) in colourings.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        for value in sorted(series.unique(), key=str):
            m = (series == value).to_numpy()
            colour = palette.get(value) if palette else None
            ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.5, label=str(value), c=colour)
        ax.legend(markerscale=4)
        ax.set_title(f"test embeddings (PCA), coloured by {name}")
        path = os.path.join(out_dir, f"embeddings_{name}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    nulls = ef.isna().to_numpy()
    ax.scatter(xy[nulls, 0], xy[nulls, 1], s=3, alpha=0.2, c="#9aa0a6", label="no tornado")
    sc = ax.scatter(xy[~nulls, 0], xy[~nulls, 1], s=8, alpha=0.8, c=ef[~nulls], cmap="viridis")
    fig.colorbar(sc, ax=ax, label="EF rating")
    ax.legend()
    ax.set_title("test embeddings (PCA), tornadoes coloured by EF")
    path = os.path.join(out_dir, "embeddings_ef.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="tornet-train-head run directory with embeddings_test.npy")
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    embs = np.load(os.path.join(args.run_dir, "embeddings_test.npy")).astype(np.float32)
    meta = pd.read_csv(os.path.join(args.run_dir, "predictions.csv"))
    assert len(embs) == len(meta), "embeddings and predictions are out of step"
    embs = unit(embs)

    rep = report(embs, meta, args.clusters)
    with open(os.path.join(args.run_dir, "embedding_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    written = plots(pca2(embs), meta, args.run_dir)

    names = list(rep["centroid_cosine"])
    print("centroid cosine similarity:")
    print(pd.DataFrame(rep["centroid_cosine"]).loc[names, names].round(3).to_string())
    print("\nnearest-centroid assignment rate (row: true category, col: nearest centroid):")
    print(pd.DataFrame(rep["nearest_centroid_rate"]).T.loc[names, names].round(3).to_string())
    print(
        f"\nmean cosine within category {rep['within_category_cosine']:.3f}, "
        f"between categories {rep['between_category_cosine']:.3f}"
    )
    print(f"\nk-means (k={args.clusters}) clusters by category:")
    print(pd.DataFrame(rep["kmeans"]["cluster_by_category"]).to_string())
    print("\nwrote " + ", ".join(os.path.basename(w) for w in written) + ", embedding_report.json")


if __name__ == "__main__":
    main()
