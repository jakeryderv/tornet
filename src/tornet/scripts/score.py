"""Score saved predictions offline: the paper's three test views, best-threshold CSI, bootstrap
confidence intervals, breakdowns by EF rating and year, reliability, and paired comparison.

Usage: tornet-score baseline.csv [variant.csv ...] [--bootstrap N] [--output scores.json]

Input CSVs come from `tornet-test --predictions` and need the columns
filename, label, category, ef_number, year, prob.
"""

import argparse
import json

import numpy as np
import pandas as pd

# The paper's three views of the test set: which null categories are kept alongside confirmed.
VIEWS = {
    "NC": ("all nulls vs confirmed", {"TOR", "NUL", "WRN"}),
    "RC": ("random nulls vs confirmed", {"TOR", "NUL"}),
    "WC": ("warnings vs confirmed", {"TOR", "WRN"}),
}


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan if only one class is present."""
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision); nan without positives."""
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def best_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Threshold maximising CSI = TP / (TP + FP + FN), with the metrics at that threshold."""
    y = np.asarray(y, bool)
    candidates = np.unique(np.round(p, 4))
    best = {"threshold": 0.5, "csi": -1.0}
    for t in candidates:
        pred = p >= t
        tp = (pred & y).sum()
        fp = (pred & ~y).sum()
        fn = (~pred & y).sum()
        csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        if csi > best["csi"]:
            best = {
                "threshold": float(t),
                "csi": float(csi),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            }
    return best


def view_metrics(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    y, p = df["label"].to_numpy(bool), df["prob"].to_numpy(float)
    out = {
        "n": int(len(df)),
        "n_confirmed": int(y.sum()),
        "accuracy_at_0.5": float(((p >= 0.5) == y).mean()),
        "auc": roc_auc(y, p),
        "aucpr": average_precision(y, p),
        **{f"best_{k}": v for k, v in best_threshold(y, p).items()},
    }
    if n_boot:
        aucs, aps = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(df), len(df))
            aucs.append(roc_auc(y[idx], p[idx]))
            aps.append(average_precision(y[idx], p[idx]))
        out["auc_ci95"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]
        out["aucpr_ci95"] = [float(np.nanpercentile(aps, 2.5)), float(np.nanpercentile(aps, 97.5))]
    return out


def breakdowns(df: pd.DataFrame, threshold: float) -> dict:
    """Detection rate per EF rating at the chosen threshold, AUC per year, and a reliability table."""
    pos = df[df["label"] == 1]
    by_ef = {
        f"EF{int(ef)}": {"n": int(len(g)), "recall": float((g["prob"] >= threshold).mean())}
        for ef, g in pos.groupby("ef_number")
        if ef >= 0
    }
    by_year = {
        str(int(yr)): {"n": int(len(g)), "auc": roc_auc(g["label"], g["prob"])}
        for yr, g in df.groupby("year")
    }
    bins = np.linspace(0, 1, 11)
    df = df.assign(bin=pd.cut(df["prob"], bins, include_lowest=True))
    reliability = [
        {
            "bin": f"{b.left:.1f}-{b.right:.1f}",
            "n": int(len(g)),
            "mean_prob": float(g["prob"].mean()) if len(g) else None,
            "observed": float(g["label"].mean()) if len(g) else None,
        }
        for b, g in df.groupby("bin", observed=False)
    ]
    return {"by_ef": by_ef, "by_year": by_year, "reliability": reliability}


def score_file(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    views = {}
    for key, (desc, cats) in VIEWS.items():
        sub = df[df["category"].isin(cats)]
        views[key] = {"description": desc, **view_metrics(sub, n_boot, rng)}
    return {"views": views, **breakdowns(df, views["NC"]["best_threshold"])}


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Bootstrap the AUC and AUCPR difference (b - a) over the same test files, per view."""
    m = a.merge(b, on="filename", suffixes=("_a", "_b"))
    if m.empty:
        return {"error": "no filenames in common"}
    rng = np.random.default_rng(seed)
    out = {"n_common": int(len(m))}
    for key, (_desc, cats) in VIEWS.items():
        sub = m[m["category_a"].isin(cats)]
        y = sub["label_a"].to_numpy(bool)
        pa, pb = sub["prob_a"].to_numpy(float), sub["prob_b"].to_numpy(float)
        d_auc = roc_auc(y, pb) - roc_auc(y, pa)
        d_ap = average_precision(y, pb) - average_precision(y, pa)
        boots_auc, boots_ap = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub), len(sub))
            boots_auc.append(roc_auc(y[idx], pb[idx]) - roc_auc(y[idx], pa[idx]))
            boots_ap.append(average_precision(y[idx], pb[idx]) - average_precision(y[idx], pa[idx]))
        out[key] = {
            "delta_auc": float(d_auc),
            "delta_auc_ci95": [
                float(np.nanpercentile(boots_auc, 2.5)),
                float(np.nanpercentile(boots_auc, 97.5)),
            ]
            if n_boot
            else None,
            "delta_aucpr": float(d_ap),
            "delta_aucpr_ci95": [
                float(np.nanpercentile(boots_ap, 2.5)),
                float(np.nanpercentile(boots_ap, 97.5)),
            ]
            if n_boot
            else None,
        }
    return out


def print_report(name: str, r: dict) -> None:
    print(f"\n=== {name} ===")
    rows = []
    for key, v in r["views"].items():
        ci = f"[{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]" if "auc_ci95" in v else ""
        rows.append(
            {
                "view": key,
                "n": v["n"],
                "confirmed": v["n_confirmed"],
                "AUC": f"{v['auc']:.4f}",
                "AUC 95% CI": ci,
                "AUCPR": f"{v['aucpr']:.4f}",
                "CSI*": f"{v['best_csi']:.4f}",
                "thr*": f"{v['best_threshold']:.2f}",
                "P*": f"{v['best_precision']:.3f}",
                "R*": f"{v['best_recall']:.3f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("  * at the CSI-maximising threshold for that view")
    ef = r["by_ef"]
    if ef:
        print(
            "  recall by EF at the NC threshold: "
            + ", ".join(f"{k} {v['recall']:.2f} (n={v['n']})" for k, v in ef.items())
        )
    print("  AUC by year: " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in r["by_year"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", nargs="+", help="CSV(s) from tornet-test --predictions")
    parser.add_argument(
        "--bootstrap", type=int, default=200, help="Bootstrap resamples (0 to skip)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write the full report as JSON")
    args = parser.parse_args()

    frames = {path: pd.read_csv(path) for path in args.predictions}
    report = {path: score_file(df, args.bootstrap, args.seed) for path, df in frames.items()}
    for path, r in report.items():
        print_report(path, r)

    if len(frames) > 1:
        base_path, base = next(iter(frames.items()))
        report["paired_vs_first"] = {}
        for path, df in list(frames.items())[1:]:
            diff = paired_difference(base, df, args.bootstrap, args.seed)
            report["paired_vs_first"][path] = diff
            print(f"\n=== {path} minus {base_path} (paired bootstrap on common files) ===")
            for key in VIEWS:
                if key in diff:
                    d = diff[key]
                    ci = (
                        f" 95% CI [{d['delta_auc_ci95'][0]:+.4f}, {d['delta_auc_ci95'][1]:+.4f}]"
                        if d["delta_auc_ci95"]
                        else ""
                    )
                    print(
                        f"  {key}: dAUC {d['delta_auc']:+.4f}{ci}   dAUCPR {d['delta_aucpr']:+.4f}"
                    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
