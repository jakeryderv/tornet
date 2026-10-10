"""Tabulate results.json files written by tornet-train.

Usage: tornet-results [experiments_dir] [--group] [--csv out.csv]
"""

import argparse
import json
import os

import pandas as pd

COLUMNS = [
    "run",
    "git_commit",
    "git_dirty",
    "train_years",
    "val_years",
    "epochs_run",
    "best_epoch",
    "AUC",
    "AUCPR",
]


def collect(exp_dir: str) -> pd.DataFrame:
    """One row per results.json found under exp_dir (symlinks such as `latest` are not followed)."""
    rows = []
    for root, _dirs, files in os.walk(exp_dir, followlinks=False):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                row = json.load(f)
            row["run"] = os.path.relpath(root, exp_dir)
            rows.append(row)
    return pd.DataFrame(rows)


def fold_group(exp_name: str) -> str:
    """`baseline_fold_a` -> `baseline`, so folds of one experiment can be averaged."""
    return pd.Series([exp_name]).str.replace(r"_fold_[a-z0-9]+$", "", regex=True).iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=os.environ.get("EXP_DIR", "experiments"),
        help="Directory to scan (default: $EXP_DIR or ./experiments)",
    )
    parser.add_argument("--group", action="store_true", help="Also average folds per experiment")
    parser.add_argument("--csv", help="Write the per-run table to this CSV path")
    args = parser.parse_args()

    df = collect(args.exp_dir)
    if df.empty:
        print(f"no results.json files under {args.exp_dir}")
        return
    df = df.sort_values("run").reset_index(drop=True)
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    table = df[COLUMNS]

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(table.to_string(index=False))
        if args.group and "exp_name" in df:
            grouped = (
                df.assign(group=df["exp_name"].map(fold_group))
                .groupby("group")[["AUC", "AUCPR"]]
                .agg(["mean", "std", "count"])
            )
            print()
            print(grouped.to_string())
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
