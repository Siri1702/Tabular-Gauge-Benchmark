"""
Merge a CSV produced on Google Colab back into the local multi-seed CSV.

The Colab script (scripts/colab_multiseed_resume.py) writes back the same
file you uploaded, so the simplest workflow is just to drop the Colab
download in place:

    cp /path/to/from_colab.csv experiments/results/few_shot/multiseed_few_shot_results.csv

This helper exists for the case where you have *new* rows in the Colab
CSV that were not in the local CSV (e.g. you ran seeds 3,4 on Colab but
seeds 0,1,2 locally and you want to keep both). It:

  1. Reads the local CSV (the source of truth) and the Colab CSV.
  2. Concatenates, deduplicating on the unique key
        (model, dataset, train_size, seed)
     and keeping the Colab row on conflict (more recent machine).
  3. Sorts by (dataset, model, train_size, seed) for readability.
  4. Writes back to the local CSV path.
  5. Prints before/after counts so you can sanity-check the merge.

Usage
-----
    python scripts/merge_colab_csv.py \
        --local experiments/results/few_shot/multiseed_few_shot_results.csv \
        --colab /Users/you/Downloads/multiseed_few_shot_results.csv

    # Optionally write to a different path and only print stats
    python scripts/merge_colab_csv.py \
        --local experiments/results/few_shot/multiseed_few_shot_results.csv \
        --colab /Users/you/Downloads/multiseed_few_shot_results.csv \
        --out   experiments/results/few_shot/multiseed_few_shot_merged.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


KEY_COLS = ["model", "dataset", "train_size", "seed"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--local", required=True, type=Path,
        help="Path to the local multiseed CSV (source of truth).",
    )
    parser.add_argument(
        "--colab", required=True, type=Path,
        help="Path to the CSV downloaded from Google Colab.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the merged CSV. Defaults to --local.",
    )
    args = parser.parse_args()

    if not args.local.exists():
        sys.exit(f"Local CSV not found: {args.local}")
    if not args.colab.exists():
        sys.exit(f"Colab CSV not found: {args.colab}")

    out = args.out or args.local

    local = pd.read_csv(args.local)
    colab = pd.read_csv(args.colab)

    # Make sure the train_size column is string-typed on both sides so the
    # merge key compares apples to apples (the local runner stores
    # train_size as the string 'full' for the full-split cells).
    for df in (local, colab):
        df["train_size"] = df["train_size"].astype(str)
        if "seed" in df.columns:
            df["seed"] = df["seed"].astype(int)

    n_local = len(local)
    n_colab = len(colab)

    # Concatenate, then drop duplicates keeping the LAST occurrence
    # (the Colab rows go in second, so they win on conflict).
    merged = pd.concat([local, colab], ignore_index=True)
    n_total_before = len(merged)
    merged = merged.drop_duplicates(subset=KEY_COLS, keep="last")
    n_total_after = len(merged)

    # Re-introduce the original column order (in case the Colab file
    # somehow drifted in column set — e.g. dropped a metric).
    expected_cols = list(local.columns)
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[expected_cols]

    # Sort for human-readable diffs vs the original.
    sort_cols = ["dataset", "model", "train_size", "seed"]
    # train_size 'full' should sort at the end, not alphabetically.
    merged["_size_sort"] = merged["train_size"].apply(
        lambda v: (1, 0) if v == "full" else (0, int(v))
    )
    merged = merged.sort_values(
        by=["dataset", "model", "_size_sort", "seed"]
    ).drop(columns=["_size_sort"])

    merged.to_csv(out, index=False)

    # Report
    print("=" * 70)
    print("MERGE REPORT")
    print("=" * 70)
    print(f"Local rows : {n_local}")
    print(f"Colab rows : {n_colab}")
    print(f"Pre-merge  : {n_total_before} rows")
    print(f"Duplicates : {n_total_before - n_total_after} ({(n_total_before - n_total_after) / max(n_total_before, 1) * 100:.1f}%)")
    print(f"Merged rows: {n_total_after}  -> {out}")
    print()

    # Coverage check
    models = sorted(merged["model"].unique())
    datasets = sorted(merged["dataset"].unique())
    seeds = sorted(merged["seed"].unique())
    sizes = sorted(merged["train_size"].unique(), key=lambda x: (x == "full", int(x) if x != "full" else 0))
    expected = len(models) * len(datasets) * len(seeds) * len(sizes)
    print(f"Models     : {len(models)} -> {models}")
    print(f"Datasets   : {len(datasets)} -> {datasets}")
    print(f"Seeds      : {len(seeds)} -> {seeds}")
    print(f"Train sizes: {len(sizes)} -> {sizes}")
    print(f"Expected coverage: {expected} cells")
    print(f"Actual coverage  : {len(merged)} cells")
    if len(merged) < expected:
        from collections import Counter
        have = set(zip(merged["model"], merged["dataset"], merged["train_size"], merged["seed"]))
        all_cells = {(m, d, s, sz) for m in models for d in datasets for s in seeds for sz in sizes}
        missing = all_cells - have
        by_model = Counter(m for (m, d, s, sz) in missing)
        by_seed = Counter(s for (m, d, s, sz) in missing)
        print(f"\n[warn] {len(missing)} cells still missing after merge.")
        print(f"  by model: {dict(by_model)}")
        print(f"  by seed : {dict(by_seed)}")
    else:
        print("\nAll expected cells present. Run scripts/regenerate_report.py next.")


if __name__ == "__main__":
    main()
