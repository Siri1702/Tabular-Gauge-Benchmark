"""Clean the multi-seed CSV of stale rows from the broken OpenML IDs.

The old (2026-08-02) OpenML IDs for kc1 / kc2 / pc1 (3917 / 3918 / 3919)
were silently re-mapped to unrelated QSAR-TID multi-class datasets by
OpenML. The first few runs of the multi-seed CSV were built on these
broken IDs and contain rows that are silently wrong (4-class labels,
p=1024). This script removes them so any downstream analysis is on the
corrected 9-dataset Act-2 list.

Run:
    python scripts/clean_multiseed_csv.py

It reads experiments/results/few_shot/multiseed_few_shot_results.csv,
removes:
  - rows for kc2 (correct id is 1063, but kc2 fails M1 so it's not in Act-2)
  - rows for pc1 (correct id is 1068, but pc1 fails M1 so it's not in Act-2)
  - rows for kc1 with actual_n_train == 711 (the broken id 3917; correct
    kc1 has actual_n_train == 1687)
and writes the cleaned CSV back to the same path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


CSV = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"


def main() -> None:
    if not CSV.exists():
        print(f"No {CSV.relative_to(ROOT)} — nothing to clean.")
        return

    df = pd.read_csv(CSV)
    print(f"Before cleanup: {len(df)} rows")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print()

    # Drop kc2 / pc1 entirely (correct openml ids give 522 / 1109 rows,
    # but the broken ids give 93 / 83 rows — both fail M1 anyway).
    pre = len(df)
    df = df[~df["dataset"].isin(["kc2", "pc1"])]
    dropped_legacy = pre - len(df)
    print(f"Dropped {dropped_legacy} rows for kc2/pc1 (failed M1, not in Act-2)")

    # For kc1: keep only rows with actual_n_train == 1687 (the correct id 1067).
    # The broken id 3917 gave 711 rows.
    if "kc1" in df["dataset"].unique():
        pre = len(df)
        bad_kc1 = df[(df["dataset"] == "kc1") & (df["actual_n_train"].fillna(-1) != 1687)]
        df = df.drop(bad_kc1.index)
        print(f"Dropped {pre - len(df)} broken kc1 rows (actual_n_train != 1687)")

    print()
    print(f"After cleanup: {len(df)} rows")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

    if "actual_n_train" in df.columns:
        print()
        print("Per-dataset actual_n_train (sanity check):")
        for ds in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == ds]
            n_train_unique = sub["actual_n_train"].dropna().unique()
            if len(n_train_unique) == 1:
                print(f"  {ds:16s}: actual_n_train={int(n_train_unique[0])} (n_rows={len(sub)})")
            else:
                print(f"  {ds:16s}: actual_n_train={sorted(n_train_unique.tolist())} (n_rows={len(sub)})")

    df.to_csv(CSV, index=False)
    print(f"\nWrote {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()