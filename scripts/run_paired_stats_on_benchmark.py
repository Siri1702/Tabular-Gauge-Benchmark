"""
Run paired statistical tests (Wilcoxon, bootstrap CI, Holm-Bonferroni, Cliff's δ)
against the EXISTING 15-dataset x 6-model x 5-seed benchmark.

Inputs : experiments/results/aggregated/results.csv   (already exists, 450 rows)
Outputs:
    experiments/results/aggregated/paired_stats.csv           (all numbers)
    reports/appendix_paired_stats.md                           (markdown table)

Why this exists:
    tabular_gauge/evaluation/paired_stats.py was originally only invoked from the few-shot
    runner. The main benchmark still relies on unpaired Wilcoxon (paired on
    dataset only, not on seed). This script applies the paired-by-seed version
    to the main benchmark so the headline 15-dataset results get the same
    statistical rigour as the new few-shot multi-seed run.

Run:
    python scripts/run_paired_stats_on_benchmark.py
    python scripts/run_paired_stats_on_benchmark.py --metric pr_auc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tabular_gauge.evaluation.paired_stats import (
    per_dataset_paired_pvalues,
    per_n_paired_pvalues,
    paired_bootstrap_diff,
    cliffs_delta,
    holm_bonferroni,
)

CSV_PATH  = ROOT / "experiments" / "results" / "aggregated" / "results.csv"
OUT_CSV   = ROOT / "experiments" / "results" / "aggregated" / "paired_stats.csv"
OUT_MD    = ROOT / "reports" / "appendix_paired_stats.md"

REF_MODEL = "TabPFN"            # headline reference for all paired tests
ALPHA     = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="roc_auc",
                    help="Metric to analyse (default roc_auc)")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=[args.metric, "seed", "model", "dataset"])
    df["seed"] = df["seed"].astype(int)

    print(f"Loaded {CSV_PATH.relative_to(ROOT)}: {len(df)} rows, "
          f"{df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['seed'].nunique()} seeds "
          f"({sorted(df['seed'].unique())})")
    print(f"Reference model: {REF_MODEL}, metric: {args.metric}, alpha={ALPHA}\n")

    # ------------------------------------------------------------------
    # 1. Per-dataset paired Wilcoxon (paired on seed)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. PAIRED WILCOXON PER DATASET  (TabPFN vs every model, paired on seed)")
    print("=" * 70)
    per_ds = per_dataset_paired_pvalues(df, metric=args.metric, ref_model=REF_MODEL)
    if per_ds.empty:
        sys.exit("No per-dataset paired comparisons could be computed.")

    # ------------------------------------------------------------------
    # 2. Overall paired Wilcoxon (one row per model across all datasets)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. OVERALL PAIRED WILCOXON  (TabPFN vs every model, all (dataset, seed) pairs)")
    print("=" * 70)

    rows = []
    ref_df = df[df["model"] == REF_MODEL].set_index(["dataset", "seed"])[args.metric]
    for model in sorted(df["model"].unique()):
        if model == REF_MODEL:
            continue
        other_df = df[df["model"] == model].set_index(["dataset", "seed"])[args.metric]
        paired_idx = ref_df.index.intersection(other_df.index)
        if len(paired_idx) < 2:
            continue
        a = ref_df.loc[paired_idx].values
        b = other_df.loc[paired_idx].values
        from scipy import stats
        try:
            w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_p = 0.0, 1.0
        boot = paired_bootstrap_diff(a, b, n_resamples=2000)
        rows.append({
            "model": model,
            "n_pairs": len(paired_idx),
            "mean_tabpfn": float(a.mean()),
            "mean_other": float(b.mean()),
            "diff_mean": float(a.mean() - b.mean()),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cliffs_delta": cliffs_delta(a, b),
            "prob_positive_ref": boot["prob_positive"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
        })
    overall = pd.DataFrame(rows)
    if not overall.empty:
        overall["holm_p_adj"] = holm_bonferroni(overall["wilcoxon_p"].values)
        overall["significant"] = overall["holm_p_adj"] < ALPHA
        overall = overall.sort_values("wilcoxon_p").reset_index(drop=True)
        print(overall.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Combine into one tidy CSV (unify the two column conventions)
    # ------------------------------------------------------------------
    per_ds_compact = per_ds.copy()
    per_ds_compact["scope"] = "per_dataset"
    per_ds_compact = per_ds_compact.rename(columns={"n_pairs": "n_pairs"})

    overall_compact = overall.copy()
    overall_compact["scope"] = "overall"
    overall_compact["dataset"] = "ALL"
    # unify mean columns to mean_ref / mean_other
    overall_compact = overall_compact.rename(
        columns={"mean_tabpfn": "mean_ref", "n_pairs_seed": "n_pairs"}
    )

    common_cols = [
        "scope", "dataset", "model", "n_pairs",
        "mean_ref", "mean_other", "diff_mean",
        "wilcoxon_stat", "wilcoxon_p",
        "cliffs_delta", "prob_positive_ref", "ci_low", "ci_high",
        "holm_p_adj", "significant",
    ]
    combined = pd.concat(
        [per_ds_compact[common_cols], overall_compact[common_cols]],
        ignore_index=True,
    )
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} ({len(combined)} rows)")

    # ------------------------------------------------------------------
    # 4. Markdown summary for the paper appendix
    # ------------------------------------------------------------------
    md = []
    P = md.append
    P("# Appendix — Paired Statistical Tests on the 15-Dataset Benchmark\n\n")
    P(f"All tests are paired on the (dataset, seed) cell. Reference model: "
      f"**{REF_MODEL}**. Metric: **{args.metric}**. "
      f"Family-wise correction: Holm-Bonferroni at alpha = {ALPHA}.\n\n")
    P(f"Source data: `{CSV_PATH.relative_to(ROOT)}` "
      f"({len(df)} rows; {df['seed'].nunique()} seeds; "
      f"{df['model'].nunique()} models; {df['dataset'].nunique()} datasets).\n\n")

    P("## A.1 Overall paired comparison (all (dataset, seed) pairs)\n\n")
    if not overall.empty:
        P("| Model | n_pairs | mean TabPFN | mean other | diff | "
          "Wilcoxon p | Holm p | Cliff's δ | prob(TabPFN>other) | "
          "95% CI of diff | Significant |\n")
        P("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in overall.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['model']} | {int(r['n_pairs'])} | "
              f"{r['mean_tabpfn']:.4f} | {r['mean_other']:.4f} | "
              f"{r['diff_mean']:+.4f} | "
              f"{r['wilcoxon_p']:.4g} | {r['holm_p_adj']:.4g} | "
              f"{r['cliffs_delta']:+.3f} | "
              f"{r['prob_positive_ref']:.3f} | "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
              f"{sig} |\n")

    P("\n## A.2 Per-dataset paired comparison\n\n")
    if not per_ds.empty:
        P("Each row is one (dataset, model) pair, paired on seed.\n\n")
        P("| Dataset | Model | n_pairs | mean TabPFN | mean other | diff | "
          "Wilcoxon p | Holm p | Cliff's δ | Significant |\n")
        P("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in per_ds.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['dataset']} | {r['model']} | {int(r['n_pairs'])} | "
              f"{r['mean_ref']:.4f} | {r['mean_other']:.4f} | "
              f"{r['diff_mean']:+.4f} | "
              f"{r['wilcoxon_p']:.4g} | {r['holm_p_adj']:.4g} | "
              f"{r['cliffs_delta']:+.3f} | {sig} |\n")

    P("\n## A.3 Notes\n\n")
    P("* `Wilcoxon p` is the two-sided paired signed-rank test on the seed-paired "
      "differences.\n")
    P("* `Holm p` is the Holm-Bonferroni step-down correction applied across the "
      "comparisons in the same scope (overall or per-dataset).\n")
    P("* `Cliff's δ ∈ [-1, 1]`: |δ|<0.147 negligible, <0.33 small, <0.474 medium, "
      "otherwise large (Romano et al. 2006).\n")
    P("* `prob(TabPFN>other)` is the posterior probability that TabPFN beats the "
      "comparison model, estimated by 2000-resample paired bootstrap.\n")
    P("* `95% CI of diff` is the percentile CI on the mean paired difference "
      "TabPFN − other.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()