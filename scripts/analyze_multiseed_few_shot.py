"""
Run paired statistical tests (Wilcoxon, bootstrap CI, Holm-Bonferroni, Cliff's δ)
on the multi-seed few-shot benchmark (5 models × 6 datasets × 9 N values × 5 seeds).

Inputs : experiments/results/few_shot/multiseed_few_shot_results.csv
         (5 models × 6 datasets × 9 train sizes × 5 seeds = 1,350 cells when full)
Outputs:
    experiments/results/few_shot/multiseed_paired_stats_per_n.csv
    experiments/results/few_shot/multiseed_paired_stats_per_dataset.csv
    experiments/results/few_shot/multiseed_paired_stats_overall.csv
    reports/appendix_multiseed_few_shot.md

Why this exists:
    scripts/run_multiseed_few_shot.py prints the paired-stats tables to stdout
    at the end of each run but does NOT save them. This script re-runs the
    analysis on the accumulated CSV and writes the full results to disk so
    they can be cited in the report and figures.

Run:
    python scripts/analyze_multiseed_few_shot.py
    python scripts/analyze_multiseed_few_shot.py --metric pr_auc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from tabular_gauge.evaluation.paired_stats import (
    per_n_paired_pvalues,
    per_dataset_paired_pvalues,
    paired_bootstrap_diff,
    cliffs_delta,
    holm_bonferroni,
)


CSV_PATH  = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
OUT_DIR   = ROOT / "experiments" / "results" / "few_shot"
OUT_PER_N      = OUT_DIR / "multiseed_paired_stats_per_n.csv"
OUT_PER_DS     = OUT_DIR / "multiseed_paired_stats_per_dataset.csv"
OUT_OVERALL    = OUT_DIR / "multiseed_paired_stats_overall.csv"
OUT_MD         = ROOT / "reports" / "appendix_multiseed_few_shot.md"

REF_MODEL = "TabPFN"
ALPHA     = 0.05

# Canonical ordering of train sizes for display (small → large, 'full' last)
TRAIN_SIZE_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]


def _sort_train_size(s: pd.Series) -> pd.Series:
    """Sort key: integers first in ascending order, 'full' last."""
    return s.map(lambda x: (str(x) == "full", int(x) if str(x) != "full" else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="roc_auc",
                    help="Metric to analyse (default roc_auc)")
    ap.add_argument("--n_resamples", type=int, default=2000,
                    help="Bootstrap resamples per comparison (default 2000)")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}. Run scripts/run_multiseed_few_shot.py first.")

    df = pd.read_csv(CSV_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=[args.metric, "seed", "model", "dataset", "train_size"])
    df["seed"] = df["seed"].astype(int)

    n_seeds    = df["seed"].nunique()
    n_models   = df["model"].nunique()
    n_datasets = df["dataset"].nunique()
    n_sizes    = df["train_size"].nunique()

    print(f"Loaded {CSV_PATH.relative_to(ROOT)}: {len(df)} rows, "
          f"{n_models} models, {n_datasets} datasets, "
          f"{n_sizes} train sizes, {n_seeds} seeds "
          f"({sorted(df['seed'].unique())})")
    print(f"Reference model: {REF_MODEL}, metric: {args.metric}, alpha={ALPHA}\n")

    if n_seeds < 2:
        sys.exit(f"Need at least 2 seeds for paired analysis (have {n_seeds}).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Per-N paired Wilcoxon (TabPFN vs every model, paired on dataset × seed)
    # ------------------------------------------------------------------
    # The library function uses dataset_col + model_col by default; we slice by
    # train_size, so each group is one (model, dataset, seed) per N.
    print("=" * 70)
    print(f"1. PER-N PAIRED WILCOXON  (TabPFN vs every model, paired on dataset×seed)")
    print("=" * 70)
    per_n = per_n_paired_pvalues(df, metric=args.metric, ref_model=REF_MODEL)
    if per_n.empty:
        sys.exit("No per-N paired comparisons could be computed.")
    # Re-sort by canonical train_size order
    per_n = per_n.sort_values(
        by="train_size", key=_sort_train_size
    ).reset_index(drop=True)
    per_n.to_csv(OUT_PER_N, index=False)
    print(f"Wrote {OUT_PER_N.relative_to(ROOT)} ({len(per_n)} rows)")
    print(f"  Distinct N values: {per_n['train_size'].nunique()}")
    print(f"  Holm-significant rows: {int(per_n['significant'].sum())} / {len(per_n)}")
    print()

    # ------------------------------------------------------------------
    # 2. Per-dataset paired Wilcoxon (paired on seed, collapsed across N)
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"2. PER-DATASET PAIRED WILCOXON  (TabPFN vs every model, paired on seed)")
    print("=" * 70)
    per_ds = per_dataset_paired_pvalues(df, metric=args.metric, ref_model=REF_MODEL)
    if per_ds.empty:
        sys.exit("No per-dataset paired comparisons could be computed.")
    per_ds = per_ds.sort_values(["dataset", "wilcoxon_p"]).reset_index(drop=True)
    per_ds.to_csv(OUT_PER_DS, index=False)
    print(f"Wrote {OUT_PER_DS.relative_to(ROOT)} ({len(per_ds)} rows)")
    print(f"  Distinct datasets: {per_ds['dataset'].nunique()}")
    print(f"  Holm-significant rows: {int(per_ds['significant'].sum())} / {len(per_ds)}")
    print()

    # ------------------------------------------------------------------
    # 3. Overall paired Wilcoxon (one row per model, all cells pooled)
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"3. OVERALL PAIRED WILCOXON  (TabPFN vs every model, all "
          f"(dataset, N, seed) pairs)")
    print("=" * 70)
    rows = []
    # Pairing key combines all three nuisance axes
    pair_idx = ["dataset", "train_size", "seed"]
    ref_df = df[df["model"] == REF_MODEL].set_index(pair_idx)[args.metric]
    for model in sorted(df["model"].unique()):
        if model == REF_MODEL:
            continue
        other_df = df[df["model"] == model].set_index(pair_idx)[args.metric]
        paired_idx = ref_df.index.intersection(other_df.index)
        if len(paired_idx) < 2:
            print(f"  [skip] {model}: only {len(paired_idx)} paired cell(s)")
            continue
        a = ref_df.loc[paired_idx].values
        b = other_df.loc[paired_idx].values
        try:
            w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_p = 0.0, 1.0
        boot = paired_bootstrap_diff(a, b, n_resamples=args.n_resamples)
        rows.append({
            "model": model,
            "n_pairs": len(paired_idx),
            "mean_ref": float(a.mean()),
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
        overall.to_csv(OUT_OVERALL, index=False)
        print(f"Wrote {OUT_OVERALL.relative_to(ROOT)} ({len(overall)} rows)")
        print(f"  Holm-significant rows: {int(overall['significant'].sum())} / {len(overall)}")
        print()
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(overall.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Markdown summary for the report appendix
    # ------------------------------------------------------------------
    md = []
    P = md.append
    P("# Appendix — Multi-Seed Few-Shot Paired Statistical Tests\n\n")
    P(f"All tests are paired on the **seed** dimension. The pairing "
      f"unit is one `(dataset, train_size, seed)` cell — the same dataset, "
      f"the same training-set size, and the same random seed is used for "
      f"every model so the within-cell variance cancels. Reference model: "
      f"**{REF_MODEL}**. Metric: **{args.metric}**. "
      f"Family-wise correction: Holm-Bonferroni at alpha = {ALPHA}.\n\n")
    P(f"Source data: `{CSV_PATH.relative_to(ROOT)}` "
      f"({len(df)} rows; {n_seeds} seeds; {n_models} models; "
      f"{n_datasets} datasets; {n_sizes} train sizes).\n\n")
    P(f"Reproduce: `python scripts/analyze_multiseed_few_shot.py` "
      f"(optionally `--metric <name>`).\n\n")

    P("## B.1 Overall paired comparison (all `(dataset, train_size, seed)` pairs)\n\n")
    if not overall.empty:
        P("| Model | n_pairs | mean TabPFN | mean other | diff | "
          "Wilcoxon p | Holm p | Cliff's δ | prob(TabPFN>other) | "
          "95% CI of diff | Significant |\n")
        P("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in overall.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['model']} | {int(r['n_pairs'])} | "
              f"{r['mean_ref']:.4f} | {r['mean_other']:.4f} | "
              f"{r['diff_mean']:+.4f} | "
              f"{r['wilcoxon_p']:.4g} | {r['holm_p_adj']:.4g} | "
              f"{r['cliffs_delta']:+.3f} | "
              f"{r['prob_positive_ref']:.3f} | "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
              f"{sig} |\n")

    P("\n## B.2 Per-train-size paired comparison (paired on dataset×seed)\n\n")
    if not per_n.empty:
        P("Each row is one `(train_size, model)` pair, paired on `(dataset, seed)`. "
          "The number of pairs per row is `n_datasets × n_seeds` (typically "
          f"{n_datasets} × {n_seeds} = {n_datasets * n_seeds} for fully "
          "populated cells).\n\n")
        P("| Train size | Model | n_pairs | mean TabPFN | mean other | diff | "
          "Wilcoxon p | Holm p | Cliff's δ | Significant |\n")
        P("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in per_n.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['train_size']} | {r['model']} | {int(r['n_pairs'])} | "
              f"{r['mean_ref']:.4f} | {r['mean_other']:.4f} | "
              f"{r['diff_mean']:+.4f} | "
              f"{r['wilcoxon_p']:.4g} | {r['holm_p_adj']:.4g} | "
              f"{r['cliffs_delta']:+.3f} | {sig} |\n")

    P("\n## B.3 Per-dataset paired comparison (paired on seed, across all N)\n\n")
    if not per_ds.empty:
        P("Each row is one `(dataset, model)` pair, paired on `seed` "
          "(one observation per `(dataset, train_size, seed)` cell, so "
          f"`n_pairs = n_train_sizes × n_seeds` here).\n\n")
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

    P("\n## B.4 Notes\n\n")
    P("* `Wilcoxon p` is the two-sided paired signed-rank test on the "
      "seed-paired differences.\n")
    P("* `Holm p` is the Holm-Bonferroni step-down correction applied across "
      "the comparisons **within the same scope** (per-N, per-dataset, or "
      "overall). Corrections are NOT pooled across scopes.\n")
    P("* `Cliff's δ ∈ [-1, 1]`: |δ|<0.147 negligible, <0.33 small, "
      "<0.474 medium, otherwise large (Romano et al. 2006).\n")
    P("* `prob(TabPFN>other)` is the posterior probability that TabPFN beats "
      "the comparison model, estimated by "
      f"{args.n_resamples}-resample paired bootstrap.\n")
    P("* `95% CI of diff` is the percentile CI on the mean paired difference "
      "TabPFN − other.\n")
    P("* SAP-RPT-1 is not present in this benchmark — it is only evaluated "
      "in the main 15-dataset benchmark because it does not require a "
      "training set and is not sensitive to N.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
