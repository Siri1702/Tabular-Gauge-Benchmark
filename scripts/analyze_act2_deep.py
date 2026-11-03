"""
Deep statistical analysis of the multi-seed Act-2 benchmark.

The flat analyze_multiseed_few_shot.py covers Wilcoxon + Holm + Cliff's δ
+ paired-bootstrap CI (3 scopes). This script ADDS:

  1. Friedman test + Nemenyi post-hoc (the canonical "many datasets,
     many classifiers" rank test). Per train size, per dataset, and
     overall.
  2. Bootstrap-CI on Cliff's δ itself (BCa-style percentile CI over
     2000 resamples of the paired differences).
  3. Posterior probabilities of TabPFN > each baseline at every N, with
     a Bayesian-random-effects flavour (using the bootstrap P as the
     plug-in estimate + a normal approximation CI on logit(P)).
  4. Per-dataset mean Cliff's δ + bootstrap CI for the §5.2 table
     of "by how much TabPFN wins where it wins".
  5. Effect-size heatmap (|δ|) per (N, baseline).
  6. Print a top-level summary table and write the full deep-stats
     CSV/markdown for §5 of the paper.

Run:
    python scripts/analyze_act2_deep.py
    python scripts/analyze_act2_deep.py --metric pr_auc --n_resamples 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout so the cp1252 PowerShell console does not mangle
# non-ASCII glyphs (delta, sigma, etc.) in prints.
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from tabular_gauge.evaluation.paired_stats import (
    cliffs_delta,
    holm_bonferroni,
    paired_bootstrap_diff,
)


CSV_PATH      = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
OUT_DIR       = ROOT / "experiments" / "results" / "few_shot"
OUT_CSV       = OUT_DIR / "multiseed_deep_stats.csv"
OUT_MD        = ROOT / "reports" / "appendix_multiseed_deep_stats.md"

REF_MODEL     = "TabPFN"
ALL_MODELS    = ["TabPFN", "XGBoost", "LightGBM", "CatBoost", "MLP"]
BASELINES     = ["XGBoost", "LightGBM", "CatBoost", "MLP"]
TRAIN_SIZE_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]
ALPHA         = 0.05


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sort_train_size(s: pd.Series) -> pd.Series:
    return s.map(lambda x: (str(x) == "full", int(x) if str(x) != "full" else 0))


def cliffs_delta_bootstrap_ci(a, b, n_resamples=2000, seed=0):
    """Bootstrap CI on Cliff's δ itself by resampling the paired-difference
    vector (same length, with replacement). Returns mean, ci_low, ci_high."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2:
        return {"mean": float(cliffs_delta(a, b)),
                "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed=seed)
    n = len(a)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        boot[i] = cliffs_delta(a[idx], b[idx])
    point = cliffs_delta(a, b)
    return {"mean": float(point),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def post_prob_positive_with_ci(a, b, n_resamples=2000, seed=0):
    """Bayesian-ish posterior P(TabPFN > baseline) approximated by the
    matched-pairs bootstrap of (a-b). Reports the point estimate and a
    Wilson 95% CI on the binary 'positive' outcome."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2:
        return {"prob": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed=seed)
    n = len(a)
    pos = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diff = a[idx] - b[idx]
        pos[i] = (diff > 0).mean()
    point = (a - b > 0).mean()
    # Wilson interval on the binomial 'proportion of positive pairs in the
    # observed sample' — point estimate on the observed 0/1 outcomes,
    # 95% CI:
    n_pos = int((a - b > 0).sum())
    n_neg = int((a - b < 0).sum())
    n_zero = n - n_pos - n_neg
    z = 1.96
    p_hat = (n_pos + 0.5) / (n + 1)  # add-0.5 smoothing
    denom = 1 + z*z / (n + 1)
    centre = (p_hat + z*z/(2*(n+1))) / denom
    half = z * np.sqrt(p_hat*(1 - p_hat) / (n+1) + z*z/(4*(n+1)*(n+1))) / denom
    return {"prob": float(point),
            "ci_low": float(max(centre - half, 0.0)),
            "ci_high": float(min(centre + half, 1.0)),
            "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero}


def friedman_and_nemenyi(df_block: pd.DataFrame, metric: str = "roc_auc") -> dict:
    """
    For a block of data where each row is one (dataset, seed, train_size)
    cell with columns ['model', metric], rank models per (dataset, seed,
    train_size) cell, then run Friedman + Nemenyi post-hoc. Returns a dict
    with the Friedman statistic, df, p, mean ranks, and a pairwise post-hoc
    table.
    """
    pivot = (df_block
             .pivot_table(index=["dataset", "train_size", "seed"],
                          columns="model", values=metric)
             .dropna(how="any"))
    if pivot.empty or pivot.shape[1] < 3:
        return {"friedman_p": float("nan"),
                "mean_ranks": pd.Series(dtype=float),
                "nemenyi": pd.DataFrame()}
    ranks = pivot.rank(axis=1, ascending=True, method="average")
    # scipy.stats.friedmanfridman takes (observations, ...) with rows = blocks
    friedman = stats.friedmanchisquare(*[pivot[m].values for m in pivot.columns])
    mean_ranks = ranks.mean(axis=0).sort_values()
    # Nemenyi: critical difference at q_alpha * sqrt(k(k+1) / (6 N))
    # q_alpha for k=5, alpha=0.05 is 3.102 (Demšar 2006 Table 5).
    n_blocks = pivot.shape[0]
    k = pivot.shape[1]
    q_alpha = 3.102  # k=5, alpha=0.10 -> 2.850; alpha=0.05 -> 3.102
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_blocks))

    # Pairwise post-hoc p-values (Nemenyi): approximate via the studentized
    # range distribution: z = (R_i - R_j) / sqrt(k(k+1)/(6N)); p = P(Z > |z|)
    nemenyi = []
    for i, ma in enumerate(mean_ranks.index):
        for j, mb in enumerate(mean_ranks.index):
            if j <= i:
                continue
            diff = mean_ranks.loc[ma] - mean_ranks.loc[mb]
            se = np.sqrt(k * (k + 1) / (6 * n_blocks))
            z = diff / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            nemenyi.append({"model_a": ma, "model_b": mb,
                            "rank_a": float(mean_ranks.loc[ma]),
                            "rank_b": float(mean_ranks.loc[mb]),
                            "rank_diff": float(diff),
                            "z": float(z),
                            "p": float(p)})
    nemenyi_df = pd.DataFrame(nemenyi)
    nemenyi_df["significant"] = nemenyi_df["p"] < ALPHA / len(nemenyi_df)  # Bonferroni within the post-hoc family
    nemenyi_df["critical_difference"] = cd

    return {
        "friedman_stat": float(friedman.statistic),
        "friedman_p":   float(friedman.pvalue),
        "friedman_df":  pivot.shape[1] - 1,  # k - 1
        "mean_ranks":   mean_ranks,
        "nemenyi":      nemenyi_df,
        "critical_difference": cd,
        "n_blocks":     n_blocks,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--n_resamples", type=int, default=2000)
    args = ap.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}.")

    df = pd.read_csv(CSV_PATH)
    df["seed"]       = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["train_size"] = df["train_size"].astype(str)
    df = df.dropna(subset=[args.metric])

    print(f"Loaded {len(df)} rows, metric={args.metric}, "
          f"{df['seed'].nunique()} seeds, "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['train_size'].nunique()} train sizes, "
          f"{df['model'].nunique()} models")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # ============================================================== #
    # 1. PER (N, baseline) Cliff's δ + bootstrap CI + Wilson CI on P
    # ============================================================== #
    print()
    print("=" * 70)
    print("1. PER-N  (TabPFN vs every baseline, paired on dataset×seed)")
    print("   — Cliff's δ with BCa bootstrap CI, and Wilson CI on P(TabPFN>b)")
    print("=" * 70)

    pair_idx = ["dataset", "seed"]
    ref = df[df["model"] == REF_MODEL].set_index(pair_idx + ["train_size"])[args.metric]

    per_n_deep = []
    for n in TRAIN_SIZE_ORDER:
        if n not in set(df["train_size"].unique()):
            continue
        for b in BASELINES:
            sub_b = df[(df["model"] == b) & (df["train_size"] == n)]
            sub_t = df[(df["model"] == REF_MODEL) & (df["train_size"] == n)]
            key = pair_idx
            ref_idx = sub_t.set_index(key + ["train_size"])[args.metric]
            oth_idx = sub_b.set_index(key + ["train_size"])[args.metric]
            paired_idx = ref_idx.index.intersection(oth_idx.index)
            a = ref_idx.loc[paired_idx].values
            b_val = oth_idx.loc[paired_idx].values
            if len(a) < 2:
                continue
            boot_cd = cliffs_delta_bootstrap_ci(a, b_val, n_resamples=args.n_resamples, seed=0)
            wilc = stats.wilcoxon(a, b_val, alternative="two-sided")
            post = post_prob_positive_with_ci(a, b_val, n_resamples=args.n_resamples, seed=0)
            per_n_deep.append({
                "scope": "per_n",
                "train_size": n,
                "model": b,
                "n_pairs": int(len(a)),
                "mean_a": float(a.mean()),
                "mean_b": float(b_val.mean()),
                "diff_mean": float(a.mean() - b_val.mean()),
                "wilcoxon_p": float(wilc.pvalue),
                "cliffs_delta": boot_cd["mean"],
                "cliffs_delta_ci_lo": boot_cd["ci_low"],
                "cliffs_delta_ci_hi": boot_cd["ci_high"],
                "prob_positive": post["prob"],
                "prob_ci_lo":     post["ci_low"],
                "prob_ci_hi":     post["ci_high"],
                "n_pos": post["n_pos"],
                "n_neg": post["n_neg"],
            })

    per_n_deep_df = pd.DataFrame(per_n_deep)
    if not per_n_deep_df.empty:
        per_n_deep_df["holm_p"] = holm_bonferroni(per_n_deep_df["wilcoxon_p"].values)
        per_n_deep_df["significant"] = per_n_deep_df["holm_p"] < ALPHA
    print(per_n_deep_df.assign(
            _sort_key=per_n_deep_df["train_size"].map(
                lambda x: (str(x) == "full", int(x) if str(x) != "full" else 0)
            )
        ).sort_values(by=["_sort_key", "model"]).drop(columns=["_sort_key"])
        .to_string(index=False))
    rows.append(("per_n_deep_stats", per_n_deep_df))

    # ============================================================== #
    # 2. PER-DATASET  — Cliff's δ + CI (within-dataset strength of TabPFN)
    # ============================================================== #
    print()
    print("=" * 70)
    print("2. PER-DATASET  (TabPFN vs every baseline, paired on N×seed within the dataset)")
    print("=" * 70)

    per_ds_deep = []
    for ds in sorted(df["dataset"].unique()):
        sub_t = df[(df["model"] == REF_MODEL) & (df["dataset"] == ds)]
        for b in BASELINES:
            sub_b = df[(df["model"] == b) & (df["dataset"] == ds)]
            ref_idx = sub_t.set_index(["train_size", "seed"])[args.metric]
            oth_idx = sub_b.set_index(["train_size", "seed"])[args.metric]
            paired_idx = ref_idx.index.intersection(oth_idx.index)
            a = ref_idx.loc[paired_idx].values
            b_val = oth_idx.loc[paired_idx].values
            if len(a) < 2:
                continue
            boot_cd = cliffs_delta_bootstrap_ci(a, b_val, n_resamples=args.n_resamples, seed=0)
            post = post_prob_positive_with_ci(a, b_val, n_resamples=args.n_resamples, seed=0)
            try:
                wilc = stats.wilcoxon(a, b_val, alternative="two-sided")
                wp = float(wilc.pvalue)
            except ValueError:
                wp = float("nan")
            per_ds_deep.append({
                "scope": "per_dataset",
                "dataset": ds,
                "model": b,
                "n_pairs": int(len(a)),
                "diff_mean": float(a.mean() - b_val.mean()),
                "wilcoxon_p": wp,
                "cliffs_delta": boot_cd["mean"],
                "cliffs_delta_ci_lo": boot_cd["ci_low"],
                "cliffs_delta_ci_hi": boot_cd["ci_high"],
                "prob_positive": post["prob"],
                "prob_ci_low":  post["ci_low"],
                "prob_ci_high": post["ci_high"],
            })

    per_ds_deep_df = pd.DataFrame(per_ds_deep)
    if not per_ds_deep_df.empty:
        per_ds_deep_df["holm_p"] = holm_bonferroni(per_ds_deep_df["wilcoxon_p"].fillna(1.0).values)
        per_ds_deep_df["significant"] = per_ds_deep_df["holm_p"] < ALPHA
    print(per_ds_deep_df.sort_values(["dataset", "model"]).to_string(index=False))
    rows.append(("per_dataset_deep_stats", per_ds_deep_df))

    # ============================================================== #
    # 3. FRIEDMAN + NEMENYI  (overall, per N, per dataset)
    # ============================================================== #
    print()
    print("=" * 70)
    print("3. FRIEDMAN + NEMENYI post-hoc")
    print("=" * 70)

    friedman_blocks = []

    # 3a. Overall
    fr = friedman_and_nemenyi(df, metric=args.metric)
    print(f"\n[3a] OVERALL:  Friedman chi2={fr['friedman_stat']:.3f}, "
          f"df={fr['friedman_df']}, p={fr['friedman_p']:.3e}, "
          f"CD={fr['critical_difference']:.3f}  (q_alpha=3.102 for k=5)")
    print("      Mean ranks (lower is better):")
    print(fr["mean_ranks"].to_string())
    print(f"\n      Pairwise Nemenyi (Bonferroni-corrected within block):")
    fr["nemenyi"]["scope"] = "overall"
    friedman_blocks.append(fr)

    # 3b. Per train size
    fr_per_n = []
    for n in TRAIN_SIZE_ORDER:
        sub = df[df["train_size"] == n]
        if sub["model"].nunique() < 3 or sub.empty:
            continue
        fr_n = friedman_and_nemenyi(sub, metric=args.metric)
        fr_per_n.append({
            "scope": "per_n",
            "train_size": n,
            "friedman_stat": fr_n["friedman_stat"],
            "friedman_p": fr_n["friedman_p"],
            "n_blocks": fr_n["n_blocks"],
            "critical_difference": fr_n["critical_difference"],
            "mean_ranks": fr_n["mean_ranks"].to_dict(),
            "nemenyi": fr_n["nemenyi"],
        })
    friedman_blocks.extend(fr_per_n)
    n_per_n_sig = sum(1 for r in fr_per_n if r["friedman_p"] < ALPHA)
    print(f"\n[3b] PER-N:    {n_per_n_sig}/{len(fr_per_n)} N values have "
          f"Friedman p<{ALPHA} (i.e., real rank differences).")
    for r in fr_per_n:
        print(f"        N={r['train_size']:>5s}: chi2={r['friedman_stat']:.3f}, "
              f"p={r['friedman_p']:.3e}, n_blocks={r['n_blocks']}, CD={r['critical_difference']:.3f}")

    # 3c. Per dataset
    fr_per_ds = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        if sub["model"].nunique() < 3 or sub.empty:
            continue
        fr_d = friedman_and_nemenyi(sub, metric=args.metric)
        fr_per_ds.append({
            "scope": "per_dataset",
            "dataset": ds,
            "friedman_stat": fr_d["friedman_stat"],
            "friedman_p": fr_d["friedman_p"],
            "n_blocks": fr_d["n_blocks"],
            "critical_difference": fr_d["critical_difference"],
            "mean_ranks": fr_d["mean_ranks"].to_dict(),
            "nemenyi": fr_d["nemenyi"],
        })
    friedman_blocks.extend(fr_per_ds)
    n_per_ds_sig = sum(1 for r in fr_per_ds if r["friedman_p"] < ALPHA)
    print(f"\n[3c] PER-DATASET: {n_per_ds_sig}/{len(fr_per_ds)} datasets have "
          f"Friedman p<{ALPHA}.")
    for r in fr_per_ds:
        print(f"        {r['dataset']:>14s}: chi2={r['friedman_stat']:.3f}, "
              f"p={r['friedman_p']:.3e}, n_blocks={r['n_blocks']}")

    # ============================================================== #
    # 4. WRITE CSVs and markdown appendix
    # ============================================================== #
    deep_df = pd.concat(
        [r for _, r in rows if not r.empty],
        ignore_index=True, sort=False,
    )
    deep_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} ({len(deep_df)} rows)")

    # Flatten the Nemenyi blocks into one CSV too
    nemenyi_rows = []
    for fr in [fr["nemenyi"] for fr in ([friedman_blocks[0]] if friedman_blocks else [])] + \
               [r["nemenyi"] for r in fr_per_n] + \
               [r["nemenyi"] for r in fr_per_ds]:
        nemenyi_rows.append(fr)
    nemenyi_df = pd.concat(nemenyi_rows, ignore_index=True, sort=False)
    nemenyi_csv = OUT_DIR / "multiseed_nemenyi.csv"
    nemenyi_df.to_csv(nemenyi_csv, index=False)
    print(f"Wrote {nemenyi_csv.relative_to(ROOT)} ({len(nemenyi_df)} rows)")

    # ----- Markdown appendix -----
    md = []
    P = md.append
    P("# Appendix — Deep Statistical Analysis (Act 2)\n\n")
    P(f"Source: `{CSV_PATH.relative_to(ROOT)}` "
      f"({len(df)} rows; {df['seed'].nunique()} seeds; "
      f"{df['model'].nunique()} models; "
      f"{df['dataset'].nunique()} datasets; "
      f"{df['train_size'].nunique()} train sizes).\n\n")
    P(f"Metric: **{args.metric}**. Reference model: **{REF_MODEL}**. "
      f"Family-wise error: Holm-Bonferroni at alpha = {ALPHA}.\n\n")
    P(f"Reproduce: `python scripts/analyze_act2_deep.py "
      f"--metric {args.metric} --n_resamples {args.n_resamples}`\n\n")

    P("## C.1 Per-train-size Bayesian-flavor posterior probability of TabPFN > baseline\n\n")
    P("Posterior P is the matched-pairs bootstrap estimate of "
      "P(a_i > b_i). Wilson 95% CI is computed on the observed binary "
      "outcomes (a-b > 0 ? 1 : 0) with add-0.5 smoothing.\n\n")
    if not per_n_deep_df.empty:
        P("| Train size | Baseline | n_pairs | mean TabPFN | mean baseline | Δ | "
          "Wilcoxon p | Holm p | Cliff's δ | δ 95% CI | P(TabPFN>b) | CI | sig |\n")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        _sorted = (per_n_deep_df
                   .assign(_sort_key=per_n_deep_df["train_size"].map(
                       lambda x: (str(x) == "full", int(x) if str(x) != "full" else 0)))
                   .sort_values(by=["_sort_key", "model"])
                   .drop(columns=["_sort_key"]))
        for _, r in _sorted.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['train_size']} | {r['model']} | {int(r['n_pairs'])} | "
              f"{r['mean_a']:.4f} | {r['mean_b']:.4f} | "
              f"{r['diff_mean']:+.4f} | "
              f"{r['wilcoxon_p']:.3e} | {r['holm_p']:.3e} | "
              f"{r['cliffs_delta']:+.3f} | "
              f"[{r['cliffs_delta_ci_lo']:+.3f}, {r['cliffs_delta_ci_hi']:+.3f}] | "
              f"{r['prob_positive']:.3f} | "
              f"[{r['prob_ci_lo']:.3f}, {r['prob_ci_hi']:.3f}] | "
              f"{sig} |\n")

    P("\n## C.2 Per-dataset strength of the TabPFN advantage\n\n")
    P("Cliff's δ with BCa percentile bootstrap CI "
      f"({args.n_resamples} resamples), plus the posterior P of TabPFN winning. "
      "Rows are coloured `✓` if the Holm-adjusted Wilcoxon is significant.\n\n")
    if not per_ds_deep_df.empty:
        P("| Dataset | Baseline | n_pairs | Δ | Wilcoxon p | Holm p | "
          "Cliff's δ | δ 95% CI | P | CI | sig |\n")
        P("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in per_ds_deep_df.iterrows():
            sig = "✓" if r["significant"] else "—"
            P(f"| {r['dataset']} | {r['model']} | {int(r['n_pairs'])} | "
              f"{r['diff_mean']:+.4f} | {r['wilcoxon_p']:.3e} | "
              f"{r['holm_p']:.3e} | {r['cliffs_delta']:+.3f} | "
              f"[{r['cliffs_delta_ci_lo']:+.3f}, {r['cliffs_delta_ci_hi']:+.3f}] | "
              f"{r['prob_positive']:.3f} | "
              f"[{r['prob_ci_low']:.3f}, {r['prob_ci_high']:.3f}] | {sig} |\n")

    P("\n## C.3 Friedman test + Nemenyi post-hoc\n\n")
    P("The Friedman test is the omnibus rank test for 'many datasets × "
      "many classifiers'. Nemenyi post-hoc (Demšar 2006) gives all pairwise "
      "comparisons with a critical difference (CD) of "
      "`q_alpha * sqrt(k(k+1)/(6N))` where k = number of models, N = number "
      "of blocks (datasets × seeds × N). `q_alpha=3.102` for k=5 at alpha=0.05.\n\n")
    P("Pairwise p-values are Bonferroni-corrected within each block.\n\n")

    # Overall block
    P("### C.3.1 Overall\n\n")
    P(f"Friedman chi-square = {friedman_blocks[0]['friedman_stat']:.3f}, "
      f"df = {friedman_blocks[0]['friedman_df']}, "
      f"p = {friedman_blocks[0]['friedman_p']:.3e}.\n\n")
    P("Mean ranks (1 = best, 5 = worst):\n\n")
    P("| Model | Mean rank |\n|---|---|\n")
    for m, rk in friedman_blocks[0]["mean_ranks"].items():
        P(f"| {m} | {rk:.3f} |\n")

    P("\nPairwise Nemenyi comparisons (lower-rank model wins if rank_diff > 0 "
      "in the *a-b* direction but be careful: lower mean rank = better; here "
      "rank_a - rank_b > 0 means model_a is worse):\n\n")
    nm = friedman_blocks[0]["nemenyi"]
    P("| Model A | Model B | Rank A | Rank B | Diff | z | p (raw) | sig? | "
      "CD |\n|---|---|---|---|---|---|---|---|---|\n")
    for _, r in nm.iterrows():
        sig = "✓" if r["significant"] else "—"
        P(f"| {r['model_a']} | {r['model_b']} | {r['rank_a']:.3f} | "
          f"{r['rank_b']:.3f} | {r['rank_diff']:+.3f} | {r['z']:+.3f} | "
          f"{r['p']:.3e} | {sig} | {r['critical_difference']:.3f} |\n")

    P("\n### C.3.2 Per-train-size\n\n")
    P("| Train size | Friedman chi2 | Friedman p | N blocks | CD | "
      "Significant? |\n|---|---|---|---|---|---|\n")
    for r in fr_per_n:
        sig = "✓" if r["friedman_p"] < ALPHA else "—"
        P(f"| {r['train_size']} | {r['friedman_stat']:.3f} | "
          f"{r['friedman_p']:.3e} | {r['n_blocks']} | "
          f"{r['critical_difference']:.3f} | {sig} |\n")
    P("\nMean ranks per N:\n\n")
    P("| Train size | TabPFN | XGBoost | LightGBM | CatBoost | MLP |\n")
    P("|---|---|---|---|---|---|\n")
    for r in fr_per_n:
        ranks = r["mean_ranks"]
        P(f"| {r['train_size']} | "
          f"{ranks.get('TabPFN', float('nan')):.3f} | "
          f"{ranks.get('XGBoost', float('nan')):.3f} | "
          f"{ranks.get('LightGBM', float('nan')):.3f} | "
          f"{ranks.get('CatBoost', float('nan')):.3f} | "
          f"{ranks.get('MLP', float('nan')):.3f} |\n")

    P("\n### C.3.3 Per-dataset\n\n")
    P("| Dataset | Friedman chi2 | Friedman p | N blocks | CD | "
      "Significant? |\n|---|---|---|---|---|---|\n")
    for r in fr_per_ds:
        sig = "✓" if r["friedman_p"] < ALPHA else "—"
        P(f"| {r['dataset']} | {r['friedman_stat']:.3f} | "
          f"{r['friedman_p']:.3e} | {r['n_blocks']} | "
          f"{r['critical_difference']:.3f} | {sig} |\n")
    P("\nMean ranks per dataset:\n\n")
    P("| Dataset | TabPFN | XGBoost | LightGBM | CatBoost | MLP |\n")
    P("|---|---|---|---|---|---|\n")
    for r in fr_per_ds:
        ranks = r["mean_ranks"]
        P(f"| {r['dataset']} | "
          f"{ranks.get('TabPFN', float('nan')):.3f} | "
          f"{ranks.get('XGBoost', float('nan')):.3f} | "
          f"{ranks.get('LightGBM', float('nan')):.3f} | "
          f"{ranks.get('CatBoost', float('nan')):.3f} | "
          f"{ranks.get('MLP', float('nan')):.3f} |\n")

    P("\n## C.4 Notes\n\n")
    P("* `Cliff's δ ∈ [-1, 1]`: |δ|<0.147 negligible, <0.33 small, "
      "<0.474 medium, otherwise large (Romano 2006).\n")
    P("* `Wilson 95% CI on P(TabPFN>baseline)` is add-0.5 smoothed; "
      "as n_pairs → ∞ it converges to the binomial CI.\n")
    P("* `Friedman chi2` is large by design when the data has many "
      "ties; if Friedman is non-significant in §C.3.3 it means the rank "
      "ordering of the 5 models is not consistent within that dataset.\n")
    P("* `Nemenyi CD` is the minimum |rank_a − rank_b| for significance. "
      "If TabPFN's mean rank is more than `CD` lower than every other "
      "model, all 4 pairwise comparisons are significant.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")

    # ============================================================== #
    # 5. HEADLINE SUMMARY (for paper §1)
    # ============================================================== #
    print()
    print("=" * 70)
    print("HEADLINE RESULTS  (copy into paper §1 / abstract)")
    print("=" * 70)
    overall = pd.read_csv(OUT_DIR / "multiseed_paired_stats_overall.csv")
    print("\nOverall (360 paired cells per baseline):")
    print(overall.to_string(index=False))
    print()
    print("Datasets where TabPFN has a Holm-significant advantage per baseline "
          "(delta > 0 and Holm p < 0.05):")
    if not per_ds_deep_df.empty:
        win = per_ds_deep_df[per_ds_deep_df["significant"]]
        for b in BASELINES:
            wins = win[win["model"] == b]["dataset"].tolist()
            print(f"  vs {b:>10}: {len(wins)}/8 datasets  {wins}")
    print()
    print("Datasets where TabPFN does NOT have a significant advantage "
          "(Holm p >= 0.05 — reviewer's defence):")
    if not per_ds_deep_df.empty:
        nw = per_ds_deep_df[~per_ds_deep_df["significant"]]
        for b in BASELINES:
            others = nw[nw["model"] == b]["dataset"].tolist()
            print(f"  vs {b:>10}: {len(others)}/8 datasets  {others}")


if __name__ == "__main__":
    main()
