"""
TabPFN-v2 in-context-learning ablation: fit with PERMUTED training
labels (keeping X_train intact). If TabPFN-v2's AUC collapses to
near-chance, that is direct evidence that the in-context learning
over the (X, y) pairs is what's driving its low-N advantage — not
just the architecture or the pre-training priors in isolation.

Standard reference: Min et al. 2022 "Rethinking Practice for
In-Context Learning" — label-shuffling is the standard ablation for
in-context learners (also used by TabPFN-v2's own paper §3).

Scope (fast):
    3 datasets x 5 N values x 3 seeds x {TabPFN-v2 baseline,
    TabPFN-v2 shuffled} = 90 fits, ~20 min.

Datasets picked to span the Stage-II heterogeneity:
    banknote-auth  (easy, n_train=1372, pos_rate=0.445)
    kc1            (hard, n_train=2109, pos_rate=0.155, highly imbalanced)
    spambase       (medium, n_train=4601, pos_rate=0.394, high p)

N values: {25, 50, 100, 500, "full"} — spans the regime where we
claim TabPFN-v2 dominates.

Outputs
-------
experiments/results/few_shot/tabpfn_shuffled_labels.csv
    Long table: 1 row per (dataset, seed, train_size, mode) where
    mode is "clean" (regular TabPFN-v2 fit) or "shuffled" (TabPFN-v2
    fit with permuted y_train).

reports/tabpfn_shuffled_labels.json
    Per-cell delta = AUC(clean) - AUC(shuffled), plus the per-N
    mean delta and a paired Wilcoxon p-value on (clean, shuffled)
    pairs across (dataset, seed, N).

reports/appendix_prior_ablation.md
    Markdown appendix summarizing the result with the four
    possible interpretations:
        1. shuffle -> 0.5 AUC, clean high
           = "in-context learning is doing the work"
        2. shuffle -> ~0.5 AUC, clean high AND small
           = "in-context learning is necessary but priors matter too"
        3. shuffle -> ~clean AUC
           = "the architecture is what's helping, not in-context learning"
        4. shuffle -> 0.5 AUC, clean low too
           = "TabPFN-v2 is just broken on our setup" (unlikely; sanity check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scipy.stats import wilcoxon  # noqa: E402

from tabular_gauge.models.tabpfn_model import TabPFNWrapper  # noqa: E402
from tabular_gauge.evaluation.metrics import compute_metrics  # noqa: E402
from tabular_gauge.data.loader import DataLoader  # noqa: E402

OUT_DIR = ROOT / "experiments" / "results" / "few_shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "tabpfn_shuffled_labels.csv"
OUT_JSON = ROOT / "reports" / "tabpfn_shuffled_labels.json"
OUT_MD = ROOT / "reports" / "appendix_prior_ablation.md"

DATASETS = [
    ("banknote-auth", 1462),   # easy
    ("kc1",            1067),  # hard, severe imbalance
    ("spambase",         44),  # medium, high p
]
SEEDS = [0, 1, 2]
N_VALUES = [25, 50, 100, 500, "full"]


def _mb(b: int) -> float:
    return round(b / (1024.0 * 1024.0), 2)


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    """Same convention as scripts/run_multiseed_few_shot.py."""
    rng = np.random.default_rng(seed * 1000 + 31)
    classes = np.unique(y)
    per_class = n // len(classes)
    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        take = min(per_class, len(cls_idx))
        indices.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_eval(ds_name: str, ds_id: int, seed: int, n_label,
             shuffle: bool) -> dict:
    dl = DataLoader(test_size=0.2, random_state=42)
    ds = dl.load_openml(ds_id, ds_name)
    X_full, y_full = ds.X_train, ds.y_train
    X_test, y_test = ds.X_test, ds.y_test

    if n_label == "full":
        X_tr, y_tr = X_full, y_full
        actual_n = len(y_tr)
    else:
        X_tr, y_tr = stratified_subsample(X_full.values,
                                          y_full.values if hasattr(y_full, "values") else y_full,
                                          int(n_label), seed)
        actual_n = len(y_tr)

    if shuffle:
        # Permute y_tr only — keep X intact. The permutation is
        # deterministic in seed so results are reproducible.
        rng = np.random.default_rng(seed * 7919 + 17)
        y_tr = np.asarray(y_tr).copy()
        rng.shuffle(y_tr)

    X_tr_arr = X_tr.values if hasattr(X_tr, "values") else X_tr

    clf = TabPFNWrapper(n_ensemble_configurations=8, random_state=seed)
    clf.max_context_size = 10_000

    tracemalloc.start()
    # Use the wrapper's run() helper so compute_metrics gets a
    # proper RunResult (compute_metrics takes a RunResult, not
    # raw y/y_proba arrays).
    rr = clf.run(X_tr_arr, y_tr,
                 X_test.values if hasattr(X_test, "values") else X_test,
                 y_test, ds_name, seed)
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    m = compute_metrics(rr)
    return {
        "dataset":       ds_name,
        "seed":          seed,
        "train_size":    str(n_label),
        "actual_n_train": actual_n,
        "mode":          "shuffled" if shuffle else "clean",
        "auc":           float(m["roc_auc"]),
        "ece_10":        float(m.get("ece_10", float("nan"))),
        "fit_time_sec":  round(float(m.get("fit_time_sec", 0.0)), 4),
        "predict_time_sec": round(float(m.get("predict_time_sec", 0.0)), 4),
        "py_peak_mem_mb": _mb(py_peak),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None,
                        help="If a precomputed CSV exists at this path, "
                             "skip the run and rebuild the appendix.")
    args = parser.parse_args()

    if args.csv and args.csv.exists():
        print(f"Loading precomputed CSV from {args.csv} ...")
        df = pd.read_csv(args.csv)
    else:
        rows = []
        total = len(DATASETS) * len(SEEDS) * len(N_VALUES) * 2
        i = 0
        for ds_name, ds_id in DATASETS:
            print(f"\n=== {ds_name} (openml={ds_id}) ===")
            for seed in SEEDS:
                for n_label in N_VALUES:
                    for shuffle in (False, True):
                        i += 1
                        print(f"  [{i}/{total}] {ds_name} seed={seed} N={n_label} "
                              f"mode={'shuffled' if shuffle else 'clean'} ...",
                              end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            row = fit_eval(ds_name, ds_id, seed, n_label, shuffle)
                            rows.append(row)
                            wall = time.perf_counter() - t0
                            print(f"AUC={row['auc']:.4f}  wall={wall:.1f}s")
                        except Exception as e:
                            print(f"FAILED: {e!r}")
                            rows.append({
                                "dataset": ds_name, "seed": seed,
                                "train_size": str(n_label),
                                "mode": "shuffled" if shuffle else "clean",
                                "auc": None, "error": repr(e),
                            })
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    # ----- analysis -----
    pivot = df.pivot_table(index=["dataset", "seed", "train_size"],
                           columns="mode", values="auc",
                           aggfunc="first").reset_index()
    pivot = pivot.dropna(subset=["clean", "shuffled"])
    pivot["delta"] = pivot["clean"] - pivot["shuffled"]

    # Per-N paired Wilcoxon (clean vs shuffled)
    per_n = []
    for n in N_VALUES:
        sub = pivot[pivot["train_size"] == str(n)]
        if len(sub) < 3:
            continue
        try:
            w, p = wilcoxon(sub["clean"], sub["shuffled"], alternative="greater")
        except ValueError:
            # All differences zero -> wilcoxon raises
            w, p = float("nan"), float("nan")
        per_n.append({
            "train_size": str(n),
            "n_pairs": int(len(sub)),
            "mean_clean":   round(float(sub["clean"].mean()), 4),
            "mean_shuffled": round(float(sub["shuffled"].mean()), 4),
            "mean_delta":   round(float(sub["delta"].mean()), 4),
            "max_abs_delta": round(float(np.abs(sub["delta"]).max()), 4),
            "wilcoxon_p_greater": round(float(p), 6) if not np.isnan(p) else None,
        })

    # Overall Wilcoxon
    overall_w = None
    overall_p = None
    if len(pivot) >= 5:
        try:
            overall_w, overall_p = wilcoxon(pivot["clean"], pivot["shuffled"],
                                            alternative="greater")
        except ValueError:
            pass

    OUT_JSON.write_text(json.dumps({
        "n_cells":  int(len(df)),
        "n_paired": int(len(pivot)),
        "overall_wilcoxon_p":   (round(float(overall_p), 6)
                                  if overall_p is not None else None),
        "overall_wilcoxon_w":   (round(float(overall_w), 4)
                                  if overall_w is not None else None),
        "per_n": per_n,
        "per_dataset": (pivot.groupby("dataset")
                        .agg(mean_clean=("clean", "mean"),
                             mean_shuffled=("shuffled", "mean"),
                             mean_delta=("delta", "mean"))
                        .reset_index().to_dict(orient="records")),
    }, indent=2), encoding="utf-8")

    # ----- markdown appendix -----
    md = []
    md.append("# Appendix C.6 — TabPFN-v2 In-Context-Learning Ablation\n\n")
    md.append("**Created:** 2026-08-19 — to address the question: *is TabPFN-v2's\n"
              "low-N lead driven by in-context learning over the (X, y) context, or\n"
              "by the model architecture / pre-training priors in isolation?*\n\n"
              "We fit TabPFN-v2 twice on the **same** (X_train, dataset, seed, N):\n"
              "once with the **real** training labels, once with the training\n"
              "labels **permuted** (deterministically, keeping X intact). The\n"
              "permutation is the standard in-context-learning ablation (Min et\n"
              "al. 2022, also used in the TabPFN-v2 paper §3): if TabPFN-v2's\n"
              "lead at low N comes from in-context learning over the labelled\n"
              "context, the shuffled-label fit must collapse to near-chance AUC\n"
              "because the labels carry no information.\n\n"
              "If, instead, shuffled-labels AUC stays close to clean AUC, the\n"
              "lead is **not** explained by in-context learning — it is either\n"
              "an architectural artefact or comes from prior-data exposure\n"
              "without using the context. Both would be informative for the\n"
              "narrative in §1.\n\n")
    md.append("## C.6.1 Per-N result\n\n")
    md.append("| N | n pairs | mean AUC (clean) | mean AUC (shuffled) | Δ AUC | "
              "Wilcoxon p (clean > shuffled) |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for r in per_n:
        md.append(f"| {r['train_size']} | {r['n_pairs']} | "
                  f"{r['mean_clean']:.4f} | {r['mean_shuffled']:.4f} | "
                  f"{r['mean_delta']:+.4f} | "
                  f"{r['wilcoxon_p_greater']} |\n")
    md.append("\n")
    md.append("## C.6.2 Per-dataset result\n\n")
    md.append("| Dataset | mean clean | mean shuffled | Δ |\n|---|---:|---:|---:|\n")
    for r in (pivot.groupby("dataset")
              .agg(mean_clean=("clean", "mean"),
                   mean_shuffled=("shuffled", "mean"),
                   mean_delta=("delta", "mean"))
              .reset_index().to_dict(orient="records")):
        md.append(f"| {r['dataset']} | {r['mean_clean']:.4f} | "
                  f"{r['mean_shuffled']:.4f} | {r['mean_delta']:+.4f} |\n")
    md.append("\n")
    if overall_p is not None:
        md.append(f"**Overall paired Wilcoxon (clean > shuffled): p = {overall_p:.4g}, "
                  f"across {len(pivot)} (dataset, seed, N) pairs.**\n\n")
    md.append("## C.6.3 Interpretation\n\n"
              "* **If shuffled AUC collapses to ~0.5 across the board:** the\n"
              "  in-context-learning mechanism is doing the work; pretrained\n"
              "  priors alone (architecture + general training) are not enough.\n"
              "  This is the standard 'ICL is the source of the advantage'\n"
              "  finding for in-context learners.\n"
              "* **If shuffled AUC collapses only at N ≤ 500:** the in-context\n"
              "  mechanism matters most when the labelled context is small;\n"
              "  pretrained priors carry the rest. This is consistent with a\n"
              "  pretrained prior that 'kicks in' once the in-context signal\n"
              "  gets weak.\n"
              "* **If shuffled AUC stays close to clean AUC:** the in-context\n"
              "  mechanism is *not* the source of the lead — the lead is\n"
              "  either an architectural artefact or a prior-data exposure\n"
              "  effect that doesn't depend on the actual context labels. We\n"
              "  would then need to weaken §1's priors narrative.\n"
              "* **Sanity check:** if clean AUC is also ~0.5, the model is\n"
              "  broken on our setup and the ablation is uninformative.\n\n")
    md.append("## C.6.4 Sources\n\n"
              "* Ablation CSV: `experiments/results/few_shot/tabpfn_shuffled_labels.csv`\n"
              "* Summary JSON: `reports/tabpfn_shuffled_labels.json`\n"
              "* Reproduce: `python scripts/tabpfn_shuffled_labels_ablation.py`\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"\nOverall Wilcoxon (clean > shuffled): p = {overall_p}")
    print("Per-N:")
    for r in per_n:
        print(f"  N={r['train_size']:>5}  Δ={r['mean_delta']:+.4f}  "
              f"p={r['wilcoxon_p_greater']}")


if __name__ == "__main__":
    main()
