"""
Multi-seed few-shot runner with idempotent cache reuse.

What this does:
    - Reuses existing seed=0 rows from experiments/results/few_shot/
      extended_few_shot_results.csv (no recompute).
    - Adds seeds 1, 2, 3, 4 for every (model, dataset, N) cell.
    - Writes a NEW file:
          experiments/results/few_shot/multiseed_few_shot_results.csv
      with one row per (model, dataset, N, seed). Existing CSVs are
      untouched.
    - Prints paired Wilcoxon and Cliff's δ at the end via
      src.evaluation.paired_stats.

Run:
    python scripts/run_multiseed_few_shot.py                # all 9 datasets, seeds 1-4
    python scripts/run_multiseed_few_shot.py --seeds 1 2 3 4
    python scripts/run_multiseed_few_shot.py --quick        # seeds 1 only
    python scripts/run_multiseed_few_shot.py --datasets diabetes titanic
    python scripts/run_multiseed_few_shot.py --models TabPFN XGBoost
    python scripts/run_multiseed_few_shot.py --train_sizes 25 50 100

Dataset list (locked 2026-08-06, see paper1/06_datasets_and_split/datasets.md
for the selection audit):
    titanic, banknote-auth, sick, kc1, ozone-level,
    spambase, phoneme, australian
Final cell count: 8 datasets x 9 train sizes x 5 models x 5 seeds = 1800.

Note: diabetes (n_train=614) was rejected from the Act-2 list because
n_train < 1000 -> N=750 and N=1000 clamp to actual_n_train=614, which
is identical to N='full' and so the upper end of the N-sweep would be
uninformative on that dataset. See datasets.md §2.3 for the full
exclusion list.

Design notes:
    - Subsampling uses a different random_state per seed, but the test split
      is fixed across seeds (same DataLoader) so that pairing is valid.
    - Each (model, dataset, N, seed) cell is cached as a row in the new CSV
      immediately after compute, so a Ctrl-C during a long run loses no work.
    - If the script is re-invoked it skips cells already present in the CSV.
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper, TabPFN35Wrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.models.mlp_model import MLPWrapper
from tabular_gauge.evaluation.metrics import compute_metrics
from tabular_gauge.evaluation.paired_stats import (
    per_n_paired_pvalues,
    per_dataset_paired_pvalues,
    paired_bootstrap_diff,
)


# Act-2 (few-shot scaling) dataset list — 8 datasets, locked 2026-08-06.
#
# Selection criteria (see paper1/06_datasets_and_split/datasets.md for full audit):
#   M1: n_train > 1000 (so N=1000 is not capped to 'full'; we need real signal
#       in the crossover region).
#   M2: n_test >= 100 (so per-cell AUC is not noisy at the smallest N).
#   M3: 0.05 <= pos_rate <= 0.95 (so majority-class AUC is not the floor).
#   M4: per-domain uniqueness (no two software-defect datasets, etc.).
#
# IMPORTANT — OpenML IDs (verified live 2026-08-06):
#   The IDs 3917 / 3918 / 3919 that previously pointed at kc1 / kc2 / pc1 have
#   been silently re-used by OpenML for unrelated QSAR-TID datasets (multi-class
#   regression, p=1024). We use the correct IDs:
#       kc1  -> 1067 (2109 rows, 21 features, binary, software-defect)
#       kc2  -> 1063 (522 rows, 21 features, binary, software-defect)
#       pc1  -> 1068 (1109 rows, 21 features, binary, software-defect)
#   kc2 / pc1 fail M1, so they are excluded; kc1 (1067) replaces the previous
#   broken kc1 entry and restores the software-defect domain.
#
#   diabetes (OpenML id 37) fails M1 (n_train=614). It was considered as a
#   small-N anchor under the previous protocol but was rejected because
#   N=750 and N=1000 would clamp to actual_n_train=614 and produce cells
#   identical to N='full', making the upper end of the N-sweep
#   uninformative. Removing diabetes makes the 8 surviving datasets
#   uniformly cover the full 9-N grid {25, 50, 100, 200, 300, 500, 750,
#   1000, 'full'}.
#
# Diversity coverage on the 8-dataset set:
#   - 8 unique domains
#   - 0 small + 4 medium + 4 large (smallest is titanic at n_train=1047)
#   - 4 mild + 4 severe imbalance
#   - 4 numeric + 4 mixed feature types
#
# Final cell count: 8 datasets x 9 train sizes x 5 models x 5 seeds = 1800.
DATASETS = [
    {"id": 40945, "name": "titanic",        "domain": "historical",       "n": 1309,  "p": 13},  # medium
    {"id": 1462,  "name": "banknote-auth",  "domain": "security",         "n": 1372,  "p": 4},   # medium
    {"id": 1067,  "name": "kc1",            "domain": "software-defect",  "n": 2109,  "p": 21},  # medium, severe imbalance (CORRECTED OpenML id)
    {"id": 1487,  "name": "ozone-level",    "domain": "environment",      "n": 2534,  "p": 72},  # medium, severe imbalance
    {"id": 38,    "name": "sick",           "domain": "healthcare",       "n": 3772,  "p": 29},  # large, severe imbalance
    {"id": 44,    "name": "spambase",       "domain": "NLP",              "n": 4601,  "p": 57},  # large
    {"id": 1489,  "name": "phoneme",        "domain": "speech",           "n": 5404,  "p": 5},   # large
    {"id": 40536, "name": "australian",     "domain": "finance",          "n": 8378,  "p": 120}, # large, severe imbalance (CORRECTED bal -> severe)
]

# Legacy 6-dataset list (used by scripts/run_extended_few_shot.py).
# Kept here for reference and for `--datasets` filter compatibility; not the
# default. The n values were also slightly off (australian was listed as
# n=690 — actual is 8378; titanic was listed as n=891 — actual is 1309);
# the OpenML id is what actually matters, the n/p fields are documentation.
#
# WARNING: legacy kc1 (3917), kc2 (3918), pc1 (3919) IDs are NO LONGER binary
# software-defect datasets on OpenML (they have been re-used for QSAR-TID
# multi-class regression datasets with p=1024). Loading them now will either
# fail (kc2) or silently run on the wrong data (kc1, pc1). Do not rely on
# these IDs.
DATASETS_LEGACY_6 = [
    {"id": 37,   "name": "diabetes",       "domain": "healthcare",  "n": 768,  "p": 8},
    {"id": 1510, "name": "wdbc",           "domain": "healthcare",  "n": 569,  "p": 30},
    {"id": 40536,"name": "australian",     "domain": "finance",     "n": 8378, "p": 120},
    {"id": 31,   "name": "credit-g",       "domain": "finance",     "n": 1000, "p": 20},
    {"id": 40945,"name": "titanic",        "domain": "historical",  "n": 1309, "p": 13},
    {"id": 1462, "name": "banknote-auth",  "domain": "security",    "n": 1372, "p": 4},
]

# Same 8 N values, matching extended_few_shot_results.csv
TRAIN_SIZES = ["full"]

MODEL_FACTORIES = {
    "TabPFN":   lambda seed: TabPFNWrapper(random_state=seed),
    "TabPFN_3.5": lambda seed: TabPFN35Wrapper(random_state=seed),
    "XGBoost":  lambda seed: XGBoostWrapper(n_trials=10, random_state=seed),
    "LightGBM": lambda seed: LightGBMWrapper(n_trials=10, random_state=seed),
    "CatBoost": lambda seed: CatBoostWrapper(n_trials=10, random_state=seed),
    "MLP":      lambda seed: MLPWrapper(n_trials=8,  random_state=seed),
}

OUTPUT_CSV = project_root / "experiments" / "results" / "tabpfn_35_benchmark" / "multiseed_few_shot_results.csv"
SEED0_SOURCE_CSV = project_root / "experiments" / "results" / "few_shot" / "extended_few_shot_results.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def stratified_subsample(X, y, n_samples, seed):
    """Stratified subsample with a per-seed RNG. Same logic as the legacy
    scripts but no longer hard-codes the random state."""
    rng = np.random.default_rng(seed * 1000 + 31)
    if n_samples is None or n_samples >= len(y):
        return X, np.asarray(y)

    if hasattr(X, "values"):
        X_array = X.values
    else:
        X_array = X
    y_array = np.asarray(y)

    pos_idx = np.where(y_array == 1)[0]
    neg_idx = np.where(y_array == 0)[0]

    pos_ratio = len(pos_idx) / len(y_array)
    n_pos = min(len(pos_idx), int(round(n_samples * pos_ratio)))
    n_neg = n_samples - n_pos
    if n_neg > len(neg_idx):
        n_neg = len(neg_idx)
        n_pos = n_samples - n_neg

    sel_pos = rng.choice(pos_idx, n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(neg_idx, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    sel = np.sort(np.concatenate([sel_pos, sel_neg]))

    if hasattr(X, "iloc"):
        return X.iloc[sel], y_array[sel]
    return X_array[sel], y_array[sel]


def normalise_size_label(n):
    """Return canonical N label used in the source CSVs."""
    if n is None or n == "full":
        return "full"
    return str(int(n))


def load_seed0_pool() -> pd.DataFrame:
    """
    Read the existing extended_few_shot_results.csv and pull out all seed=0
    rows. We use it to pre-populate the new multi-seed CSV so we never
    recompute seed=0.

    We map N values in the source CSV (which uses train_size = 25..1000 with
    'full' as a string) onto our TRAIN_SIZES list, where 'full' means "use
    the entire train split".
    """
    if not SEED0_SOURCE_CSV.exists():
        return pd.DataFrame()
    raw = pd.read_csv(SEED0_SOURCE_CSV)

    # The CSV has both train_size and actual_n_train. We re-derive a clean
    # integer N where possible.
    def _norm(value):
        if isinstance(value, str):
            if value.lower() == "full":
                return "full"
            try:
                return str(int(value))
            except ValueError:
                return value
        if isinstance(value, (int, np.integer)) and not pd.isna(value):
            return str(int(value))
        return "full"

    raw["train_size"] = raw["train_size"].apply(_norm)

    # The dataset column in the source CSV is suffixed with _N<value>.
    # We strip the suffix to get a clean dataset name.
    raw["dataset"] = raw["dataset"].str.replace(r"_N(full|\d+)$", "", regex=True)
    return raw


def cell_already_computed(df: pd.DataFrame, model: str, dataset: str, n_label: str, seed: int) -> bool:
    if df.empty:
        return False
    mask = (
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["train_size"] == n_label)
        & (df["seed"] == seed)
    )
    return bool(mask.any())


# --------------------------------------------------------------------------- #
# main runner
# --------------------------------------------------------------------------- #

def run(args):
    print("=" * 70)
    print("MULTI-SEED FEW-SHOT RUNNER")
    print("=" * 70)

    # 1. Pre-populate from seed-0 pool
    pool = load_seed0_pool()
    if not pool.empty:
        print(f"\nReusing {len(pool)} existing seed=0 rows from {SEED0_SOURCE_CSV.name}")
        print(f"  (Note: seed=0 rows exist for the legacy 6 datasets only —")
        print(f"   kc1(1067), sick, ozone, spambase, phoneme, australian will all")
        print(f"   need fresh seed=0 cells.)")

    if OUTPUT_CSV.exists() and not args.overwrite:
        existing = pd.read_csv(OUTPUT_CSV)
        print(f"Found existing {OUTPUT_CSV.name} with {len(existing)} rows — will skip already-computed cells")
        # Warn if existing has rows for stale OpenML ids
        stale_ids = {'kc2', 'pc1'}
        stale_rows = existing[existing['dataset'].isin(stale_ids)]
        if len(stale_rows):
            print(f"  [warn] {len(stale_rows)} rows in existing CSV use stale OpenML IDs")
            print(f"         ({stale_rows['dataset'].unique().tolist()}). They will NOT")
            print(f"         be recomputed and should be removed before analysis:")
            print(f"           df = df[~df['dataset'].isin(['kc2','pc1'])]")
    else:
        # Fresh build: pool may be empty (e.g. --overwrite after the seed=0
        # pool file was deleted). We seed `existing` with a header-only CSV
        # so the per-row flush (mode='a', header=False) below writes a
        # header-less body, and the final pd.read_csv re-reads the right
        # columns.
        existing = pool.copy()
        # Always delete any stale output before a fresh start so the
        # header row is written exactly once.
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
        if not existing.empty:
            existing.to_csv(OUTPUT_CSV, index=False)

    # 2. Filter to the requested subsets.
    # If the user names a dataset that is in DATASETS_LEGACY_6 but not in
    # the current DATASETS list, we still honour it (so legacy CSV rows can
    # be queried / extended). Datasets not in either list are warned about.
    available = {d["name"]: d for d in DATASETS}
    legacy    = {d["name"]: d for d in DATASETS_LEGACY_6}
    resolved  = []
    missing   = []
    for name in args.datasets:
        if name in available:
            resolved.append(available[name])
        elif name in legacy:
            resolved.append(legacy[name])
        else:
            missing.append(name)
    if missing:
        print(f"  [warn] Unknown dataset(s) (skipped): {missing}")
    datasets = resolved
    models = [m for m in MODEL_FACTORIES if m in args.models]
    sizes = [s for s in TRAIN_SIZES if normalise_size_label(s) in {normalise_size_label(x) for x in args.train_sizes}]

    print(f"\nDatasets     ({len(datasets)}): {[d['name'] for d in datasets]}")
    print(f"Models       ({len(models)}): {models}")
    print(f"Train sizes  ({len(sizes)}): {sizes}")
    print(f"Seeds        ({len(args.seeds)}): {args.seeds}")
    print(f"Total cells to consider: {len(datasets) * len(models) * len(sizes) * len(args.seeds)}")

    # 3. Load each dataset once; cache it in memory
    loaded = {}
    for d in datasets:
        loader = DataLoader(test_size=0.2, random_state=42)
        loaded[d["name"]] = loader.load_openml(d["id"], d["name"])
        print(f"  Loaded {d['name']}: N={loaded[d['name']].n_rows}, p={loaded[d['name']].n_features}")

    # 4. Iterate (model, dataset, size, seed), skipping cells already done
    rows_buffer = []
    n_done, n_skipped, n_failed = 0, 0, 0
    t_start = time.perf_counter()
    # Track whether the output CSV already has a header. After the
    # `if OUTPUT_CSV.exists() and not args.overwrite:` branch above, the
    # CSV may exist (with or without header). We probe by reading 1 row.
    csv_has_header = False
    if OUTPUT_CSV.exists():
        try:
            pd.read_csv(OUTPUT_CSV, nrows=0)
            csv_has_header = True
        except Exception:
            csv_has_header = False

    def _flush():
        nonlocal rows_buffer, csv_has_header, existing
        if not rows_buffer:
            return
        # First flush into an existing file appends without header; first
        # flush into a missing / header-less file must include the header
        # so pd.read_csv later reads the columns correctly.
        write_header = not csv_has_header
        pd.DataFrame(rows_buffer).to_csv(
            OUTPUT_CSV, mode="a", header=write_header, index=False
        )
        csv_has_header = True
        existing = pd.read_csv(OUTPUT_CSV)
        rows_buffer = []

    for d in datasets:
        ds = loaded[d["name"]]
        for n in sizes:
            n_label = normalise_size_label(n)
            for model_name in models:
                for seed in args.seeds:
                    if cell_already_computed(existing, model_name, d["name"], n_label, seed):
                        n_skipped += 1
                        continue

                    try:
                        if n == "full":
                            Xtr, ytr = ds.X_train, ds.y_train
                            actual_n = len(ytr)
                        else:
                            Xtr, ytr = stratified_subsample(ds.X_train, ds.y_train, int(n), seed)
                            actual_n = len(ytr)

                        model = MODEL_FACTORIES[model_name](seed)
                        result = model.run(
                            Xtr, ytr, ds.X_test, ds.y_test,
                            # dataset_name carries ONLY the dataset so that
                            # rows from different seeds (and the seed=0 pool)
                            # can be paired by (model, dataset, train_size).
                            # The N lives in the train_size column below.
                            dataset_name=d["name"],
                            seed=seed,
                        )
                        metrics = compute_metrics(result)
                        metrics["train_size"] = n_label
                        metrics["actual_n_train"] = actual_n
                        metrics["domain"] = d["domain"]
                        rows_buffer.append(metrics)
                        n_done += 1

                        elapsed = time.perf_counter() - t_start
                        print(
                            f"  [done {n_done:>4}] {d['name']:>14s} N={n_label:>5s} "
                            f"{model_name:>9s} seed={seed} "
                            f"AUC={metrics.get('roc_auc', 0):.4f} "
                            f"({elapsed:6.1f}s total)"
                        )

                        # Flush every row so a long run survives a Ctrl-C
                        if len(rows_buffer) >= 1:
                            _flush()

                    except Exception as e:
                        n_failed += 1
                        print(f"  [FAIL] {d['name']} N={n_label} {model_name} seed={seed}: {e}")

    # 5. Flush remaining rows
    _flush()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Cells computed this run : {n_done}")
    print(f"Cells skipped (cached)  : {n_skipped}")
    print(f"Cells failed            : {n_failed}")
    print(f"Total rows in {OUTPUT_CSV.name}: {len(existing)}")

    # 6. If we have multi-seed data, print the paired stats now
    have_pairs = (
        not existing.empty
        and "seed" in existing.columns
        and existing["seed"].nunique() >= 2
    )
    if have_pairs:
        print("\n" + "=" * 70)
        print("PAIRED WILCOXON (TabPFN vs every other, paired by seed)")
        print("=" * 70)
        per_n = per_n_paired_pvalues(existing, metric="roc_auc")
        if not per_n.empty:
            cols = [
                "train_size", "model", "n_pairs", "diff_mean",
                "wilcoxon_p", "holm_p_adj", "cliffs_delta", "significant",
            ]
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(per_n[cols].to_string(index=False))

        print("\n" + "=" * 70)
        print("PAIRED WILCOXON (TabPFN vs every other, per dataset, paired by seed)")
        print("=" * 70)
        per_d = per_dataset_paired_pvalues(existing, metric="roc_auc")
        if not per_d.empty:
            cols = [
                "dataset", "model", "n_pairs", "diff_mean",
                "wilcoxon_p", "holm_p_adj", "cliffs_delta", "significant",
            ]
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(per_d[cols].to_string(index=False))
    else:
        print("\nNot enough seeds for paired analysis yet (need >=2).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Seeds to add on top of seed=0 (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only seed=1 (smoke test, ~5 minutes).",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[d["name"] for d in DATASETS],
        help="Subset of datasets.",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_FACTORIES.keys()),
        help="Subset of models.",
    )
    parser.add_argument(
        "--train_sizes", nargs="+",
        default=[normalise_size_label(s) for s in TRAIN_SIZES],
        help="Subset of training sizes (e.g. 25 50 100 full).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete the multi-seed CSV and start fresh (still reuses seed=0).",
    )
    args = parser.parse_args()
    if args.quick:
        args.seeds = [1]
    args.train_sizes = [normalise_size_label(s) for s in args.train_sizes]
    run(args)


if __name__ == "__main__":
    main()