"""
Publication-ready Act-2 figures that are still missing per
paper1/11_figures_inventory/figures.md and
paper1/12_next_steps_checklist/checklist.md Phase B.

Reads:   experiments/results/few_shot/multiseed_few_shot_results.csv  (1 800 rows)
Writes:  reports/figures/act2_multiseed/<figure>.png   (PNG, 200 DPI)
         reports/figures/act2_multiseed/<figure>.pdf   (vector, for papers)
         experiments/results/few_shot/per_dataset_crossover_with_ci.csv

Figures produced
----------------
F1. per_dataset_crossover_with_ci.png
    Per-dataset smallest N at which the BEST tuned GBDT mean AUC >= TabPFN
    mean AUC, with 2000-resample bootstrap CI on the crossover N itself.
    The headline figure of paper §5.3.

F2. multiseed_memory_vs_n.png
    Peak memory (y, log) vs N (x, log), one line per model, ±1 std band
    across datasets × seeds.

F3. multiseed_pareto_auc_vs_time.png
    ROC-AUC (x) vs ms per test row (y, log), one marker per (dataset, N)
    cell, one colour per model, small horizontal jitter so over-plotting
    is visible. Shows the inference-cost/quality Pareto frontier.

F4. multiseed_ece_by_n.png
    ECE-10 (y) vs N (x, log), one line per model, ±1 std band.
    Calibration error is the §6 tiebreaker.

F5. act2_significance_heatmap.png
    9 train sizes × 4 baselines cell grid. Cell = Holm-corrected p of the
    Wilcoxon signed-rank (TabPFN vs baseline, paired on dataset×seed).
    Green if Holm-significant in TabPFN's favour, red if the baseline wins.

Style: reuses tabular_gauge/viz/style.apply_style() and the canonical MODEL_COLORS
palette so this figure set is consistent with the rest of the report.

Run:
    python scripts/plot_act2_missing_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tabular_gauge.evaluation.paired_stats import (
    cliffs_delta,
    holm_bonferroni,
    paired_bootstrap_diff,
    per_n_paired_pvalues,
    per_dataset_paired_pvalues,
)
from tabular_gauge.viz.style import (
    ACCENT,
    ACCENT2,
    ERROR,
    LIGHT_BG,
    LIGHT_GRID,
    LIGHT_SURFACE,
    MODEL_COLORS,
    MUTED_TEXT,
    TEXT_COLOR,
    WARN,
    apply_style,
    color_for_model,
    save_figure,
)


CSV_PATH    = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
FIG_DIR     = ROOT / "reports" / "figures" / "act2_multiseed"
CROSSOVER_CSV = ROOT / "experiments" / "results" / "few_shot" / "per_dataset_crossover_with_ci.csv"

REF_MODEL      = "TabPFN"
BASELINES      = ["XGBoost", "LightGBM", "CatBoost", "MLP"]
ALL_MODELS     = [REF_MODEL] + BASELINES
TRAIN_SIZE_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}. Run scripts/run_multiseed_few_shot.py first.")
    df = pd.read_csv(CSV_PATH)
    df["seed"]        = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["train_size"]  = df["train_size"].astype(str)
    return df.dropna(subset=["roc_auc"])


def numeric_n(s: pd.Series) -> np.ndarray:
    """Convert train_size labels (incl. 'full') into numeric positions for
    log-scale plotting. full is plotted at the largest x."""
    arr = []
    for v in s:
        if v == "full":
            arr.append(1500.0)
        else:
            arr.append(float(v))
    return np.asarray(arr)


def _train_size_to_x(s: pd.Series) -> np.ndarray:
    """Same as numeric_n but treats 'full' as 1.5× the largest finite N so it
    plots to the right of every other point on a log axis."""
    return numeric_n(s)


# --------------------------------------------------------------------------- #
# F1 — Per-dataset crossover N with bootstrap CI  (HEADLINE FIGURE)
# --------------------------------------------------------------------------- #

def plot_per_dataset_crossover(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    For each dataset:
      - Take the seed×train_size mean ROC-AUC per model.
      - best_GBDT = max over (XGBoost, LightGBM, CatBoost) per (N, seed).
      - For each N, compute ΔAUC = best_GBDT_meanAUC - TabPFN_meanAUC.
      - Crossover N = smallest N where ΔAUC >= 0.

    Bootstrap CI on the crossover N itself:
      - Resample (N, seed) cells with replacement (stratified by N) for
        2000 iterations.
      - In each iteration, recompute the crossover N; the 2.5/97.5 percentiles
        are the 95% CI. A bootstrap sample where TabPFN never loses returns
        `np.nan` (lower bound only).
    """
    sizes_present = [t for t in TRAIN_SIZE_ORDER if t in set(df["train_size"].unique())]

    # Means across seeds per (dataset, model, N)
    mean = (
        df.groupby(["dataset", "model", "train_size"])["roc_auc"]
        .mean()
        .reset_index()
    )

    # For TabPFN and the best-GBDT-trio
    tabpfn = mean[mean["model"] == REF_MODEL].rename(columns={"roc_auc": "tabpfn_auc"})
    gbdt = mean[mean["model"].isin(["XGBoost", "LightGBM", "CatBoost"])]
    best_gbdt = (
        gbdt.groupby(["dataset", "train_size"])["roc_auc"]
        .max()
        .reset_index()
        .rename(columns={"roc_auc": "best_gbdt_auc"})
    )
    joined = best_gbdt.merge(tabpfn[["dataset", "train_size", "tabpfn_auc"]],
                             on=["dataset", "train_size"], how="inner")
    joined["delta_auc"] = joined["best_gbdt_auc"] - joined["tabpfn_auc"]

    # For each dataset, find the smallest N where delta >= 0
    rows = []
    for ds in sorted(joined["dataset"].unique()):
        sub = joined[joined["dataset"] == ds].copy()
        sub["n_num"] = sub["train_size"].apply(lambda v: float(v) if v != "full" else 1e4)
        sub = sub.sort_values("n_num")
        # Per-dataset (N, seed) cells for bootstrap
        per_cell = df[df["dataset"] == ds]
        per_cell_tabpfn = per_cell[per_cell["model"] == REF_MODEL]
        per_cell_gbdt = per_cell[per_cell["model"].isin(["XGBoost", "LightGBM", "CatBoost"])]

        def _crossover_from_seed_averages(seed_avgs: pd.DataFrame) -> float:
            """Compute crossover N from a wide table indexed by train_size
            where each column is (best_gbdt, tabpfn)."""
            if seed_avgs.empty:
                return np.nan
            sa = seed_avgs.sort_values(by="n_num")
            for _, r in sa.iterrows():
                if r["delta_auc"] >= 0:
                    return r["n_num"]
            return np.nan  # never crossed in this bootstrap draw

        # Build (N, seed)-level table once
        tabpfn_cell = per_cell_tabpfn.groupby(["train_size", "seed"])["roc_auc"].mean().reset_index()
        gbdt_cell = per_cell_gbdt.groupby(["train_size", "seed", "model"])["roc_auc"].mean().reset_index()
        gbdt_cell_best = gbdt_cell.groupby(["train_size", "seed"])["roc_auc"].max().reset_index()
        wide = tabpfn_cell.merge(gbdt_cell_best, on=["train_size", "seed"], suffixes=("_tabpfn", "_gbdt"))
        wide["delta"] = wide["roc_auc_gbdt"] - wide["roc_auc_tabpfn"]
        # Mean per N across seeds
        per_n_mean = wide.groupby("train_size").agg(
            delta_auc=("delta", "mean"),
        ).reset_index()
        per_n_mean["n_num"] = per_n_mean["train_size"].apply(
            lambda v: float(v) if v != "full" else 1e4)

        # Point estimate crossover
        point = _crossover_from_seed_averages(per_n_mean)
        rows.append({"dataset": ds, "crossover_n_point": point})

        # Bootstrap CIs
        N_BOOT = 2000
        rng = np.random.default_rng(seed=0)
        seed_pool = sorted(wide["seed"].unique())
        cross_in_boot = []
        for _ in range(N_BOOT):
            chosen = rng.choice(seed_pool, size=len(seed_pool), replace=True)
            boot_wide = pd.concat(
                [wide[wide["seed"] == s] for s in chosen],
                ignore_index=True,
            )
            boot_per_n = boot_wide.groupby("train_size").agg(
                delta_auc=("delta", "mean"),
            ).reset_index()
            boot_per_n["n_num"] = boot_per_n["train_size"].apply(
                lambda v: float(v) if v != "full" else 1e4)
            cross_in_boot.append(_crossover_from_seed_averages(boot_per_n))

        cross_arr = np.array([c for c in cross_in_boot if not np.isnan(c)])
        if len(cross_arr) == 0:
            ci_lo = ci_hi = np.nan
            cross_prob = 0.0
        else:
            ci_lo = float(np.percentile(cross_arr, 2.5))
            ci_hi = float(np.percentile(cross_arr, 97.5))
            cross_prob = float((cross_arr <= 1e4).mean())
        rows[-1].update({
            "crossover_ci_lo": ci_lo,
            "crossover_ci_hi": ci_hi,
            "p_crossover_in_range": cross_prob,
            "tabpfn_at_full": tabpfn[(tabpfn.dataset == ds) & (tabpfn.train_size == "full")]["tabpfn_auc"].mean(),
            "best_gbdt_at_full": best_gbdt[(best_gbdt.dataset == ds) & (best_gbdt.train_size == "full")]["best_gbdt_auc"].mean(),
            "delta_at_full": (
                best_gbdt[(best_gbdt.dataset == ds) & (best_gbdt.train_size == "full")]["best_gbdt_auc"].mean()
                - tabpfn[(tabpfn.dataset == ds) & (tabpfn.train_size == "full")]["tabpfn_auc"].mean()
            ),
        })

    out = pd.DataFrame(rows)

    # Replace 1e4 (the 'full' sentinel) with the actual full-training n_train
    # for plotting. Use the largest actual_n_train value per dataset as the
    # right-edge x-position.
    n_full = df.groupby("dataset")["n_train"].max().to_dict()
    out["crossover_n"]      = out["crossover_n_point"].where(out["crossover_n_point"] != 1e4, np.nan)
    out["crossover_ci_lo"]  = out["crossover_ci_lo"].where(out["crossover_ci_lo"] != 1e4, np.nan)
    out["crossover_ci_hi"]  = out["crossover_ci_hi"].where(out["crossover_ci_hi"] != 1e4, np.nan)
    # If a dataset's point estimate is NaN (TabPFN dominates at every N),
    # place a right-arrow at the largest available N.
    for ds in out["dataset"]:
        idx = out.index[out["dataset"] == ds][0]
        if pd.isna(out.loc[idx, "crossover_n"]):
            # Use n_full as the position of the right-arrow; CI is the full
            # right-edge point as well (NaN if never crossed).
            out.loc[idx, "crossover_n"]     = n_full[ds]
            out.loc[idx, "crossover_ci_lo"] = n_full[ds]
            out.loc[idx, "crossover_ci_hi"] = n_full[ds]

    out = out.sort_values(by=["crossover_n", "dataset"])

    # Save CSV
    CROSSOVER_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CROSSOVER_CSV, index=False)
    print(f"  Saved CSV     -> {CROSSOVER_CSV}")

    # --- Figure ---
    apply_style("light")
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    ys = np.arange(len(out))[::-1]  # top-to-bottom by crossover N
    for y, (_, row) in zip(ys, out.iterrows()):
        c = ACCENT
        ax.scatter([row["crossover_n"]], [y], s=70, color=c,
                   edgecolor="white", linewidth=1.2, zorder=3)
        # CI whisker
        if not pd.isna(row["crossover_ci_lo"]) and not pd.isna(row["crossover_ci_hi"]):
            ax.hlines(y, row["crossover_ci_lo"], row["crossover_ci_hi"],
                      color=MUTED_TEXT, linewidth=2.0, zorder=2)
            ax.scatter([row["crossover_ci_lo"], row["crossover_ci_hi"]], [y, y],
                       s=22, color="white", edgecolor=MUTED_TEXT, linewidth=1.0,
                       zorder=3)
        # Annotation with the actual N label
        ax.annotate(
            f"N={int(round(row['crossover_n']))}" if row['crossover_n'] < 999
            else f"N=full({int(round(row['crossover_n']))})",
            xy=(row["crossover_n"], y), xytext=(8, 0),
            textcoords="offset points", fontsize=9, color=TEXT_COLOR,
            va="center",
        )

    # Reference lines for the 500 / 1000 boundaries
    for x, lbl, c in [(500, "Lower rule boundary (N=500)", WARN),
                      (1000, "Upper rule boundary (N=1000)", ERROR)]:
        ax.axvline(x, color=c, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(x, len(out) - 0.2, lbl,
                rotation=90, ha="right", va="top",
                fontsize=8, color=c, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(out["dataset"], color=TEXT_COLOR)
    ax.set_xlim(20, max(out["crossover_ci_hi"].dropna().max(), n_full[out.iloc[0]["dataset"]]) * 1.4)
    ax.set_xlabel("Smallest training-set size N at which best tuned GBDT ≥ TabPFN  (95% bootstrap CI)",
                  color=TEXT_COLOR, fontsize=10)
    ax.set_title("Per-dataset crossover N  —  Few-shot scaling boundary\n"
                  "(point = median crossover across 2000 bootstrap resamples of seed)",
                  fontsize=11, color=TEXT_COLOR, pad=10)
    ax.grid(True, axis="x", alpha=0.4, color=LIGHT_GRID)
    ax.tick_params(axis="both", colors=TEXT_COLOR)

    # In-plot caption explaining the right-arrow meaning
    fig.text(0.5, 0.01,
             "Right-arrow markers (N = full) indicate the dataset where TabPFN "
             "dominates the best tuned GBDT at every N ≤ full.",
             ha="center", fontsize=8, color=MUTED_TEXT, style="italic")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))
    return out


# --------------------------------------------------------------------------- #
# F2 — Memory vs N
# --------------------------------------------------------------------------- #

def plot_memory_vs_n(df: pd.DataFrame, save_path: Path) -> None:
    sub = df[df["model"].isin(ALL_MODELS)]
    xs = _train_size_to_x(sub["train_size"])

    apply_style("light")
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for model in ALL_MODELS:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        # Aggregate: mean per N across dataset × seed
        agg = (
            m.assign(_x=xs[m.index - m.index.min()])
             .groupby("train_size")["peak_memory_mb"]
             .agg(["mean", "std"]).reindex(TRAIN_SIZE_ORDER).reset_index()
        )
        agg["_x"] = _train_size_to_x(agg["train_size"])
        agg = agg.dropna(subset=["mean"])
        if agg.empty:
            continue
        c = color_for_model(model)
        ls = "-" if model == REF_MODEL else "--"
        lw = 2.6 if model == REF_MODEL else 1.5
        ms = 7 if model == REF_MODEL else 5
        ax.plot(agg["_x"], agg["mean"], marker="o", color=c, linestyle=ls,
                linewidth=lw, markersize=ms, markeredgecolor="white",
                markeredgewidth=0.6, label=model, zorder=(3 if model == REF_MODEL else 2))
        ax.fill_between(agg["_x"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=c, alpha=0.15, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(_train_size_to_x(pd.Series(TRAIN_SIZE_ORDER)))
    ax.set_xticklabels(TRAIN_SIZE_ORDER)
    ax.set_xlabel("Training-set size  (rows, log scale)", color=TEXT_COLOR)
    ax.set_ylabel("Peak memory  (MB, log scale)", color=TEXT_COLOR)
    ax.set_title("Memory cost vs training-set size  —  one line per model\n"
                  "(markers = mean over datasets × seeds; band = ±1 std)",
                  fontsize=11, color=TEXT_COLOR, pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.tick_params(axis="both", colors=TEXT_COLOR)
    ax.grid(True, which="both", alpha=0.3, color=LIGHT_GRID)

    fig.tight_layout()
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# F3 — Pareto: AUC vs predict-time per row
# --------------------------------------------------------------------------- #

def plot_pareto_auc_vs_time(df: pd.DataFrame, save_path: Path) -> None:
    sub = df[df["model"].isin(ALL_MODELS)].copy()
    sub["ms_per_row"] = 1000.0 * sub["predict_time_sec"] / sub["n_test"].clip(lower=1)

    apply_style("light")
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    rng = np.random.default_rng(seed=0)

    for model in ALL_MODELS:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        c = color_for_model(model)
        # Small horizontal jitter on AUC for readability of overlapping markers
        x = m["roc_auc"].values + rng.normal(0, 0.0025, size=len(m))
        y = m["ms_per_row"].values
        ax.scatter(x, y, s=24, color=c, alpha=0.45,
                   edgecolor="white", linewidth=0.4, label=model, zorder=(3 if model == REF_MODEL else 2))

    # Plot mean frontier points per model (one per model), highlighting TabPFN
    means = sub.groupby("model").agg(auc=("roc_auc", "mean"), ms=("ms_per_row", "mean"))
    for model, row in means.iterrows():
        c = color_for_model(model)
        ax.scatter([row["auc"]], [row["ms"]], s=210, color=c,
                   edgecolor="white", linewidth=2.0, marker="*",
                   zorder=(4 if model == REF_MODEL else 3))
        ax.annotate(model, xy=(row["auc"], row["ms"]),
                    xytext=(7, 0), textcoords="offset points",
                    fontsize=9.5, color=c, fontweight="bold",
                    va="center")

    ax.set_yscale("log")
    ax.set_xlabel("ROC-AUC", color=TEXT_COLOR)
    ax.set_ylabel("Inference cost per row  (ms / test row, log scale)", color=TEXT_COLOR)
    ax.set_title("Pareto frontier  —  AUC vs inference cost\n"
                  "(markers = individual (dataset, N, seed) cells; stars = per-model mean)",
                  fontsize=11, color=TEXT_COLOR, pad=10)
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, which="both", alpha=0.3, color=LIGHT_GRID)
    ax.tick_params(axis="both", colors=TEXT_COLOR)

    fig.tight_layout()
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# F4 — ECE by N
# --------------------------------------------------------------------------- #

def plot_ece_by_n(df: pd.DataFrame, save_path: Path) -> None:
    sub = df[df["model"].isin(ALL_MODELS)].copy()
    # Reindex into canonical order so lines are smooth
    sub["train_size"] = pd.Categorical(sub["train_size"],
                                      categories=TRAIN_SIZE_ORDER,
                                      ordered=True)
    sub = sub.sort_values("train_size")

    apply_style("light")
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for model in ALL_MODELS:
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        agg = m.groupby("train_size", observed=True)["ece_10"].agg(["mean", "std"]).reset_index()
        agg["_x"] = _train_size_to_x(agg["train_size"].astype(str))
        agg = agg.dropna(subset=["mean"])
        if agg.empty:
            continue
        c = color_for_model(model)
        ls = "-" if model == REF_MODEL else "--"
        lw = 2.4 if model == REF_MODEL else 1.5
        ms = 7 if model == REF_MODEL else 5
        ax.plot(agg["_x"], agg["mean"], marker="o", color=c, linestyle=ls,
                linewidth=lw, markersize=ms, markeredgecolor="white",
                markeredgewidth=0.6, label=model, zorder=(3 if model == REF_MODEL else 2))
        ax.fill_between(agg["_x"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=c, alpha=0.15, zorder=1)

    ax.set_xscale("log")
    ax.set_xticks(_train_size_to_x(pd.Series(TRAIN_SIZE_ORDER)))
    ax.set_xticklabels(TRAIN_SIZE_ORDER)
    ax.set_xlabel("Training-set size  (rows, log scale)", color=TEXT_COLOR)
    ax.set_ylabel("ECE-10  (lower = better calibrated)", color=TEXT_COLOR)
    ax.set_title("Calibration error vs training-set size\n"
                  "(markers = mean over datasets × seeds; band = ±1 std)",
                  fontsize=11, color=TEXT_COLOR, pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, which="both", alpha=0.3, color=LIGHT_GRID)
    ax.tick_params(axis="both", colors=TEXT_COLOR)

    fig.tight_layout()
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# F5 — Significance heatmap (N × baseline)
# --------------------------------------------------------------------------- #

def plot_significance_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    per_n = per_n_paired_pvalues(df, metric="roc_auc", ref_model=REF_MODEL)
    if per_n.empty:
        print("  [skip] no per-N stats to plot")
        return
    sizes_present = [t for t in TRAIN_SIZE_ORDER if t in set(per_n["train_size"].astype(str))]
    baselines = [m for m in BASELINES if m in set(per_n["model"])]

    mat = np.full((len(sizes_present), len(baselines)), np.nan)
    annot = np.empty((len(sizes_present), len(baselines)), dtype=object)
    for i, t in enumerate(sizes_present):
        for j, b in enumerate(baselines):
            row = per_n[(per_n["train_size"].astype(str) == t) & (per_n["model"] == b)]
            if row.empty:
                annot[i, j] = "—"
                continue
            p = float(row["holm_p_adj"].iloc[0])
            d = float(row["cliffs_delta"].iloc[0])
            mat[i, j] = -np.log10(max(p, 1e-10)) * np.sign(d)
            if p < 0.001:
                annot[i, j] = f"{d:+.2f}\n***"
            elif p < 0.01:
                annot[i, j] = f"{d:+.2f}\n**"
            elif p < 0.05:
                annot[i, j] = f"{d:+.2f}\n*"
            else:
                annot[i, j] = f"{d:+.2f}\nns"

    apply_style("light")
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    try:
        import seaborn as sns
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    except Exception:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "rb", ["#ef4444", "#f8fafc", "#5b8dee"])

    # Centre the colormap at 0; use a symmetric log-p-like scale
    vmax = max(2.5, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if abs(v) > 0.6 * vmax else TEXT_COLOR
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    color=color, fontsize=9.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(
        "sign(-log10 Holm-p) × sign(Cliff's δ)\n"
        "blue ⇒ TabPFN significantly wins   red ⇒ baseline significantly wins",
        fontsize=8.5, color=TEXT_COLOR,
    )
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)

    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, color=TEXT_COLOR)
    ax.set_yticks(range(len(sizes_present)))
    ax.set_yticklabels(sizes_present, color=TEXT_COLOR)
    ax.set_xlabel(f"Baseline (paired Wilcoxon, paired on dataset × seed)", color=TEXT_COLOR)
    ax.set_ylabel("Training-set size", color=TEXT_COLOR)
    ax.set_title("Statistical significance of TabPFN advantage per N\n"
                  "(*** p<0.001  ·  ** p<0.01  ·  * p<0.05  ·  ns p≥0.05 after Holm correction)",
                  fontsize=11, color=TEXT_COLOR, pad=10)

    fig.tight_layout()
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("ACT-2 PUBLICATION FIGURES  (headline set, Phase B of checklist)")
    print("=" * 70)
    df = load()
    n_seeds    = df["seed"].nunique()
    n_datasets = df["dataset"].nunique()
    n_models   = df["model"].nunique()
    n_sizes    = df["train_size"].nunique()
    print(f"Loaded {len(df)} rows, {n_models} models, {n_datasets} datasets, "
          f"{n_sizes} train sizes, {n_seeds} seeds "
          f"({sorted(df['seed'].unique())})")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[F1] per-dataset crossover with bootstrap CI (HEADLINE) ...")
    cross = plot_per_dataset_crossover(df, FIG_DIR / "per_dataset_crossover_with_ci.png")
    print(f"   datasets with crossover <= N=1000: "
          f"{int(((cross['crossover_n'] <= 1000)).sum())} / {len(cross)}")
    print(f"   summary row counts:\n{cross[['dataset','crossover_n','crossover_ci_lo','crossover_ci_hi']].to_string(index=False)}")

    print("\n[F2] memory vs N ...")
    plot_memory_vs_n(df, FIG_DIR / "multiseed_memory_vs_n.png")

    print("\n[F3] Pareto: AUC vs predict-time per row ...")
    plot_pareto_auc_vs_time(df, FIG_DIR / "multiseed_pareto_auc_vs_time.png")

    print("\n[F4] ECE vs N ...")
    plot_ece_by_n(df, FIG_DIR / "multiseed_ece_by_n.png")

    print("\n[F5] per-N significance heatmap ...")
    plot_significance_heatmap(df, FIG_DIR / "act2_significance_heatmap.png")

    print("\nAll Act-2 figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
