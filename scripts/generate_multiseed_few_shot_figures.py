"""
Generate publication-ready figures for the multi-seed few-shot benchmark.

Inputs : experiments/results/few_shot/multiseed_few_shot_results.csv  (one row
         per (dataset, train_size, model, seed))
Outputs:
    reports/figures/multiseed_per_n_curves.png
        Mean ROC-AUC vs train_size with paired-bootstrap CI bands, one
        line per model (TabPFN in front, baselines behind).
    reports/figures/multiseed_per_dataset_heatmap.png
        Cliff's δ heatmap (dataset × model) for the six few-shot datasets.

Conventions:
    All figures follow the shared light theme in tabular_gauge/viz/style.py so they
    are visually consistent with the rest of the report. The pairing unit
    is `(dataset, seed)` — TabPFN and the baseline are compared on the
    same cells.

Run:
    python scripts/generate_multiseed_few_shot_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_gauge.evaluation.paired_stats import (
    cliffs_delta,
    paired_bootstrap_diff,
)

from tabular_gauge.viz.style import (
    MODEL_COLORS,
    LIGHT_BG,
    LIGHT_SURFACE,
    LIGHT_GRID,
    MUTED_TEXT,
    TEXT_COLOR,
    ACCENT,
    ERROR,
    color_for_model,
    palette,
    save_figure,
)


CSV_PATH = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
FIG_DIR  = ROOT / "reports" / "figures"

# Few-shot benchmark does NOT include SAP-RPT-1 (zero-shot, no training set).
# Order matches the canonical ranking used elsewhere in the report.
MODEL_ORDER = ["TabPFN", "XGBoost", "LightGBM", "CatBoost", "MLP"]

# Canonical ordering of train sizes for the x-axis (small → large, 'full' last)
TRAIN_SIZE_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}. "
                 "Run: python scripts/run_multiseed_few_shot.py first.")
    df = pd.read_csv(CSV_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["roc_auc", "seed", "model", "dataset", "train_size"])
    df["seed"] = df["seed"].astype(int)
    return df


def _train_size_to_numeric(s):
    """Convert '25' / '50' / ... / 'full' to numeric x positions."""
    return np.arange(len(TRAIN_SIZE_ORDER))


def _filter_train_sizes(present: pd.Series) -> list[str]:
    """Return the canonical TRAIN_SIZE_ORDER subset that appears in `present`."""
    present_str = set(present.astype(str).tolist())
    return [t for t in TRAIN_SIZE_ORDER if t in present_str]


# --------------------------------------------------------------------------- #
# Figure A — Mean ROC-AUC vs train_size with paired-bootstrap CI bands
# --------------------------------------------------------------------------- #
def plot_per_n_curves(df: pd.DataFrame, save_path: Path) -> None:
    """
    For each (model, train_size), plot the seed-mean ROC-AUC and a
    2 000-resample paired-bootstrap 95% CI for the mean. CI is computed
    across the `(dataset, seed)` cells per N — the same pairing unit
    used in scripts/analyze_multiseed_few_shot.py.
    """
    metric = "roc_auc"
    n_resamples = 2000

    # Restrict to models in MODEL_ORDER and to canonical train sizes.
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    sizes_present = _filter_train_sizes(df["train_size"])
    df = df[df["train_size"].astype(str).isin(sizes_present)].copy()

    x_pos = {t: i for i, t in enumerate(sizes_present)}

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Draw baselines first so TabPFN ends on top.
    plot_order = [m for m in MODEL_ORDER if m != "TabPFN"] + ["TabPFN"]

    for model in plot_order:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        xs, means, los, his = [], [], [], []
        for t in sizes_present:
            cell = sub[sub["train_size"].astype(str) == t]
            if cell.empty:
                continue
            scores = cell[metric].values
            if len(scores) >= 2:
                # Bootstrapping the mean: pass `scores` minus a zero baseline.
                boot = paired_bootstrap_diff(
                    scores, np.zeros_like(scores),
                    n_resamples=n_resamples,
                    seed=0,
                )
            else:
                m = float(scores.mean())
                boot = {"mean": m, "ci_low": m, "ci_high": m}
            xs.append(x_pos[t])
            means.append(boot["mean"])
            los.append(boot["ci_low"])
            his.append(boot["ci_high"])

        xs, means, los, his = map(np.asarray, (xs, means, los, his))
        color = color_for_model(model)
        linewidth = 2.4 if model == "TabPFN" else 1.6
        markersize = 7 if model == "TabPFN" else 5
        ax.plot(xs, means, marker="o", color=color,
                linewidth=linewidth, markersize=markersize,
                markeredgecolor="white", markeredgewidth=0.6,
                label=model, zorder=(3 if model == "TabPFN" else 2))
        ax.fill_between(xs, los, his, color=color, alpha=0.15, zorder=1)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(sizes_present, color=TEXT_COLOR)
    ax.set_xlabel("Training-set size (rows)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Mean ROC-AUC  (across datasets × seeds)",
                  color=TEXT_COLOR, fontsize=10)
    ax.set_title(
        "Few-shot learning curves — mean ROC-AUC ± 95% bootstrap CI",
        fontsize=11, color=TEXT_COLOR, pad=10,
    )
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y", alpha=0.4, color=LIGHT_GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              labelcolor=TEXT_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR)

    fig.tight_layout()
    save_figure(fig, str(save_path))


# --------------------------------------------------------------------------- #
# Figure B — Cliff's δ heatmap (dataset × model)
# --------------------------------------------------------------------------- #
def plot_cliffs_delta_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    For each (dataset, model) cell, compute Cliff's δ of the TabPFN vs
    baseline scores across seeds × train_sizes. Plot as a diverging heatmap.
    """
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    datasets = sorted(df["dataset"].unique())
    baselines = [m for m in MODEL_ORDER if m != "TabPFN"]

    mat = np.full((len(datasets), len(baselines)), np.nan)
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        ref = sub[sub["model"] == "TabPFN"].set_index(["train_size", "seed"])["roc_auc"]
        for j, model in enumerate(baselines):
            other = sub[sub["model"] == model].set_index(["train_size", "seed"])["roc_auc"]
            paired_idx = ref.index.intersection(other.index)
            if len(paired_idx) < 2:
                continue
            a = ref.loc[paired_idx].values
            b = other.loc[paired_idx].values
            mat[i, j] = cliffs_delta(a, b)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    try:
        import seaborn as sns
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    except Exception:
        # Fallback if seaborn missing: build a red/blue diverging cmap.
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "rb", ["#ef4444", "#f8fafc", "#5b8dee"]
        )

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-1.0, vmax=1.0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > 0.5 else TEXT_COLOR
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Cliff's δ  (positive ⇒ TabPFN wins)",
                   fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)

    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, rotation=20, ha="right",
                       fontsize=10, color=TEXT_COLOR)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9, color=TEXT_COLOR)

    ax.set_title(
        "Few-shot Cliff's δ per dataset (TabPFN vs baseline)\n"
        "Pairing: (train_size, seed); |δ|<0.147 negligible · "
        "<0.33 small · <0.474 medium · ≥0.474 large",
        fontsize=10.5, color=TEXT_COLOR, pad=10,
    )

    fig.tight_layout()
    save_figure(fig, str(save_path))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("MULTI-SEED FEW-SHOT FIGURES")
    print("=" * 70)
    df = load()
    print(f"Loaded {CSV_PATH.relative_to(ROOT)}: {len(df)} rows, "
          f"{df['model'].nunique()} models, {df['dataset'].nunique()} datasets, "
          f"{df['seed'].nunique()} seeds")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Per-N learning curves with paired-bootstrap CI ...")
    plot_per_n_curves(df, FIG_DIR / "multiseed_per_n_curves.png")

    print("\n[2/2] Per-dataset Cliff's δ heatmap ...")
    plot_cliffs_delta_heatmap(df, FIG_DIR / "multiseed_per_dataset_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
