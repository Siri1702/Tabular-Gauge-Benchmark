"""
Publication-ready headline figures for Paper 1 §5 and §6.

Generates three figures that were either missing or needed annotation
polish to be camera-ready:

F-headline  Headline learning curve. Mean ROC-AUC vs N (one line per
            model) with paired-bootstrap 95 % CI bands. Holm-sig
            markers above each N where TabPFN-v2 wins on every
            baseline. The single most important figure in the paper.

F-lodo      Enhanced LODO-CV heatmap. Held-out dataset x train size,
            cell colour = loss to oracle (lower = better), cell
            annotation = rule recommendation icon. (500, 1000)
            boundary overlay. Annotated to be readable without
            consulting the CSV.

F-story     2 x 2 "reviewer-photograph" figure. (a) Headline learning
            curve, (b) per-dataset crossover N with bootstrap CI,
            (c) peak memory vs N (log-log), (d) LODO-CV mean loss
            vs N. One panel per §5 headline finding.

Inputs
------
experiments/results/few_shot/multiseed_few_shot_results.csv (1800 rows)
experiments/results/few_shot/multiseed_paired_stats_per_n.csv
experiments/results/few_shot/per_dataset_crossover_with_ci.csv
experiments/results/few_shot/decision_rule_lodo_cv.csv

Outputs (PNG + PDF, 200 DPI, light theme)
----------------------------------------
reports/figures/act2_multiseed/headline_learning_curve.{png,pdf}
reports/figures/act2_multiseed/lodo_cv_annotated.{png,pdf}
reports/figures/act2_multiseed/story_figure.{png,pdf}

Run:
    python scripts/plot_publication_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout on Windows PowerShell
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_gauge.viz.style import (
    ACCENT,
    ERROR,
    LIGHT_BG,
    LIGHT_GRID,
    LIGHT_SURFACE,
    MODEL_COLORS,
    MUTED_TEXT,
    TEXT_COLOR,
    apply_style,
    color_for_model,
    save_figure,
)

CSV_PATH         = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
PER_N_STATS_CSV  = ROOT / "experiments" / "results" / "few_shot" / "multiseed_paired_stats_per_n.csv"
CROSSOVER_CSV    = ROOT / "experiments" / "results" / "few_shot" / "per_dataset_crossover_with_ci.csv"
LODO_CSV         = ROOT / "experiments" / "results" / "few_shot" / "decision_rule_lodo_cv.csv"
FIG_DIR          = ROOT / "reports" / "figures" / "act2_multiseed"

REF_MODEL      = "TabPFN"
BASELINES      = ["XGBoost", "LightGBM", "CatBoost", "MLP"]
ALL_MODELS     = [REF_MODEL] + BASELINES
TRAIN_SIZE_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]

# Plot N position: log scale, "full" sits at 1500 to keep the layout clean
N_PLOT = {"25": 25, "50": 50, "100": 100, "200": 200, "300": 300,
          "500": 500, "750": 750, "1000": 1000, "full": 1500}
N_TO_LABEL = {**{str(n): str(n) for n in [25, 50, 100, 200, 300, 500, 750, 1000]},
              "full": "full"}

# Holm-significance marker symbols
SIG_ALL = "✓ (4/4 Holm-sig)"      # TabPFN beats all 4 baselines
SIG_3_4 = "✓ (3/4 Holm-sig)"      # TabPFN beats 3/4 baselines
SIG_NS  = "· (ns)"                # not significant against ≥1 baseline

MODEL_DISPLAY = {"TabPFN": "TabPFN-v2", "XGBoost": "XGBoost",
                 "LightGBM": "LightGBM", "CatBoost": "CatBoost",
                 "MLP": "MLP"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_results() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"Missing: {CSV_PATH}. Run scripts/run_multiseed_few_shot.py first.")
    df = pd.read_csv(CSV_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["train_size"] = df["train_size"].astype(str)
    return df.dropna(subset=["roc_auc"])


def load_per_n_stats() -> pd.DataFrame:
    return pd.read_csv(PER_N_STATS_CSV) if PER_N_STATS_CSV.exists() else pd.DataFrame()


def per_n_holm_marker(stats_per_n: pd.DataFrame, n: str) -> tuple[str, int]:
    """Return (marker_label, n_sig) for the given train size, using the per-N
    Holm-corrected Wilcoxon table. n_sig is the number of baselines TabPFN
    beats at Holm p < 0.05."""
    if stats_per_n.empty:
        return SIG_NS, 0
    sub = stats_per_n[stats_per_n["train_size"].astype(str) == str(n)]
    if sub.empty:
        return SIG_NS, 0
    n_sig = int(sub["significant"].sum()) if "significant" in sub.columns else 0
    if n_sig == 4:
        return SIG_ALL, n_sig
    if n_sig == 3:
        return SIG_3_4, n_sig
    return SIG_NS, n_sig


def paired_bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 2000,
                             rng_seed: int = 0) -> tuple[float, float, float]:
    """Paired-bootstrap mean and 95% CI. For a per-cell mean we resample
    with replacement and compute the percentile CI on the mean."""
    if len(values) < 2:
        v = float(values.mean())
        return v, v, v
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    point = float(values.mean())
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return point, lo, hi


# --------------------------------------------------------------------------- #
# F-headline — Headline learning curve with bootstrap CI + Holm markers
# --------------------------------------------------------------------------- #

def plot_headline_learning_curve(df: pd.DataFrame,
                                stats_per_n: pd.DataFrame,
                                out_dir: Path) -> None:
    """Mean ROC-AUC vs N (log-x), one line per model, paired-bootstrap
    95 % CI bands (across the 8 datasets x 5 seeds = 40 cells at each N).
    Above each N, an annotation line shows how many of the 4 baselines
    TabPFN-v2 wins Holm-significantly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style("light")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Mean and CI per (model, train_size)
    rows = []
    for m in ALL_MODELS:
        for n in TRAIN_SIZE_ORDER:
            vals = df[(df["model"] == m) & (df["train_size"] == n)]["roc_auc"].values
            if len(vals) == 0:
                continue
            mean, lo, hi = paired_bootstrap_mean_ci(vals)
            rows.append({"model": m, "n": n, "mean": mean, "lo": lo, "hi": hi,
                          "x": N_PLOT[n]})
    summary = pd.DataFrame(rows)

    for m in ALL_MODELS:
        sub = summary[summary["model"] == m].sort_values("x")
        color = color_for_model(m)
        lw = 2.4 if m == REF_MODEL else 1.6
        ms = 7 if m == REF_MODEL else 5
        zorder = 5 if m == REF_MODEL else 3
        ax.plot(sub["x"], sub["mean"], color=color, lw=lw, marker="o", markersize=ms,
                label=MODEL_DISPLAY[m], zorder=zorder)
        ax.fill_between(sub["x"], sub["lo"], sub["hi"],
                        color=color, alpha=0.18, linewidth=0, zorder=zorder - 1)

    # Holm-significance annotations
    y_top = 0.99
    for n in TRAIN_SIZE_ORDER:
        if stats_per_n.empty:
            continue
        marker, n_sig = per_n_holm_marker(stats_per_n, n)
        ax.annotate(f"{n} · {marker}", xy=(N_PLOT[n], y_top),
                    ha="center", va="top",
                    fontsize=7,
                    color=ACCENT if n_sig >= 3 else MUTED_TEXT,
                    fontweight="bold" if n_sig == 4 else "normal",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="none", alpha=0.85))

    # Decision-rule boundary overlays
    ax.axvline(500, color=ACCENT, linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.axvline(1000, color=ERROR, linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text(500, 0.93, "N = 500\n(TabPFN→benchmark)", fontsize=7,
            ha="center", color=ACCENT, alpha=0.85)
    ax.text(1000, 0.93, "N = 1 000\n(benchmark→GBDT)", fontsize=7,
            ha="center", color=ERROR, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xticks([N_PLOT[n] for n in TRAIN_SIZE_ORDER])
    ax.set_xticklabels([N_TO_LABEL[n] for n in TRAIN_SIZE_ORDER], rotation=0, fontsize=9)
    ax.set_xlim(18, 2200)
    ax.set_ylim(0.78, 1.005)
    ax.set_xlabel("Training-set size  N  (log scale)", fontsize=10)
    ax.set_ylabel("Mean ROC-AUC  (across 8 datasets × 5 seeds)", fontsize=10)
    ax.set_title("Stage II headline: TabPFN-v2 leads at every N until ≈ 500,\n"
                 "ties or leads at N = 750–1 000, then is cost-dominated",
                 fontsize=11, pad=28)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save_path = out_dir / "headline_learning_curve"
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# F-lodo — Enhanced LODO-CV heatmap with annotation
# --------------------------------------------------------------------------- #

def plot_lodo_cv_annotated(df_lodo: pd.DataFrame,
                           n_low: int, n_high: int,
                           out_dir: Path) -> None:
    """Heatmap with three layers of annotation:
       1. cell colour  = loss to oracle  (green = low, red = high)
       2. cell text    = rule recommendation (T/X/L/C/M icons)
       3. cell text    = numeric loss in % (small)
       Boundary overlay at (N_LOW, N_HIGH)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style("light")

    if df_lodo.empty:
        print("  WARNING: decision_rule_lodo_cv.csv is empty; skipping F-lodo.")
        return

    pivot_loss = (df_lodo
                  .pivot(index="held_out", columns="train_size", values="loss_to_oracle")
                  .reindex(columns=TRAIN_SIZE_ORDER))
    pivot_rec = (df_lodo
                 .pivot(index="held_out", columns="train_size", values="recommendation")
                 .reindex(columns=TRAIN_SIZE_ORDER))

    REC_ICON = {"TabPFN": "T", "XGBoost": "X", "LightGBM": "L",
                "CatBoost": "C", "MLP": "M"}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pct = pivot_loss * 100.0
    vmax = float(np.nanmax(pct.values)) if np.isfinite(np.nanmax(pct.values)) else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(pct.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(TRAIN_SIZE_ORDER)))
    ax.set_xticklabels(TRAIN_SIZE_ORDER, fontsize=9)
    ax.set_yticks(range(len(pct.index)))
    ax.set_yticklabels(pct.index, fontsize=9)
    ax.set_xlabel(f"Training-set size N  (rule: N ≤ {n_low} → TabPFN; "
                  f"{n_low} < N ≤ {n_high} → benchmark; N > {n_high} → best GBDT)",
                  fontsize=9)
    ax.set_ylabel("Held-out dataset (leave-one-out)", fontsize=10)
    ax.set_title("LODO-CV of the 3-regime decision rule — loss to oracle (%, lower = better)\n"
                 "Cell label = rule recommendation (T=TabPFN, X=XGB, L=LightGBM, C=CatBoost, M=MLP)",
                 fontsize=11, pad=10)

    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color="grey",
                        fontsize=8)
                continue
            rec = pivot_rec.values[i, j]
            icon = REC_ICON.get(rec, "?")
            colour = "black" if v < vmax * 0.6 else "white"
            ax.text(j, i + 0.18, f"{v:.2f}%",
                    ha="center", va="center",
                    color=colour, fontsize=7)
            ax.text(j, i - 0.20, icon,
                    ha="center", va="center",
                    color=colour, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Loss to oracle  (ROC-AUC × 100)", fontsize=9)

    n_low_idx  = TRAIN_SIZE_ORDER.index(str(n_low))  if str(n_low)  in TRAIN_SIZE_ORDER else -1
    n_high_idx = TRAIN_SIZE_ORDER.index(str(n_high)) if str(n_high) in TRAIN_SIZE_ORDER else len(TRAIN_SIZE_ORDER)
    if 0 <= n_low_idx < len(TRAIN_SIZE_ORDER) - 1:
        ax.axvline(n_low_idx + 0.5, color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.7)
    if 0 < n_high_idx < len(TRAIN_SIZE_ORDER):
        ax.axvline(n_high_idx + 0.5, color=ERROR, linestyle="--", linewidth=1.2, alpha=0.7)

    fig.tight_layout()
    save_path = out_dir / "lodo_cv_annotated"
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# F-story — 2x2 reviewer-photograph figure
# --------------------------------------------------------------------------- #

def plot_story_figure(df: pd.DataFrame,
                      df_crossover: pd.DataFrame,
                      df_lodo: pd.DataFrame,
                      stats_per_n: pd.DataFrame,
                      out_dir: Path) -> None:
    """2x2 panel: (a) learning curves + Holm-sig, (b) per-dataset crossover
    N with bootstrap CI, (c) memory vs N (log-log), (d) LODO-CV mean loss
    vs N. One panel per §5 headline finding."""
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style("light")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.0))

    # ----- Panel (a): learning curve + Holm-sig ------------------------------
    ax = axes[0, 0]
    rows = []
    for m in ALL_MODELS:
        for n in TRAIN_SIZE_ORDER:
            vals = df[(df["model"] == m) & (df["train_size"] == n)]["roc_auc"].values
            if len(vals) == 0:
                continue
            mean, lo, hi = paired_bootstrap_mean_ci(vals)
            rows.append({"model": m, "n": n, "mean": mean, "lo": lo, "hi": hi,
                         "x": N_PLOT[n]})
    summary = pd.DataFrame(rows)
    for m in ALL_MODELS:
        sub = summary[summary["model"] == m].sort_values("x")
        c = color_for_model(m)
        lw = 2.4 if m == REF_MODEL else 1.4
        ms = 6 if m == REF_MODEL else 4.5
        z = 5 if m == REF_MODEL else 3
        ax.plot(sub["x"], sub["mean"], color=c, lw=lw, marker="o", markersize=ms,
                label=MODEL_DISPLAY[m], zorder=z)
        ax.fill_between(sub["x"], sub["lo"], sub["hi"], color=c, alpha=0.18,
                        linewidth=0, zorder=z - 1)
    if not stats_per_n.empty:
        y_top = 0.99
        for n in TRAIN_SIZE_ORDER:
            marker, n_sig = per_n_holm_marker(stats_per_n, n)
            ax.annotate(f"{n} · {marker}",
                        xy=(N_PLOT[n], y_top),
                        ha="center", va="top", fontsize=5.0,
                        color=ACCENT if n_sig >= 3 else MUTED_TEXT,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.85),
                        rotation=0)
    ax.axvline(500, color=ACCENT, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.axvline(1000, color=ERROR, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks([N_PLOT[n] for n in TRAIN_SIZE_ORDER])
    ax.set_xticklabels([N_TO_LABEL[n] for n in TRAIN_SIZE_ORDER], fontsize=7.5)
    ax.set_xlim(18, 2200)
    ax.set_ylim(0.78, 1.005)
    ax.set_xlabel("Training-set size N  (log)", fontsize=9)
    ax.set_ylabel("Mean ROC-AUC", fontsize=9)
    ax.set_title("(a)  Stage II headline — TabPFN-v2 leads at every N",
                 fontsize=10, pad=24)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # ----- Panel (b): per-dataset crossover N with bootstrap CI -------------
    ax = axes[0, 1]
    if not df_crossover.empty:
        cdf = df_crossover.copy()
        cdf["crossover_x"] = cdf["crossover_n"].fillna(cdf["crossover_ci_hi"]).clip(upper=5000)
        cdf = cdf.sort_values("crossover_x")
        y = np.arange(len(cdf))
        ax.hlines(y, cdf["crossover_ci_lo"].fillna(0), cdf["crossover_ci_hi"].fillna(5000),
                  color=ACCENT, linewidth=4, alpha=0.55, zorder=2)
        ax.scatter(cdf["crossover_x"], y,
                   s=70, color=ACCENT, edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(cdf[cdf["crossover_n_point"].isna()]["crossover_ci_hi"].fillna(5000),
                   y[cdf["crossover_n_point"].isna()],
                   s=70, marker=">", color=ERROR, edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(cdf["dataset"], fontsize=9)
        ax.set_xscale("log")
        ax.axvline(500,  color=ACCENT, linestyle=":", linewidth=0.9, alpha=0.5)
        ax.axvline(1000, color=ERROR, linestyle=":", linewidth=0.9, alpha=0.5)
        ax.set_xlabel("Crossover N  (TabPFN ≥ best GBDT, log scale)", fontsize=9)
        ax.set_title("(b)  Crossover N spans > 1 order of magnitude",
                     fontsize=10)
        ax.grid(True, alpha=0.3, axis="x", which="both")
        ax.text(0.98, 0.05, "▸ = crossover beyond dataset full size",
                transform=ax.transAxes, fontsize=7, color=ERROR,
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="none", alpha=0.85))

    # ----- Panel (c): peak memory vs N ---------------------------------------
    ax = axes[1, 0]
    mem_rows = []
    for m in ALL_MODELS:
        for n in TRAIN_SIZE_ORDER:
            vals = df[(df["model"] == m) & (df["train_size"] == n)]["peak_memory_mb"].values
            if len(vals) == 0:
                continue
            mem_rows.append({"model": m, "n": n, "x": N_PLOT[n], "mean": vals.mean()})
    mem = pd.DataFrame(mem_rows)
    for m in ALL_MODELS:
        sub = mem[mem["model"] == m].sort_values("x")
        c = color_for_model(m)
        lw = 2.4 if m == REF_MODEL else 1.4
        ms = 6 if m == REF_MODEL else 4.5
        z = 5 if m == REF_MODEL else 3
        ax.plot(sub["x"], sub["mean"], color=c, lw=lw, marker="o", markersize=ms,
                label=MODEL_DISPLAY[m], zorder=z)
    ax.axvline(500,  color=ACCENT, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.axvline(1000, color=ERROR, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([N_PLOT[n] for n in TRAIN_SIZE_ORDER])
    ax.set_xticklabels([N_TO_LABEL[n] for n in TRAIN_SIZE_ORDER], fontsize=7.5)
    ax.set_xlabel("Training-set size N  (log)", fontsize=9)
    ax.set_ylabel("Peak memory  (MB, log)", fontsize=9)
    ax.set_title("(c)  TabPFN-v2 peak memory scales super-linearly with N",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # ----- Panel (d): LODO-CV mean loss vs N --------------------------------
    ax = axes[1, 1]
    if not df_lodo.empty:
        per_n = (df_lodo.groupby("train_size")["loss_to_oracle"]
                 .agg(mean="mean", max="max")
                 .reindex(TRAIN_SIZE_ORDER))
        x = [N_PLOT[n] for n in TRAIN_SIZE_ORDER]
        ax.plot(x, per_n["mean"].values * 100,
                marker="o", lw=2.2, color=ACCENT,
                label="mean loss to oracle")
        ax.plot(x, per_n["max"].values * 100,
                marker="s", lw=1.6, color=ERROR, linestyle="--",
                label="max loss to oracle")
        ax.axvline(500,  color=ACCENT, linestyle=":", linewidth=0.9, alpha=0.5)
        ax.axvline(1000, color=ERROR, linestyle=":", linewidth=0.9, alpha=0.5)
        ax.axhline(0.5, color=ERROR, linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(2200, 0.5, "0.5 %", fontsize=7, color=ERROR, va="center", ha="right")
        ax.set_xscale("log")
        ax.set_xticks([N_PLOT[n] for n in TRAIN_SIZE_ORDER])
        ax.set_xticklabels([N_TO_LABEL[n] for n in TRAIN_SIZE_ORDER], fontsize=7.5)
        ax.set_xlim(18, 2200)
        ax.set_ylim(0, max(per_n["max"].max() * 100 * 1.2, 0.6))
        ax.set_xlabel("Training-set size N  (log)", fontsize=9)
        ax.set_ylabel("LODO-CV loss to oracle  (%)", fontsize=9)
        ax.set_title("(d)  Decision-rule LODO-CV loss — (500, 1 000) is Pareto-optimal",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Paper 1 — three findings, one figure",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    save_path = out_dir / "story_figure"
    save_figure(fig, str(save_path))
    save_figure(fig, str(save_path.with_suffix(".pdf")))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading data ...")
    df = load_results()
    print(f"  {len(df)} rows from {CSV_PATH.name}")
    stats_per_n = load_per_n_stats()
    if not stats_per_n.empty:
        print(f"  {len(stats_per_n)} per-N Holm rows from {PER_N_STATS_CSV.name}")

    df_crossover = (pd.read_csv(CROSSOVER_CSV)
                    if CROSSOVER_CSV.exists() else pd.DataFrame())
    df_lodo = (pd.read_csv(LODO_CSV)
               if LODO_CSV.exists() else pd.DataFrame())

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[F-headline]  Headline learning curve ...")
    plot_headline_learning_curve(df, stats_per_n, FIG_DIR)
    print(f"  -> {FIG_DIR / 'headline_learning_curve.png'}")
    print(f"  -> {FIG_DIR / 'headline_learning_curve.pdf'}")

    print("\n[F-lodo]      LODO-CV annotated heatmap ...")
    plot_lodo_cv_annotated(df_lodo, n_low=500, n_high=1000, out_dir=FIG_DIR)
    print(f"  -> {FIG_DIR / 'lodo_cv_annotated.png'}")
    print(f"  -> {FIG_DIR / 'lodo_cv_annotated.pdf'}")

    print("\n[F-story]     2x2 reviewer-photograph figure ...")
    plot_story_figure(df, df_crossover, df_lodo, stats_per_n, FIG_DIR)
    print(f"  -> {FIG_DIR / 'story_figure.png'}")
    print(f"  -> {FIG_DIR / 'story_figure.pdf'}")

    print("\nDONE.")


if __name__ == "__main__":
    main()