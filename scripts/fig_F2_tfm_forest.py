"""
F2 — TabPFN-v2 vs SAP-RPT-1 forest plot (§4.4)

Per-dataset ΔAUC = TabPFN-v2 − SAP-RPT-1 across 15 Stage-I datasets.
Each row: point = mean difference; line = 95% bootstrap CI;
significance (Holm p < 0.05) indicated by a separate marker.

Purpose: show whether the Stage-I TFM selection (TabPFN-v2 over SAP-RPT-1)
is consistent across datasets or driven by the aggregate mean.

Source CSV: experiments/results/aggregated/paired_stats.csv (scope=per_dataset,
model=SAP-RPT-1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures_style import (
    STATUS, apply_style, model_color, save,
)


def load() -> pd.DataFrame:
    df = pd.read_csv("experiments/results/aggregated/paired_stats.csv")
    df = df[(df["scope"] == "per_dataset") & (df["model"] == "SAP-RPT-1")]
    # CI columns: ci_low / ci_high are the 95% bootstrap CI on diff_mean.
    df = df[["dataset", "diff_mean", "ci_low", "ci_high",
             "cliffs_delta", "holm_p_adj", "significant"]].copy()
    # Sort by diff_mean so the strongest-wins-to-strongest-loses ordering
    # travels top-to-bottom — the reader's eye lands on the extremes first.
    df = df.sort_values("diff_mean", ascending=True).reset_index(drop=True)
    return df


def main() -> Path:
    apply_style(mode="light")
    df = load()

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    y_pos = np.arange(len(df))
    diffs = df["diff_mean"].values
    ci_lo = df["ci_low"].values
    ci_hi = df["ci_high"].values
    sigs = df["significant"].values

    # Forest-plot horizontal error bars.
    sig_mask = sigs.astype(bool)
    color_sig = model_color("TabPFN")
    color_nonsig = "#898781"  # muted grey for non-significant rows

    # Compute error bar extents (relative to point).
    err_lo = np.clip(diffs - ci_lo, 0, None)
    err_hi = np.clip(ci_hi - diffs, 0, None)

    # Draw all rows in the muted color first.
    ax.errorbar(
        diffs[~sig_mask], y_pos[~sig_mask],
        xerr=[err_lo[~sig_mask], err_hi[~sig_mask]],
        fmt="o", color=color_nonsig, ecolor=color_nonsig,
        markersize=7, markeredgecolor="white", markeredgewidth=1.0,
        elinewidth=1.2, capsize=3, zorder=2, label="not Holm-significant",
    )
    # Then overlay significant rows in TabPFN blue so they pop.
    if sig_mask.any():
        ax.errorbar(
            diffs[sig_mask], y_pos[sig_mask],
            xerr=[err_lo[sig_mask], err_hi[sig_mask]],
            fmt="o", color=color_sig, ecolor=color_sig,
            markersize=8, markeredgecolor="#0b0b0b", markeredgewidth=1.0,
            elinewidth=1.4, capsize=3, zorder=3, label="Holm-significant (p<0.05)",
        )

    # Reference line at Δ = 0 — TabPFN-v2 == SAP-RPT-1.
    ax.axvline(0, color="#0b0b0b", linewidth=1.0, zorder=1)

    # Right-side annotation: count of wins / ties / losses.
    n_win = int((diffs > 0).sum())
    n_tie = int((diffs == 0).sum())
    n_loss = int((diffs < 0).sum())
    txt = f"TabPFN-v2 wins: {n_win}/15\nTies: {n_tie}/15\nSAP-RPT-1 wins: {n_loss}/15"
    # Place at upper-left to avoid collision with the legend at lower-right.
    ax.text(
        0.020, 0.965, txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#898781",
                  boxstyle="round,pad=0.4", linewidth=0.6),
    )

    # x-axis: symmetric around 0 with a small headroom.
    xmax = float(np.nanmax(np.abs(np.r_[diffs, ci_lo, ci_hi])))
    ax.set_xlim(-xmax * 1.15, xmax * 1.35)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["dataset"].values, fontsize=9)
    ax.set_xlabel("ΔROC-AUC (TabPFN-v2 − SAP-RPT-1) — 95% bootstrap CI",
                  fontsize=10)
    ax.grid(True, axis="x", linewidth=0.5)
    # No legend: Holm-significant overlay has 0/15 members; the muted color
    # is the only treatment, so a legend would imply a non-existent category.

    fig.tight_layout()
    return save(fig, "F2_tfm_forest_plot")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
