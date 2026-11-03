"""
F10 — Evidence synthesis schematic (§6 / §7)

A minimal research-style schematic that synthesizes the paper's
message visually without introducing any new quantitative result.
This figure answers the §6 question "so what?" by mapping the four
empirical axes (AUC, dataset heterogeneity, memory, inference cost)
into the practical decision framework: pick TabPFN-v2 when the AUC
gap is the bottleneck and the deployment budget can absorb the
context-driven cost; pick a tuned GBDT/MLP when deployment cost is
the bottleneck and the dataset is in the easy/low-dim regime.

Designed per dataviz skill (artifact-diagramming for inline-SVG mechanics,
adapted to matplotlib for print publication): muted surface, hairline
chrome, blue/red poles for the protagonist-vs-baseline split, no
decorative texture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle
import numpy as np

from paper_figures_style import (
    CATEGORICAL_LIGHT, INK_MUTED, INK_PRIMARY_LIGHT,
    INK_SECONDARY_LIGHT, SURFACE_LIGHT, apply_style, model_color, save,
)


def main() -> Path:
    apply_style(mode="light")

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- 1. Title bar (no figure number — caption goes underneath) --
    # Removed — caption sits underneath in the paper.

    # -- 2. Top row: empirical axes (the four "what we measured" boxes) --
    axes_y = 5.2
    box_w, box_h = 2.1, 1.2

    boxes = [
        (1.05, "AUC", "TabPFN-v2 leads\nacross the entire N grid\n(Δ up to +0.05 at low N)"),
        (3.30, "Dataset\nheterogeneity", "Crossover N* ranges from\n25 (easy) to \"≥ full\"\n(large/high-dim)"),
        (5.55, "Peak memory", "TabPFN-v2 grows with N;\ntuned GBDTs/MLP stay below\n250 MB across the grid"),
        (7.80, "Inference cost", "TabPFN-v2 grows with N\n(ms/row up to ≈200);\ntuned GBDTs/MLP sub-ms"),
    ]
    for x, title, body in boxes:
        rect = FancyBboxPatch(
            (x, axes_y), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=SURFACE_LIGHT, edgecolor=INK_MUTED,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, axes_y + box_h - 0.20, title,
                ha="center", va="center", fontsize=10.5,
                fontweight="semibold", color=INK_PRIMARY_LIGHT)
        ax.text(x + box_w / 2, axes_y + box_h / 2 - 0.12, body,
                ha="center", va="center", fontsize=8.5,
                color=INK_SECONDARY_LIGHT, linespacing=1.25)

    # Arrow from "empirical axes" down to "decision rule".
    ax.annotate(
        "", xy=(5.0, 3.0), xytext=(5.0, 4.9),
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.0),
    )

    # -- 3. Middle row: the decision split --
    decision_y = 1.85

    # Two outcome boxes (TabPFN-v2 vs tuned baseline).
    tabpfn_x, base_x = 1.05, 5.55
    outcome_w, outcome_h = 3.90, 1.05

    tabpfn_color = model_color("TabPFN")
    base_color = model_color("CatBoost")

    rect_a = FancyBboxPatch(
        (tabpfn_x, decision_y), outcome_w, outcome_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=tabpfn_color, edgecolor="none",
        alpha=0.12,
    )
    ax.add_patch(rect_a)
    ax.add_patch(FancyBboxPatch(
        (tabpfn_x, decision_y), outcome_w, outcome_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor="none", edgecolor=tabpfn_color, linewidth=1.2,
    ))
    ax.text(
        tabpfn_x + outcome_w / 2, decision_y + outcome_h - 0.22,
        "Pick TabPFN-v2 when",
        ha="center", va="center", fontsize=11, fontweight="semibold",
        color=tabpfn_color,
    )
    ax.text(
        tabpfn_x + outcome_w / 2, decision_y + outcome_h / 2 - 0.05,
        "AUC is the deployment bottleneck\nand the budget can absorb "
        "context-driven cost",
        ha="center", va="center", fontsize=8.5,
        color=INK_PRIMARY_LIGHT, linespacing=1.30,
    )

    rect_b = FancyBboxPatch(
        (base_x, decision_y), outcome_w, outcome_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=base_color, edgecolor="none",
        alpha=0.12,
    )
    ax.add_patch(rect_b)
    ax.add_patch(FancyBboxPatch(
        (base_x, decision_y), outcome_w, outcome_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor="none", edgecolor=base_color, linewidth=1.2,
    ))
    ax.text(
        base_x + outcome_w / 2, decision_y + outcome_h - 0.22,
        "Pick tuned GBDT/MLP when",
        ha="center", va="center", fontsize=11, fontweight="semibold",
        color=base_color,
    )
    ax.text(
        base_x + outcome_w / 2, decision_y + outcome_h / 2 - 0.05,
        "Deployment cost (memory/latency) is the bottleneck\nand "
        "the dataset is in the easy / low-dim regime",
        ha="center", va="center", fontsize=8.5,
        color=INK_PRIMARY_LIGHT, linespacing=1.30,
    )

    # "VS" divider.
    ax.text(5.0, decision_y + outcome_h / 2, "vs",
            ha="center", va="center", fontsize=11,
            fontweight="semibold", color=INK_MUTED)

    # -- 4. Bottom row: the audit disclaimers --
    ax.text(
        5.0, 0.7,
        "No fixed threshold survives the leave-one-dataset-out CV cleanly "
        "(§6.3). Decision rule is dataset- and budget-dependent; "
        "no universal prescription.",
        ha="center", va="center", fontsize=8.5,
        color=INK_SECONDARY_LIGHT, style="italic",
    )

    fig.tight_layout()
    return save(fig, "F10_evidence_synthesis_schematic")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")