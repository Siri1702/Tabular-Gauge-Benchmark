# Stage I: Foundation-Model Selection

## Purpose
Stage I serves as a "model filter." Before conducting the expensive multi-dimensional scaling study in Stage II, we evaluate the two leading open-weight Tabular Foundation Models (TFMs) to determine which provides the strongest baseline for the research.

## Experimental Matrix
| Axis | Configuration |
|---|---|
| **Datasets** | 15 OpenML-CC18 binary classification tasks |
| **Models** | TabPFN-v2, SAP-RPT-1 |
| **Seeds** | $\{0, 1, 2, 3, 4\}$ |
| **Training Size** | Full training set (no $N$-sweep) |
| **Primary Metric** | ROC-AUC on fixed 20% held-out test split |
| **Stat Protocol** | Paired Wilcoxon $\rightarrow$ Holm-Bonferroni $\rightarrow$ Cliff's $\delta$ |

## Key Results
- **Headline:** TabPFN-v2 outperformed SAP-RPT-1 in **57 of 75** paired cells (76%).
- **Significance:** Paired Wilcoxon $p < 10^{-9}$ (Holm-adjusted), Cliff's $\delta = +0.587$ (Large effect).
- **Outcome:** TabPFN-v2 was selected as the primary TFM for Stage II.

## Evidence Traceability
- **Source Data:** `experiments/results/aggregated/results.csv` and `experiments/results/aggregated/paired_stats.csv`.
- **Visualization Scripts:**
  - `scripts/fig_F1_stage1_overall.py`
  - `scripts/fig_F2_tfm_forest.py`
- **Reported in Paper:** Section 4 (Stage I — Foundation-Model Selection).
