# Stage II: Multi-Dimensional Few-Shot Scaling

## Purpose
Stage II is the core of the research. It investigates the "utility frontier" of Tabular Foundation Models by measuring how performance, cost, and calibration evolve as the amount of available training data increases.

## Experimental Grid
The study uses a shared grid of **1,800 cells** (8 datasets $\times$ 5 models $\times$ 9 training sizes $\times$ 5 seeds).

### 1. Axes of Measurement
- **Performance:** ROC-AUC (and secondary metrics: PR-AUC, Brier).
- **Cost (Memory):** Peak RAM usage during inference.
- **Cost (Time):** Per-row inference latency (ms/row).
- **Calibration:** ECE-10 (Expected Calibration Error).

### 2. Configuration
- **Datasets:** 8 selected OpenML datasets (see `docs/research/METHODOLOGY.md`).
- **Training Sizes ($N$):** $\{25, 50, 100, 200, 300, 500, 750, 1000, \text{full}\}$.
- **Models:** TabPFN-v2, XGBoost, LightGBM, CatBoost, MLP (Tuned via Optuna).
- **Seeds:** 5 stratified subsamples per $N$.

## Key Findings
- **AUC Dominance:** TabPFN-v2 leads at every $N$, though the lead narrows as $N \to \text{full}$.
- **Crossover Heterogeneity:** The point where GBDTs match TabPFN-v2 varies wildly by dataset (from $N=25$ to $N \ge \text{full}$), proving no single "rule of thumb" for $N$ exists.
- **Memory Cost:** TabPFN-v2 peak memory grows $\approx 25\times$ as $N$ goes from 25 to 1000, whereas GBDTs remain flat.
- **Inference Cost:** A $5\text{--}10\times$ latency gap exists at full size.

## Evidence Traceability
- **Source Data:** `experiments/results/few_shot/multiseed_few_shot_results.csv` and associated paired stats.
- **Visualization Scripts:**
  - `scripts/fig_F3_headline.py` (AUC curves)
  - `scripts/fig_F6_memory.py` (Memory scaling)
  - `scripts/fig_F7_inference_time.py` (Latency)
  - `scripts/fig_F9_calibration.py` (ECE curves)
- **Reported in Paper:** Section 5 (Stage II — Multi-Dimensional Few-Shot Scaling).
