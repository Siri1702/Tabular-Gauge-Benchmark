# Experimental Methodology

This document details the scientific protocol used to generate the results in the associated research paper.

## 1. Dataset Selection & Preparation
### Selection Criteria
Datasets were selected from **OpenML-CC18** based on:
1. **Task:** Binary classification only.
2. **Size:** $n_{train} > 1,000$ after 80/20 split (to ensure "full" is meaningful).
3. **Test Set:** $n_{test} \ge 100$ (to ensure non-trivial variance).
4. **Diversity:** At most one dataset per application domain.

### Preprocessing
- **Categorical Features:** Handled by the respective model wrappers (e.g., CatBoost native handling, TabPFN's in-context processing).
- **Missing Values:** Handled by model-specific defaults (e.g., XGBoost's native missing value handling).
- **Splitting:** A fixed 80/20 train/test split was used with `random_state = 42`.

## 2. Experimental Setup
### Training-Size Grid
The training set size $N$ was varied over:
$N \in \{25, 50, 100, 200, 300, 500, 750, 1000, \text{full}\}$

### Subsampling Protocol
To enable paired statistical tests, for each seed $s$, a stratified subsample of size $N$ was drawn using `numpy.random.default_rng(s * 1000 + 31)`. This ensures that for a given $(dataset, seed, N)$ cell, all models were evaluated on the exact same training rows.

### Model Configurations
- **TabPFN-v2:** Zero-shot. `max_context_size = 8,000`.
- **Conventional Baselines:** Tuned via **Optuna** on a 3-fold stratified CV per cell.
- **Tuning Budget:** 10 trials for GBDTs, 8 trials for MLP.

## 3. Evaluation & Statistics
### Metrics
- **Primary:** ROC-AUC on the fixed held-out test split.
- **Secondary:** PR-AUC, Brier Score, ECE-10, Peak RAM (MB), Inference Time (ms/row).

### Statistical Protocol
1. **Paired Wilcoxon Signed-Rank Test:** Comparing TabPFN-v2 vs baselines per cell.
2. **Holm-Bonferroni Correction:** Applied across the four baselines at each $N$.
3. **Effect Size:** Cliff's $\delta$ used to quantify the magnitude of the difference.
4. **Confidence Intervals:** Paired bootstrap 95% CI (2,000 resamples) for AUC differences.
5. **Omnibus Testing:** Friedman test + Nemenyi post-hoc for overall model rankings.

## 4. Hardware & Environment
- **Software:** Python 3.10, `tabular-gauge==0.1.0`.
- **Dependency Lock:** All versions pinned in `scripts/requirements_lock.txt`.
- **Hardware:** [Insert Hardware Specs, e.g., NVIDIA A100 40GB, Intel Xeon Gold]
