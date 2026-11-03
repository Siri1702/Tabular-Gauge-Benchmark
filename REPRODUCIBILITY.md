# Reproducibility Guide: Tabular Foundation Models (IJERT)

This document provides a step-by-step mapping between the results presented in the paper **"Tabular Foundation Models"** and the code/data in this repository.

## 🎯 Goal
The goal of this repository is to ensure that every claim, table, and figure in the published paper can be independently verified and replicated.

---

## 📊 Mapping: Paper Results $\rightarrow$ Repository Artifacts

### 1. Foundation Model Selection (Act 1)
*   **Paper Claim:** Comparison of TabPFN vs. Gradient Boosted Decision Trees (XGBoost, CatBoost) and MLPs.
*   **Implementation:** `tabular_gauge/models/tabpfn_model.py`, `tabular_gauge/models/xgboost_model.py`, `tabular_gauge/models/catboost_model.py`.
*   **Execution:** Run `experiments/run_benchmark.py` with `configs/experiment.yaml`.
*   **Result Artifact:** `reports/tables/main_results.csv` (Derived from `experiments/results/aggregated/results.csv`).

### 2. Few-Shot Scaling Analysis (Act 2)
*   **Paper Claim:** TabPFN maintains superior performance in low-data regimes (Few-Shot) compared to traditional models.
*   **Implementation:** `tabular_gauge/analysis/few_shot.py`.
*   **Execution:** Run `experiments/run_benchmark.py` using `configs/datasets_few_shot.yaml`.
*   **Result Artifact:** `reports/tables/few_shot_scaling.csv` (Derived from `experiments/results/few_shot/multiseed_few_shot_results.csv`).
*   **Visual Evidence:** `reports/figures/paper_figures/headline_learning_curve.png`.

### 3. Statistical Significance
*   **Paper Claim:** Performance gains are statistically significant based on paired non-parametric tests.
*   **Implementation:** `tabular_gauge/evaluation/statistical.py` and `tabular_gauge/evaluation/paired_stats.py`.
*   **Note**: We employ the **paired Wilcoxon signed-rank test** with **Holm-Bonferroni correction** and **Cliff's $\delta$** to ensure robustness against non-normal distributions and multiple comparison bias.
*   **Result Artifact:** `reports/tables/paired_stats.csv` (Derived from `experiments/results/aggregated/paired_stats.csv`).

### 4. Crossover and Win-Loss Analysis
*   **Paper Claim:** Analysis of where TabPFN wins vs. where GBDTs win.
*   **Implementation:** `tabular_gauge/analysis/few_shot.py`.
*   **Result Artifact:** `experiments/results/crossover/comprehensive_analysis.csv`.
*   **Visual Evidence:** `reports/figures/paper_figures/per_dataset_crossover_with_ci.png`.

### 5. Ablation Study (Label Shuffling)
*   **Paper Claim:** TabPFN's few-shot advantage is due to genuine in-context learning.
*   **Implementation:** `experiments/run_ablation_study.py`.
*   **Result Artifact:** `reports/ablation/shuffled_labels_results.csv`.

---

## 🛠️ How to Replicate Experiments

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure your datasets are placed in the `data/` directory (see `tabular_gauge/data/loader.py` for supported formats).

### Step-by-Step Replication
1. **Run Full Benchmark:**
   ```bash
   python experiments/run_benchmark.py --config configs/experiment.yaml
   ```
2. **Run Few-Shot Analysis:**
   ```bash
   python experiments/run_benchmark.py --config configs/datasets_few_shot.yaml
   ```
3. **Run Ablation Study:**
   ```bash
   python experiments/run_ablation_study.py
   ```
4. **Generate Tables/Plots:**
   ```bash
   python experiments/regenerate_report.py
   ```

## 🔍 Deep-Dive for Reviewers
For reviewers wishing to examine the raw, un-aggregated data:
- **Raw Run JSONs:** Every single experiment run is saved as a JSON file in `experiments/results/raw/`.
- **Supplemental Plots:** Detailed heatmaps, Pareto curves, and latency rankings are located in `reports/figures/supplemental/`.
- **Dataset Metadata:** Characteristics of all benchmark datasets are available in `reports/dataset_characteristics_regression.json`.
