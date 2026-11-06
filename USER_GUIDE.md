# 🛠️ Tabular-Gauge: Comprehensive User Guide

Welcome to **Tabular-Gauge**, a professional-grade evaluation suite for tabular machine learning models. This guide provides everything you need to run exhaustive benchmarks on your own datasets and models, covering performance, calibration, and operational efficiency.

---

## 🌟 1. Overview: The "Holistic View" Philosophy

Traditional tabular ML benchmarks often suffer from a **"Performance-Only" bias**, focusing exclusively on a single metric like ROC-AUC. In real-world deployment, this is insufficient. A model with a slightly higher AUC is a liability if it is poorly calibrated (overconfident) or computationally prohibitive.

**Tabular-Gauge addresses these limitations by providing a one-stop shop for practitioners.** With a single command, you can evaluate a model's "Full Utility" across three dimensions:

1.  **Predictive Performance**: How accurate is the model? (AUC-ROC, PR-AUC, F1-Macro).
2.  **Probability Calibration**: Can you trust the predicted probabilities for decision-making? (Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Reliability Diagrams).
3.  **Operational Efficiency**: What is the real-world cost? (Tuning time, Fit time, Inference latency, Peak RAM).

---

## 🚀 2. Installation & Quick Start

### Prerequisites
- Python 3.10+
- Virtual Environment (Recommended)

### Setup
```bash
git clone https://github.com/Siri1702/Tabular-Gauge-Benchmark.git
cd Tabular-Gauge-Benchmark
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\Activate.ps1 # Windows
```

### API Configuration (Required for TabPFN 3.5)
TabPFN requires a valid API key to function. If the key is missing, the tool will automatically skip TabPFN and continue with the other models.

**First-Time Setup (Authentication):**
If you are using TabPFN for the first time, you must authenticate your account to obtain a key:
1. Log in or register at [https://ux.priorlabs.ai/account](https://ux.priorlabs.ai/account).
2. **Crucial:** Accept the license agreement at [https://ux.priorlabs.ai/account/licenses](https://ux.priorlabs.ai/account/licenses).
3. Copy your unique **API Key** from the account dashboard.

**Updating Your API Key:**
You must create a `.env` file in the **root directory** of the project (the same folder where `README.md` is located):
1. Create a new text file named `.env`
2. Add your key exactly as follows:
   ```env
   TABPFN_API_KEY=your_api_key_here
   ```
*Note: Ensure there are no spaces around the `=` sign.*

**⚠️ Troubleshooting Tip:**
If you notice that TabPFN is being skipped during your run, check the following:
- Is the `.env` file in the root directory?
- Is the variable name exactly `TABPFN_API_KEY`?
- Did you accept the license agreement at [https://ux.priorlabs.ai/account/licenses](https://ux.priorlabs.ai/account/licenses)?

### Installation Options
The toolkit uses optional dependency groups so you only install what you need:
- **Core Framework**: `pip install -e .`
- **Conventional Baselines**: `pip install -e ".[baselines]"` (XGBoost, LightGBM, CatBoost)
- **Foundation Models**: `pip install -e ".[foundation]"` (TabPFN 3.5 - Latest Version)
- **Full Suite**: `pip install -e ".[baselines,foundation]"`

### Your First Run
Run the suite on all built-in datasets to see the pipeline in action. We recommend using the `--force` flag for your first run to ensure all results are computed freshly:
```bash
# Step 1: Run the benchmark (compute metrics)
python experiments/run_benchmark.py --all --force

# Step 2: Generate the reports (create figures and tables)
python experiments/generate_reports.py
```
**Important:** Running the benchmark only computes the raw numbers. You **must** run the report generator afterward to create the visual heatmaps, rank charts, and the summary narrative.

**Note on Caching:** 
Tabular-Gauge saves intermediate results to `.cache.json` files. 
- **With `--force`**: All existing caches for the current experiment are deleted, and models are re-trained from scratch.
- **Without `--force`**: The system will skip already-computed model/seed pairs, allowing you to resume interrupted runs or add new datasets without starting over.

---

## 📈 3. Generating Reports & Visualizations

One of the most powerful features of Tabular-Gauge is the ability to transform raw CSV results into a publication-ready report automatically.

### 📂 Result Namespacing & Storage
To prevent overwriting previous runs, Tabular-Gauge uses **Experiment Profiles**. Your results are stored in a folder named after your experiment (defined in `configs/experiment.yaml`).

**How to change your experiment name:**
Open `configs/experiment.yaml` and update the `name` field:
```yaml
experiment:
  name: "my_custom_run"
  ...
```

**Where to find your output:**

#### 1. Raw Data (The Foundation)
- **Detailed Logs**: `experiments/results/raw/{exp_name}/`
  - Contains `.cache.json` files for every dataset. These are the most granular results.

#### 2. Aggregated Results (The Master Table)
- **Master CSV**: `experiments/results/aggregated/{exp_name}/results.csv`
  - A comprehensive table containing all metrics for all models, seeds, and datasets. This is the source file for all reports.

#### 3. Analytical Reports (Generated via `generate_reports.py`)
These are professional-grade summaries stored in `reports/{exp_name}/`.

| Artifact | Path | Description | Population Requirement |
|---|---|---|---|
| **Summary Report** | `summary.md` | High-level narrative identifying the "Best Model" and statistical winners. | **Minimum 1 seed** (Basic) / **3+ Seeds** (Full stats) |
| **AUC Heatmap** | `figures/auc_heatmap.png` | Matrix of ROC-AUC across datasets. | **Minimum 1 seed** |
| **Rank Charts** | `figures/average_ranks.png` | Bar chart of average model rankings. | **Minimum 1 seed** |
| **Efficiency Plots** | `figures/timing_comparison.png` & `memory_comparison.png` | Comparison of fit time and peak RAM. | **Minimum 1 seed** |
| **Win-Loss Matrix** | `figures/win_loss_matrix.png` | Head-to-head comparison of model superiority. | **Minimum 1 seed** |
| **Calibration Curves**| `figures/calibration_comparison.png` | Reliability diagrams for probability truthfulness. | **Minimum 1 seed** |
| **Significance Table**| `tables/wilcoxon_significance.csv` | p-values and Cliff's $\delta$ for model gaps. | **Minimum 3 seeds** |
| **Confidence Int.** | `tables/confidence_intervals.csv` | 95% CI for headline metrics. | **Minimum 2 seeds** |
| **Average Ranks** | `tables/average_ranks.csv` | The raw ranking table used for the rank charts. | **Minimum 1 seed** |

**⚠️ Critical Note on Statistical Rigor:** 
While a single seed is enough to generate basic figures, statistical evidence requires variance. If you run with `--n_seeds 1`, the **Significance Table** and **Confidence Intervals** will be empty or contain NaNs. For publication-grade results, we strongly recommend $\ge 5$ seeds.

**💡 Pro Tip on Defaults:**
By default, if you run `python experiments/run_benchmark.py --all`, the system uses the seeds and models defined in `configs/experiment.yaml` and `configs/models.yaml`. You can easily override these using the CLI flags (e.g., `--n_seeds 10`) as described in Section 5.1.

---

### How to Generate the Full Report
If you have already run the benchmark, you can regenerate all figures and the summary report without re-training the models:
```bash
python experiments/generate_reports.py
```

**Advanced: Generating Reports for Specific Metrics**
By default, reports are generated based on `roc_auc`. You can override this to analyze the benchmark from the perspective of a different metric (e.g., PR-AUC):
```bash
python experiments/generate_reports.py --metric avg_precision
```
Commonly used metrics: `roc_auc`, `avg_precision`, `f1_macro`, `balanced_accuracy`, `accuracy`.

**Report-Generation Flags:**
- `--figures-only`: Skips writing `summary.md`, only updates the PNG figures.
- `--summary-only`: Skips updating figures, only regenerates the markdown text.
- `--csv [path]`: Generate a report from a specific aggregated results CSV.

---

## 🔬 4. Advanced Analysis: Data-Efficiency & Scaling

Beyond the standard holistic benchmark, Tabular-Gauge supports a specialized **Data-Efficiency & Scaling Analysis** suite. 

This module allows you to analyze how models behave as the amount of training data changes—essentially plotting the **"Learning Curves"** of different architectures. This is critical for understanding:
- **Model Convergence**: How many samples does a model need to reach peak performance?
- **Data Efficiency**: Which models dominate when data is scarce (the "Low-Data Regime")?
- **Computational Scaling**: How do time and memory costs scale as data grows?

### How to Run Scaling Analysis

1. **Execute the few-shot benchmark**:
   Choose your datasets and the specific training sizes you want to test.
   ```bash
   python experiments/run_few_shot.py --datasets smoke_test --train_sizes 25 50 100 full --n_seeds 3
   ```
   *Note: If a requested size exceeds the total available samples in a dataset, the tool automatically uses the "full" training set.*

2. **Generate the scaling reports**:
   Once the run is complete, transform the raw results into a visual report:
   ```bash
   python experiments/generate_few_shot_reports.py --csv experiments/results/few_shot/my_test/all_few_shot_results.csv
   ```
   *Replace `my_test` with your current experiment name from `configs/experiment.yaml`.*

3. **Review the Output**:
   Reports are saved to `reports/few_shot/{exp_name}/` and include Learning Curves, Efficiency Scaling plots, and Stability Heatmaps.

👉 **For an in-depth guide on interpreting these results, configuring custom scaling scenarios, and understanding the "Low-Data Winner" vs. "Asymptotic Winner" logic, refer to the [FEW_SHOT_GUIDE.md](./FEW_SHOT_GUIDE.md).**

---

## ⚙️ 5. Customizing & Extending Your Evaluation

### 🎯 5.1 Selective Testing & CLI Options
You can control exactly what runs using CLI flags in `experiments/run_benchmark.py`:

**Defaults:**
- **Seeds**: `5` random seeds per model/dataset.
- **Models**: All enabled models in `configs/models.yaml`.

**CLI Examples:**
- **Select Specific Datasets**: `python experiments/run_benchmark.py --datasets smoke_test --force`
- **Filter Models**: `python experiments/run_benchmark.py --all --skip_models mlp_sklearn lightgbm`
- **Control Robustness**: `python experiments/run_benchmark.py --all --n_seeds 10`
- **Fresh Start**: `python experiments/run_benchmark.py --all --force`

### 📊 5.2 Adding New Datasets
You can add your own data via `configs/datasets.yaml`.

**Example: Adding a Local CSV file**
1. Place your CSV in the `data/` folder (e.g., `data/heart_disease.csv`).
2. Add an entry to `configs/datasets.yaml`:
   ```yaml
   heart_disease:
     source: csv
     path: "data/heart_disease.csv"
     target: "target_label"
     description: "Heart disease classification dataset"
   ```
3. Run: `python experiments/run_benchmark.py --datasets heart_disease`.

**Example: Adding an OpenML dataset**
1. Find a dataset ID on OpenML.org.
2. Add an entry to `configs/datasets.yaml`:
   ```yaml
   adult_census:
     source: openml
     id: 31
     name: "Adult"
   ```
3. Run: `python experiments/run_benchmark.py --datasets adult_census`.

### 🤖 5.3 Adding New Models
To integrate a new model, follow these three steps:
1. **Create a Wrapper**: Create a file in `tabular_gauge/models/` inheriting from `ModelWrapper`. Implement `fit()` and `predict_proba()`.
2. **Register the Model**: Add your model to the `_REGISTRY` in `tabular_gauge/models/factory.py`.
3. **Enable in Config**: Add the model to `configs/models.yaml` with `enabled: true`.

---

## 🤖 6. Configuration & Hyperparameters

### Default Models
By default, Tabular-Gauge runs a comprehensive suite of foundation and baseline models:
- **TabPFN 3.5 (Latest)**: State-of-the-art Tabular Foundation Model.
- **MLP**: Multi-Layer Perceptron (Neural Network).
- **XGBoost**: Gradient Boosted Decision Trees.
- **LightGBM**: Light Gradient Boosting Machine.
- **CatBoost**: Categorical Boosting.

### Model Settings (`configs/models.yaml`)
- **Hyperparameter Tuning**: Conventional models are optimized via **Optuna**. 
- **Customizing Trials**: Default is **10 trials**. Increase `n_trials` in `configs/models.yaml` for better optimization.
- **MLP Pipeline**: Automatically includes a `StandardScaler`.

### Dataset Settings (`configs/datasets.yaml`)
Add your own CSV by specifying `source: csv`, `path`, and `target`.

---

## 🔬 7. Statistical Rigor & Trust

We do not rely on a single number. To ensure the results are not biased, Tabular-Gauge employs a rigorous statistical layer:
- **Paired-by-Seed Testing**: We use the **Wilcoxon signed-rank test** paired by seed to cancel out dataset variance.
- **Family-Wise Error Control**: We apply the **Holm-Bonferroni correction** to prevent false-positive significance claims.
- **Effect Size**: We report **Cliff's $\delta$** to quantify the magnitude of the difference between models.

**For a deep dive into the math, refer to [METRICS_DOCUMENTATION.md](./METRICS_DOCUMENTATION.md).**

---

## ⚠️ 8. Limitations & Future Scope
- **Current**: Focused on **Binary Classification**.
- **Next**: Expansion to **Regression** and **Multi-class** classification.
- **Hardware**: Operational metrics (time/memory) are hardware-dependent; use relative ratios for your own hardware.

---

## 🤝 Contributing
If you implement a new model or improve a metric, please open an issue or submit a PR. Refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the full developer guide.
