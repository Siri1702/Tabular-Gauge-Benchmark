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
git clone https://github.com/Siri1702/Tabular-Gauge.git
cd Tabular-Gauge
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\Activate.ps1 # Windows
```

### Installation Options
The toolkit uses optional dependency groups so you only install what you need:

- **Core Framework**: `pip install -e .`
- **Conventional Baselines**: `pip install -e ".[baselines]"` (XGBoost, LightGBM, CatBoost)
- **Foundation Models**: `pip install -e ".[foundation]"` (TabPFN)
- **Full Suite**: `pip install -e ".[baselines,foundation]"`

### Your First Run
Run the suite on all built-in datasets to see the pipeline in action:
```bash
python experiments/run_benchmark.py --all
```

---

## 📈 3. Generating Reports & Visualizations

One of the most powerful features of Tabular-Gauge is the ability to transform raw CSV results into a publication-ready report automatically.

### 📂 Result Namespacing & Storage
To prevent overwriting research evidence, Tabular-Gauge uses **Experiment Profiles**. Your results are stored in a folder named after your experiment.

**How to change your experiment name:**
Open `configs/experiment.yaml` and update the `name` field:
```yaml
experiment:
  name: "my_custom_run"
  ...
```

**Where to find your output:**
- **Aggregated CSV**: `experiments/results/aggregated/{exp_name}/results.csv`
- **Raw Logs**: `experiments/results/raw/{exp_name}/`
- **Analytical Figures**: `reports/figures/{exp_name}/`

---

### How to Generate the Full Report
If you have already run the benchmark, you can regenerate all figures and the summary report without re-training the models:
```bash
python experiments/regenerate_report.py
```

---

## ⚙️ 4. Customizing Your Evaluation

### 🎯 Example 1: Selective Testing
If you only want to compare **TabPFN** and **XGBoost** on two specific datasets:
```bash
python experiments/run_benchmark.py --datasets credit-g diabetes --models tabpfn xgboost
```

### 🎲 Example 2: Ensuring Robustness (Random Seeds)
To avoid "seed luck", increase the number of seeds:
```bash
python experiments/run_benchmark.py --all --n_seeds 10
```

### 📉 Example 3: Comparing Training Set Sizes (Few-Shot Analysis)
To understand how a model scales as you add more data:
```bash
python experiments/run_few_shot.py --datasets diabetes wdbc --train_sizes 25 100 500 full
```

---

## 🤖 5. Configuration & Hyperparameters

### Model Settings (`configs/models.yaml`)
The ModelFactory uses this file to instantiate wrappers.
- **Zero-Shot Models**: Settings are in `fixed_params`.
- **Tuned Models**: Conventional models are optimized via **Optuna**. You can modify `n_trials` or the `search_space` (e.g., `max_depth`, `learning_rate`) directly in the YAML.

### Dataset Settings (`configs/datasets.yaml`)
To add your own CSV:
1. Add an entry:
   ```yaml
   my_custom_data:
     source: csv
     path: "data/my_dataset.csv"
     target: "label_column"
     description: "My custom classification dataset"
   ```
2. Run: `python experiments/run_benchmark.py --datasets my_custom_data`.

---

## 🔬 6. Statistical Rigor & Trust

We do not rely on a single number. To ensure the results are not biased, Tabular-Gauge employs a rigorous statistical layer:
- **Paired-by-Seed Testing**: We use the **Wilcoxon signed-rank test** paired by seed to cancel out dataset variance.
- **Family-Wise Error Control**: We apply the **Holm-Bonferroni correction** to prevent false-positive significance claims.
- **Effect Size**: We report **Cliff's $\delta$** to quantify the magnitude of the difference between models.

**For a deep dive into the math, refer to [METRICS_DOCUMENTATION.md](./METRICS_DOCUMENTATION.md).**

---

## ⚠️ 7. Limitations & Future Scope
- **Current**: Focused on **Binary Classification**.
- **Next**: Expansion to **Regression** and **Multi-class** classification.
- **Hardware**: Operational metrics (time/memory) are hardware-dependent; use relative ratios for your own hardware.

---

## 🤝 Contributing
If you implement a new model or improve a metric, please open an issue or submit a PR. Refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the full developer guide.
