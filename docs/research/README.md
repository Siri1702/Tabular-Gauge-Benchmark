# 🏆 Tabular-Gauge: Holistic Tabular ML Evaluation Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Tabular-Gauge** is a professional-grade, open-source framework designed to provide a 360-degree evaluation of tabular machine learning models. Moving beyond simple accuracy metrics, Tabular-Gauge enables practitioners to weigh the trade-offs between **predictive performance**, **probability calibration**, and **operational efficiency**.

## 🎯 Why Tabular-Gauge?

Traditional tabular ML benchmarks often suffer from a **"Performance-Only" bias**, focusing exclusively on a single metric like ROC-AUC. In real-world deployment, this is insufficient. A model with a slightly higher AUC is a liability if it is poorly calibrated (overconfident) or computationally prohibitive.

Tabular-Gauge solves this by tracking three critical dimensions:

| Dimension | Key Metrics | Why it Matters |
|---|---|---|
| **Performance** | ROC-AUC, PR-AUC, F1-Macro | Does the model actually work? |
| **Calibration** | ECE, MCE, Reliability Diagrams | Can you trust the probabilities for decision-making? |
| **Operational** | Fit Time, Latency, Peak RAM | Can you afford to run this in production? |

---

## 🚀 Key Features

- **Config-Driven Execution**: Enable/disable models and datasets via YAML without touching Python code.
- **Hybrid Dataset Support**: Seamlessly benchmark curated **OpenML** datasets or your own **Custom CSVs**.
- **Foundation Model vs. Baselines**: Out-of-the-box support for Tabular Foundation Models (TFMs) and tuned conventional baselines.
- **Statistical Rigor**: Built-in Friedman and Nemenyi tests for model rankings, with Bootstrap Confidence Intervals.
- **Operational Profiling**: Automated tracking of wall-clock time and peak memory consumption.
- **Automated Reporting**: Generates publication-ready figures and a comprehensive summary report.

---

## 🤖 Supported Models

Tabular-Gauge provides a fair comparison between Zero-Shot Foundation Models and optimized conventional baselines.

### Tabular Foundation Models (TFMs)
- **TabPFN / TabPFN 3.5**: High-performance in-context learners.

### Conventional ML Baselines (Tuned via Optuna)
- **XGBoost**, **LightGBM**, **CatBoost**, and **MLP**.

---

## 📦 Installation

Tabular-Gauge is distributed as a Python package. We recommend using a virtual environment.

### 1. Basic Setup
```bash
git clone https://github.com/Siri1702/Tabular-Gauge.git
cd Tabular-Gauge
python -m venv venv
source venv/bin/activate  # Linux/Mac (or venv\\Scripts\\Activate.ps1 on Windows)
pip install -e .
```

### 2. Install Model Dependencies
To avoid installing heavy libraries you don't need, dependencies are split into optional groups:

**For Conventional Baselines (XGBoost, LightGBM, etc.):**
```bash
pip install -e ".[baselines]"
```

**For Foundation Models (TabPFN, etc.):**
```bash
pip install -e ".[foundation]"
```

**For Everything (All models + Dev tools):**
```bash
pip install -e ".[baselines,foundation,dev]"
```

---

## 🛠️ Quick Start: Benchmark Your Own CSV

You can evaluate your own tabular data in minutes. For a detailed guide on configuring experiments, adding custom models, and interpreting results, see the [Configuration and Extension Guide](./docs/CONFIGURATION_AND_EXTENDING.md).

**Basic Workflow:**
1. **Prepare your data**: Place your binary classification CSV in the `data/` folder (e.g., `data/my_data.csv`).
2. **Configure the dataset**: Add an entry to `configs/datasets.yaml`:
   ```yaml
   datasets:
     - name: my_dataset
       source: csv
       path: data/my_dataset.csv
       target: target_column_name
       description: "My custom test dataset"
   ```
3. **Select your models**: Ensure the models you want to compare are listed in `configs/models.yaml`.
4. **Run the benchmark**:
   ```bash
   python experiments/run_benchmark.py --datasets my_dataset
   ```
5. **Check results**: Find your results in `experiments/results/aggregated/results.csv` and your figures in `reports/figures/`.

---

## 📊 Understanding the Outputs

Tabular-Gauge produces a comprehensive suite of artifacts:

- **`results.csv`**: The raw master table containing every metric for every model, seed, and dataset.
- **`auc_heatmap.png`**: A visual map of model performance across all datasets.
- **`average_ranks.png` & `cd_diagram.png`**: Statistical rankings of models with Critical Difference bars.
- **`calibration_comparison.png`**: Visual proof of how "truthful" the model probabilities are.
- **`timing_comparison.png` & `memory_comparison.png`**: Analysis of the operational cost (Wall-time and Peak RAM).

---

## ⚠️ Current Limitations

- **Task Scope**: Currently supports **Binary Classification** only. (Regression and Multi-class are planned).
- **Hardware**: Operational metrics (time/memory) are hardware-dependent; use relative ratios for comparisons.

## 🔬 Research Paper & Reproducibility

Tabular-Gauge serves as the official software artifact for our research on Tabular Foundation Models. We provide a dedicated hub for academic reviewers to inspect the experimental protocol and reproduce the reported results.

- **[Research Overview](./docs/research/README.md):** High-level summary of the study.
- **[Experimental Methodology](./docs/research/METHODOLOGY.md):** Detailed scientific protocol.
- **[Results Index](./docs/research/RESULTS_INDEX.md):** Mapping of paper tables/figures to code and data.
- **[Reproduction Guide](./docs/research/REPRODUCING_RESULTS.md):** Step-by-step instructions to regenerate results.
- **[Dataset & Model Inventory](./docs/research/INVENTORY.md):** Verified metadata for all experiments.
- **[Citation Information](./CITATION.cff):** How to cite the software and paper.

---

## 🤝 Contributing

We welcome contributions! Whether it's adding a new model wrapper, improving a metric, or providing a new dataset, please open an issue or submit a PR. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 📝 License
MIT License.
