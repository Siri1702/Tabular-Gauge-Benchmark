# 🛠️ Tabular-Gauge: Comprehensive Evaluation Suite for Tabular Foundation Models

Tabular-Gauge is a professional-grade toolkit designed to evaluate the "Full Utility" of tabular machine learning models. Unlike traditional benchmarks that focus solely on predictive performance, Tabular-Gauge provides a holistic view across three critical dimensions: **Predictive Performance**, **Probability Calibration**, and **Operational Efficiency**.

This repository serves as both a **practitioner's toolkit** and a **transparent research artifact** for the accompanying paper: *"Tabular Foundation Models: A Comprehensive Benchmark on Few-Shot Learning"*.

---

## 🎯 Quick Start

### 1. Installation
```bash
git clone https://github.com/Siri1702/Tabular-Gauge-Benchmark.git
cd Tabular-Gauge-Benchmark
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\Activate.ps1 # Windows

# Install based on your needs:
pip install -e .                       # Core Framework
pip install -e ".[baselines]"          # XGBoost, LightGBM, CatBoost
pip install -e ".[foundation]"        # TabPFN
pip install -e ".[baselines,foundation]" # Full Suite
```

### 2. Run Your First Benchmark
```bash
python experiments/run_benchmark.py --all
```

---

## 🗺️ Navigation Hub

Depending on your goal, please refer to the following entry points:

### 🚀 For Practitioners (Using the Toolkit)
If you want to run benchmarks on your own data or extend the model library, start here:
👉 **[`USER_GUIDE.md`](USER_GUIDE.md)**
*Includes: Setup, Custom Dataset Loading, and Result Namespacing instructions.*

### 🔬 For Reviewers (Validating the Research)
If you are reviewing the research paper and wish to verify the results, claims, and evidence:
👉 **[`docs/research/RESULTS_INDEX.md`](docs/research/RESULTS_INDEX.md)**
*Includes: The traceability matrix mapping paper tables/figures to the authoritative data in `experiments/paper_evidence/`.*

### 📚 For General Documentation
- **[`INDEX.md`](INDEX.md)**: The master documentation map.
- **[`METRICS_DOCUMENTATION.md`](METRICS_DOCUMENTATION.md)**: Detailed definitions of all evaluated metrics.
- **[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**: Detailed environment and protocol specifications.

---

## 📂 Repository Structure

```
tabular-gauge-benchmark/
├── tabular_gauge/         ← Core engine (Wrappers, Metrics, Viz)
├── experiments/           ← Benchmark runners & research evidence
│   ├── paper_evidence/    ← 🔒 Authoritative research artifacts
│   └── results/           ← 📁 User-generated namespaced results
├── configs/               ← Central YAML configurations
├── docs/                  ← Research methodology & traceability
├── reports/               ← Publication-ready tables & figures
└── scripts/               ← Specialized analysis scripts
```

---

## 📜 License & Citation
Refer to `CITATION.cff` for the correct way to cite this work.
