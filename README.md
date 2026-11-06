# 🛠️ Tabular-Gauge: Comprehensive Evaluation Suite for Tabular Models

Tabular-Gauge is a professional-grade, open-source toolkit designed to evaluate the **"Full Utility"** of tabular machine learning models. While most benchmarks focus exclusively on a single predictive metric (like ROC-AUC), Tabular-Gauge recognizes that in real-world deployment, a model's value is defined by a tripod of critical dimensions: **Predictive Performance**, **Probability Calibration**, and **Operational Efficiency**.

Whether you are evaluating a traditional GBDT (XGBoost, LightGBM, CatBoost), a Neural Network (MLP), or a Tabular Foundation Model (TabPFN), Tabular-Gauge provides the rigorous framework needed to determine which model is truly the most "useful" for your specific data.

---

## 🚀 Why Tabular-Gauge?

### The "Performance-Only" Trap
Most practitioners choose models based on a single leaderboard score. However, a model with a slightly higher AUC can be a liability if:
- **It is overconfident**: Poorly calibrated probabilities lead to risky decision-making.
- **It is computationally prohibitive**: Extreme training or inference latency can make a model unusable in production.
- **It is data-hungry**: A model that requires 10k samples to beat a baseline by 0.1% is often less valuable than a foundation model that performs similarly with only 50 samples.

### Our Unique Value Propositions (USPs)
1. **Multidimensional Evaluation**: We move beyond accuracy. We measure **calibration error (ECE/MCE)** to assess trust and **resource profiling (RAM/Time)** to assess cost.
2. **Data-Efficiency & Scaling Analysis**: Tabular-Gauge includes a specialized **Few-Shot Analysis** suite. It generates "Learning Curves" that reveal exactly how a model's performance scales as training data increases—essential for understanding the "foundation" capabilities of a model.
3. **Production-Ready Baselines**: Integrated, tuned wrappers for the industry standards (XGBoost, LightGBM, CatBoost, MLP) ensuring an apples-to-apples comparison.
4. **Zero-Config Reporting**: Transform raw CSV results into publication-quality figures and statistical tables automatically.

---

## 🎯 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Siri1702/Tabular-Gauge-Benchmark.git
cd Tabular-Gauge-Benchmark

# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\Activate.ps1 # Windows

# Install based on your needs:
pip install -e .                       # Core Framework
pip install -e ".[baselines]"          # XGBoost, LightGBM, CatBoost
pip install -e ".[foundation]"        # TabPFN 3.5
pip install -e ".[baselines,foundation]" # Full Suite
```

### 2. Run Your First Benchmark
Run the suite on all built-in datasets to see the pipeline in action. We recommend the `--force` flag for your first run to ensure fresh computation:
```bash
python experiments/run_benchmark.py --all --force
```

### 3. Analyze Data Efficiency (Few-Shot)
To see how models scale with increasing data (e.g., from 25 to 500 samples):
```bash
python experiments/run_few_shot.py --datasets smoke_test --train_sizes 25 50 100 500
python experiments/generate_few_shot_reports.py --csv experiments/results/few_shot/my_test/all_few_shot_results.csv
```

---

## 🗺️ Navigation Hub

Depending on your goal, please refer to the following entry points:

### 📚 Documentation & Guides
Depending on your goal, please refer to the following entry points:

**For Practitioners (Getting Started)**
- 👉 **[`USER_GUIDE.md`](USER_GUIDE.md)**: The definitive guide for installation, API configuration (TabPFN), custom dataset loading, and running benchmarks.
- 👉 **[`FEW_SHOT_GUIDE.md`](FEW_SHOT_GUIDE.md)**: Specialized instructions for analyzing data efficiency and generating learning curves.

**For Technical Deep-Dives**
- **[`INDEX.md`](INDEX.md)**: The master documentation map for the entire repository.
- **[`METRICS_DOCUMENTATION.md`](METRICS_DOCUMENTATION.md)**: Rigorous mathematical definitions and interpretation guides for every metric used.

---

## 📂 Repository Structure

```
tabular-gauge-benchmark/
├── tabular_gauge/         ← Core engine (Wrappers, Metrics, Viz)
├── experiments/           ← Benchmark runners (run_benchmark, run_few_shot, etc.)
├── configs/               ← Central YAML configurations (Models, Datasets)
├── data/                  ← Sample datasets (e.g., smoke_test.csv)
├── reports/               ← 📈 Generated publication-ready figures
├── pyproject.toml         ← Project metadata and dependencies
├── README.md              ← Main Landing Page
├── USER_GUIDE.md          ← Setup & Usage Guide
├── FEW_SHOT_GUIDE.md      ← Data-Efficiency Analysis Guide
├── INDEX.md               ← Documentation Map
└── METRICS_DOCUMENTATION.md ← Detailed Metric Definitions
```

---

## 📜 License & Citation
Refer to `CITATION.cff` for the correct way to cite this work.
