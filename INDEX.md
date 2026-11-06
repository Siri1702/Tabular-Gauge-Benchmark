# Tabular-Gauge-Benchmark — Documentation Index

> **Reading order:** Read this file first. Follow the pointer that matches your goal; skip the others.

---

## 🌟 What is Tabular-Gauge?

Tabular-Gauge is a professional-grade, open-source evaluation suite for tabular machine learning models. It moves beyond simple accuracy leaderboards by evaluating models across a **tripod of critical dimensions**:

1.  **Predictive Performance**: Rigorous evaluation of accuracy and discriminative power.
2.  **Probability Calibration**: Assessing the trust-worthiness of predicted probabilities (ECE/MCE).
3.  **Operational Efficiency**: Profiling the real-world cost in terms of time and memory.

Additionally, the toolkit features a specialized **Few-Shot Scaling Analysis** suite to visualize "Learning Curves," revealing how models converge as training data volume increases.

---

## 🎯 Pointer by Goal

| If you want to … | Open this |
|---|---|
| **Run the benchmark** (Install, configure, and generate your own results) | [`USER_GUIDE.md`](USER_GUIDE.md) |
| **Understand the Metrics** (Definitions of ROC-AUC, ECE, log-loss, etc.) | [`METRICS_DOCUMENTATION.md`](METRICS_DOCUMENTATION.md) |
| **Contribute to the project** (Add a new dataset or model wrapper) | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| **Check the project history** | [`CHANGELOG.md`](CHANGELOG.md) |

---

## 📂 Repository Layout (Canonical)

```
tabular-gauge-benchmark/
├── INDEX.md                              ← You are here
├── USER_GUIDE.md                         ← Setup & Execution guide (Practitioner entry point)
├── METRICS_DOCUMENTATION.md              ← Canonical metric definitions
├── CONTRIBUTING.md                       ← Guide for adding models/datasets
├── pyproject.toml                        ← Dependency & Package management
│
├── tabular_gauge/                        ← Core engine (Wrappers, Metrics, Viz)
├── experiments/                          ← Benchmark runners
│   ├── run_benchmark.py                   ← Holistic evaluation runner
│   ├── run_few_shot.py                    ← Data-efficiency analysis runner
│   ├── generate_reports.py               ← Main report orchestrator
│   └── generate_few_shot_reports.py       ← Few-shot report orchestrator
│
├── configs/                              ← Central YAML configurations
│   ├── datasets.yaml                     ← Dataset registry
│   ├── models.yaml                       ← Model registry & tuning params
│   └── experiment.yaml                   ← Global run settings
│
├── data/                                 ← Sample datasets (e.g., smoke_test.csv)
└── reports/                              ← 📈 Generated publication-ready figures
```

---

## 🏁 Where to start

**If you are a Practitioner:** Start at [`USER_GUIDE.md`](USER_GUIDE.md) to set up the toolkit, configure your API keys, and run your first benchmark.
