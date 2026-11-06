# ⚙️ Configuration and Extension Guide

This guide explains how to configure Tabular-Gauge experiments and how to extend the toolkit with your own custom models.

## 1. Overview
Tabular-Gauge is designed to be configuration-driven. Most experiment settings are controlled via YAML files in the `configs/` directory, allowing you to design complex benchmarks without modifying Python code.

## 2. Quick Start: Running a Benchmark
To run a benchmark with default settings:
```bash
python experiments/run_benchmark.py --all
```

To run a specific subset of datasets:
```bash
python experiments/run_benchmark.py --datasets credit-g diabetes
```

## 3. Configuration Architecture

### A. Model Selection (`configs/models.yaml`)
You can control which models are active and how they are configured. The toolkit supports two formats:

**1. List Format (Recommended for flexibility):**
```yaml
models:
  - id: tabpfn
    enabled: true
    fixed_params:
      device: "cuda"
  - id: xgboost
    enabled: true
    n_trials: 20
  - id: lightgbm
    enabled: false # This model will be skipped
```

**2. Dictionary Format (Legacy):**
```yaml
models:
  tabpfn:
    fixed_params: {device: "cuda"}
  xgboost:
    n_trials: 20
```

### B. Dataset Configuration (`configs/datasets.yaml`)
Define the datasets you wish to evaluate.

**OpenML Datasets:**
```yaml
datasets:
  - name: australian
    id: 40536
    source: openml
    target: class
```

**Custom CSV Datasets:**
```yaml
datasets:
  - name: my_custom_data
    source: csv
    path: data/my_dataset.csv
    target: target_column
```

### C. Experiment Settings (`configs/experiment.yaml`)
Control the global execution parameters.

```yaml
experiment:
  n_seeds: 5            # Number of random seeds for stability
  test_size: 0.2        # Proportion of data for the holdout test set
  cache_results: true    # Skip already completed (model, seed, dataset) cells
  metrics:              # List of metrics to compute
    - roc_auc
    - brier_score
    - log_loss
```

---

## 4. Adding Your Own Model (Extension Framework)

Tabular-Gauge is extensible. You can integrate any model that follows the `ModelWrapper` interface.

### Step 1: Create a Model Wrapper
Create a new Python file (e.g., `tabular_gauge/models/my_model.py`) and inherit from `ModelWrapper`.

```python
from tabular_gauge.models.base import ModelWrapper
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MyCustomWrapper(ModelWrapper):
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__(name="MyCustomModel", random_state=random_state)
        self.n_estimators = n_estimators

    def fit(self, X_train, y_train):
        # Initialize and train your model
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        # Return probabilities for the positive class (shape: (n_samples,))
        return self._model.predict_proba(X_test)[:, 1]
```

### Step 2: Register the Model
You can register the model in two ways:

**Method A: Permanent Registration (via `factory.py`)**
Add your model to the `_REGISTRY` in `tabular_gauge/models/factory.py`:
```python
_REGISTRY = {
    # ... existing models ...
    "my_model": "tabular_gauge.models.my_model.MyCustomWrapper",
}
```

**Method B: Dynamic Registration (via Code)**
If you are running a custom script, you can register the model at runtime:
```python
from tabular_gauge.models.factory import ModelFactory
ModelFactory.register_model("my_model", "my_module.MyCustomWrapper")
```

### Step 3: Configure and Run
Add the model to `configs/models.yaml`:
```yaml
models:
  - id: my_model
    enabled: true
    n_estimators: 200 # This is passed to the wrapper __init__
```
Then run the benchmark normally:
```bash
python experiments/run_benchmark.py --all
```

---

## 5. Designing Your Benchmark Experiments

| Experiment | What it investigates | What you can learn |
|---|---|---|
| **Single dataset, multiple models** | Relative performance | Which model works best for this specific data |
| **Multiple datasets, multiple models** | Generalization | Whether a model's advantage is consistent across domains |
| **Few-shot training sizes** | Low-data performance | Which models learn effectively with limited examples |
| **Training-size scaling** | Data efficiency | Whether model rankings change as $N$ increases |
| **Multiple random seeds** | Stability | Whether results are robust or seed-dependent |
| **Predictive vs Calibration** | Reliability | Whether high accuracy comes with overconfidence |

### Understanding Key Metrics
- **ROC-AUC**: Ability to distinguish between classes (Higher is better).
- **Brier Score**: Mean squared error of probability predictions (Lower is better).
- **ECE (Expected Calibration Error)**: Difference between predicted confidence and actual accuracy (Lower is better).
- **Fit Time / Latency**: Computational cost of training and prediction.

---

## 6. Understanding Outputs
- **Raw Results**: Found in `experiments/results/raw/`. Detailed JSONs per dataset.
- **Aggregated Results**: Found in `experiments/results/aggregated/results.csv`. A master table of all runs.
- **Reports**: `reports/figures/` contains heatmaps, ranking plots, and calibration curves.

To regenerate reports from existing CSVs:
```bash
python experiments/generate_reports.py
```
