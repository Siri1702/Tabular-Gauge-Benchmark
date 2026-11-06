# 📈 Data-Efficiency & Scaling Analysis Guide

Welcome to the **Data-Efficiency & Scaling Analysis** module of Tabular-Gauge. 

While the standard benchmark tells you **which** model is the best, this analysis tells you **how** they get there. It measures model performance and operational cost as a function of training data volume, revealing the "learning curves" of different architectures.

---

## 🌟 1. Why Data-Efficiency Analysis?

In real-world ML, data is often expensive or scarce. Understanding a model's scaling behavior provides critical insights:

1. **Convergence Speed**: How many samples does a model need to reach 95% of its peak performance?
2. **The "Foundation" Advantage**: Foundation models (like TabPFN) often show massive leads in the "Low-Data Regime" (e.g., < 100 samples) but may be overtaken by tuned GBMs as data increases.
3. **Stability/Variance**: Small datasets lead to high variance. We measure this to ensure a model's performance isn't just "lucky" on a specific subset of data.
4. **Computational Scaling**: Does doubling the data double the training time, or is the cost exponential?

---

## 🚀 2. Quick Start: The Two-Step Pipeline

Data-Efficiency analysis is a two-stage process: **Execution** (computing metrics) and **Reporting** (generating visuals).

### Step 1: Run the Analysis
Run the benchmark across various training sizes.
```powershell
$env:PYTHONPATH = "."; python experiments/run_few_shot.py
```

### Step 2: Generate the Professional Report
Transform the raw CSVs into publication-ready figures and narratives.
```powershell
$env:PYTHONPATH = "."; python experiments/generate_few_shot_reports.py --csv experiments/results/few_shot/[your_exp_name]/all_few_shot_results.csv
```
*(Replace `[your_exp_name]` with the name in your `configs/experiment.yaml`. Results are stored in `experiments/results/few_shot/[exp_name]/`)*

---

## ⚙️ 3. Advanced Configuration & Flexibility

You can precisely control the analysis using CLI flags in `run_few_shot.py`.

### Default Settings
If no flags are provided, the tool uses these professional defaults:
- **Training Sizes**: `[50, 100, 200, 500, 1000, None]` (where `None` is the full dataset).
- **Robustness**: `3` random seeds per size.
- **Models**: All models marked as `enabled: true` in `configs/models.yaml` (typically TabPFN 3.5, XGBoost, LightGBM, CatBoost, and MLP).

### CLI Options
| Flag | Description | Example |
| :--- | :--- | :--- |
| `--datasets` | Run analysis on specific datasets only. | `--datasets smoke_test adult` |
| `--train_sizes` | Define custom sample sizes to test. | `--train_sizes 10 50 100 500` |
| `--n_seeds` | Increase seeds for higher statistical rigor. | `--n_seeds 10` |

### Model Selection
Models are not selected via CLI to ensure consistency with the main benchmark. To change which models are run:
1. Open `configs/models.yaml`.
2. Set `enabled: true` for the models you want to include and `enabled: false` for those you wish to skip.

### Example Scenarios

**Scenario A: The "Rapid Smoke Test"**
*Goal: Quickly verify the pipeline on one dataset with 3 custom sizes and low seed count.*
```powershell
$env:PYTHONPATH = "."; python experiments/run_few_shot.py --datasets smoke_test --train_sizes 20 50 100 --n_seeds 2
```

**Scenario B: The "Publication-Grade Deep Dive"**
*Goal: High-confidence results across all datasets with a wide range of sizes.*
```powershell
$env:PYTHONPATH = "."; python experiments/run_few_shot.py --n_seeds 10 --train_sizes 50 100 200 500 1000
```

### 🧠 Smart Optimization: Automatic Pruning
The toolkit includes a **Dynamic Pruning** engine. If a dataset has only 300 samples, but you requested sizes `[50, 100, 200, 500, 1000]`, the tool automatically:
- Skips the 500 and 1000 buckets (as they are identical to the 300-sample "Full" set).
- Prevents redundant computations.
- Ensures your final tables are clean and not biased by duplicate data.

---

## 📊 4. Interpreting the Results

Your reports are saved in `reports/few_shot/[exp_name]/`.

### 🖼️ Visualizations (`/figures`)
1. **Learning Curves (`learning_curves.png`)**:
   - **What it shows**: Mean metric (Y-axis) vs. Training Size (X-axis, Log Scale).
   - **Analysis**: Look for the "elbow" of the curve. A steeper initial climb indicates higher data efficiency.
2. **Efficiency Scaling (`time_scaling.png`, `memory_scaling.png`)**:
   - **What it shows**: Fit time or Peak RAM vs. Training Size.
   - **Analysis**: Helps you decide if the performance gain of more data justifies the increased computational cost.
3. **Stability Heatmap (`stability_heatmap.png`)**:
   - **What it shows**: Standard deviation across seeds for each (model, size) pair.
   - **Analysis**: Darker/warmer cells indicate high instability. Foundation models often show lower instability in low-data regimes compared to GBMs.

### 📋 Analytical Tables (`/tables`)
- **`performance_by_size.csv`**: A matrix of `Size x Model`. Perfect for creating LaTeX tables for research papers.
- **`stability_by_size.csv`**: A matrix of `Std Dev x Model`, showing exactly where models become unreliable.

### 📝 The Summary Report (`summary.md`)
The automated narrative identifies the **"Low-Data Winner"** (best at the smallest size) vs. the **"Asymptotic Winner"** (best at the largest size), highlighting if there is a shift in leadership as more data becomes available.

---

## ⚠️ 5. Troubleshooting
- **Memory Pressure**: Few-shot analysis runs models many times. If you face crashes, reduce `--n_seeds` or `--datasets`.
- **TabPFN API**: Ensure your `.env` file is configured, or TabPFN will be skipped.
