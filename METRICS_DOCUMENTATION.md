# 📖 Metrics & Evaluation Documentation — Tabular-Gauge

This guide is the single, authoritative reference for every metric, statistical test, and figure used in the **Tabular-Gauge** evaluation suite. It is designed to ensure that any user, reviewer, or researcher can interpret the results with complete transparency.

---

## 🛠️ 1. The Evaluation Philosophy: A Holistic View

Most tabular benchmarks suffer from "Metric Tunnel Vision," focusing only on predictive accuracy (e.g., ROC-AUC). Tabular-Gauge addresses this by evaluating models across **three orthogonal dimensions**. 

A model that wins on AUC but fails on calibration or latency is often a liability in production. We provide a "Full Utility" picture to help practitioners decide: *"Is a Foundation Model's zero-shot speed worth a slight drop in calibration compared to a tuned GBDT?"*

### 📊 The Evidence Layers

| Layer | What it answers | Key Metrics | Where to find it |
|---|---|---|---|
| **Discrimination** | Does the model rank positives above negatives? | ROC-AUC, PR-AUC, F1-Macro | `results.csv` $\rightarrow$ `roc_auc`, `avg_precision` |
| **Calibration** | Do the predicted probabilities reflect real-world frequency? | ECE, MCE, Brier Score | `results.csv` $\rightarrow$ `ece_10`, `max_calibration_error` |
| **Efficiency** | Can I afford to run this in production? | Wall-time, Peak RAM, Throughput | `results.csv` $\rightarrow$ `total_wall_time_sec`, `peak_memory_mb` |
| **Significance** | Is the gap real or just "seed luck"? | Paired Wilcoxon, Cliff's $\delta$ | `paired_stats.csv` |

---

## 📈 2. Performance Metrics (Discrimination)

### 2.1 ROC-AUC (The Headline Metric)
**What is it?** The probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample.
**Why it matters**: It is **threshold-free**. It measures the model's ability to separate classes regardless of the decision cut-off you choose.
**Interpretation**:
- `0.9 - 1.0`: Excellent (Deploy-ready).
- `0.8 - 0.9`: Good (Useful, but may need calibration).
- `0.5`: Random guessing.

### 2.2 PR-AUC / Average Precision
**What is it?** The area under the Precision-Recall curve.
**Why it matters**: ROC-AUC can be overly optimistic on highly imbalanced data. PR-AUC is more sensitive to the positive class and is critical for fraud or rare-disease detection.

### 2.3 F1-Macro & Balanced Accuracy
**What is it?** The harmonic mean of precision/recall and the average of sensitivity/specificity.
**Why it matters**: These are **threshold-dependent** (usually at 0.5). They answer: *"If I deploy this model as-is, how will it perform?"*

---

## ⚖️ 3. Calibration Metrics (Truthfulness)

A model can have a perfect ROC-AUC but be "confidently wrong." Calibration tells us if the probability $p=0.8$ actually means an 80% chance of success.

### 3.1 Expected Calibration Error (ECE)
**What is it?** The weighted average difference between predicted confidence and empirical accuracy across bins.
**Why it matters**: High ECE means the model is overconfident or underconfident. 
**Interpretation**:
- `< 0.05`: Well-calibrated.
- `0.05 - 0.10`: Acceptable.
- `> 0.15`: Poor; probabilities are unreliable.

### 3.2 Maximum Calibration Error (MCE)
**What is it?** The worst-case error in any single confidence bin.
**Why it matters**: In high-stakes fields (e.g., medical), one "danger zone" of miscalibration is more dangerous than a slightly higher average ECE.

---

## ⏱️ 4. Operational & Efficiency Metrics

These metrics capture the "Hidden Cost" of ML.

| Metric | Description | Practical Impact |
|---|---|---|
| `tuning_time_sec` | Time spent in Optuna search | Zero for Foundation Models; high for GBDTs. |
| `fit_time_sec` | Time to train the model | One-time deployment cost. |
| `predict_time_sec` | Inference time per sample | Determines real-time latency. |
| `peak_memory_mb` | Max RAM usage | Determines hardware requirements (Edge vs Cloud). |
| `throughput` | Rows processed per second | Determines the maximum scale of the system. |

---

## 🔬 5. Statistical Rigor

To prevent "p-hacking" or seed-based anomalies, we use a rigorous two-layer test:

1.  **Paired-by-Seed Wilcoxon Test**: We pair every model's performance by the random seed. This cancels out dataset-level variance and focuses on the model's inherent advantage.
2.  **Holm-Bonferroni Correction**: We correct p-values for multiple comparisons to avoid false positives.
3.  **Cliff's $\delta$ (Effect Size)**: While p-values tell us *if* a difference exists, $\delta$ tells us **how big** it is (Negligible $\rightarrow$ Small $\rightarrow$ Medium $\rightarrow$ Large).

---

## ⚠️ 6. Important Hardware Note for Users

**Operational metrics (Time and Memory) are hardware-dependent.**
The numbers reported in the benchmark were generated on a specific hardware configuration. When running your own evaluations:
- **Memory**: Your `peak_memory_mb` may vary based on your OS, background processes, and RAM architecture.
- **Latency**: `predict_time_sec` will differ significantly between CPU and GPU (CUDA) executions.
- **Throughput**: Throughput is highly sensitive to CPU clock speed and threading.

**Recommendation**: When comparing models on your own hardware, always use **relative ratios** (e.g., *"Model A is 5x faster than Model B"*) rather than absolute seconds, as ratios remain consistent across similar hardware.

---

## 🏁 7. Summary Decision Matrix

| If your priority is... | Recommended Model Class | Key Metric to Watch |
|---|---|---|
| **Rapid Prototyping** | Foundation Model (TabPFN/SAP-RPT) | `total_wall_time_sec` |
| **Absolute Max AUC** | Tuned GBDT (XGB/LGBM/Cat) | `roc_auc` |
| **Reliable Probabilities** | Tuned GBDT / SAP-RPT | `ece_10` / `mce` |
| **High-Traffic Inference** | Tuned GBDT / MLP | `throughput` / `predict_time_sec` |
| **Small Data (N < 1000)** | Foundation Model | `roc_auc` (Few-Shot) |
