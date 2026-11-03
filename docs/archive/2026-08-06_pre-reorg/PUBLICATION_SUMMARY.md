# Publication Summary: Tabular-Gauge Benchmark

> **For Paper Writing** - Key statistics and results ready for publication  
> **Last Updated**: July 2026

---

## 1. Main Benchmark Results (5 Models, 15 Datasets)

### Table 1: Model Performance Summary

| Model | Mean ROC-AUC | Std Dev | Mean Time (s) | Win Rate |
|-------|-------------|----------|---------------|----------|
| **TabPFN** | **0.9320** | 0.0846 | 5.16 | 67% (10/15) |
| LightGBM | 0.8894 | 0.1134 | 213.25 | 7% (1/15) |
| XGBoost | 0.8883 | 0.1189 | 62.45 | 13% (2/15) |
| CatBoost | 0.8870 | 0.1259 | 330.56 | 13% (2/15) |
| MLP | 0.8656 | 0.1394 | 33.00 | 0% (0/15) |

**Key Finding**: TabPFN achieves **+4.4%** higher AUC than XGBoost with **12× faster** execution time.

---

## 2. Few-Shot Learning Analysis (5 Models, 6 Datasets)

### Table 2: Few-Shot Results by Training Size

| Training Size | TabPFN | XGBoost | LightGBM | CatBoost | MLP |
|--------------|---------|----------|----------|----------|-----|
| **N=25** | **0.8180** | 0.7059 | 0.6354 | 0.7955 | 0.7677 |
| **N=50** | **0.8466** | 0.8164 | 0.8314 | 0.8185 | 0.7915 |
| **N=100** | **0.8824** | 0.8546 | 0.8492 | 0.8298 | 0.8414 |
| **N=200** | **0.8972** | 0.8747 | 0.8764 | 0.8843 | 0.8601 |
| **N=500** | **0.9044** | 0.8929 | 0.9013 | 0.8949 | 0.8779 |
| N=Full | 0.9101 | **0.9112** | 0.9060 | 0.9092 | 0.9013 |

### Table 3: TabPFN vs XGBoost (Key Finding)

| Training Size | TabPFN | XGBoost | TabPFN Advantage |
|--------------|--------|----------|------------------|
| N=25 | **0.8180** | 0.7059 | **+11.21%** |
| N=50 | **0.8466** | 0.8164 | +3.02% |
| N=100 | **0.8824** | 0.8546 | +2.79% |
| N=200 | **0.8972** | 0.8747 | +2.24% |
| N=500 | **0.9044** | 0.8929 | +1.15% |
| N=Full | 0.9101 | **0.9112** | -0.11% |

**Key Finding**: TabPFN wins at ALL training sizes except full data. At N=25, TabPFN achieves **+11.21%** higher AUC than XGBoost.

---

## 3. Crossover Analysis (Adult Dataset, 48K Samples)

### Table 4: Crossover Point Results

| Training Size | TabPFN | XGBoost | LightGBM | Winner |
|--------------|---------|----------|----------|--------|
| N=50 | 0.798 | 0.705 | 0.734 | TabPFN ✅ |
| N=500 | 0.899 | 0.887 | 0.887 | TabPFN ✅ |
| N=1500 | 0.907 | 0.906 | 0.906 | Tie |
| N=3000 | 0.909 | 0.911 | 0.907 | XGBoost ✅ |
| N=Full | 0.916 | **0.930** | **0.931** | XGBoost ✅ |

**Crossover Zone**: N ≈ 1,500 - 3,000

---

## 4. Key Statistics for Paper

### Performance Metrics
- **TabPFN Mean AUC**: 0.932 (rank 1)
- **TabPFN Win Rate**: 67% (10/15 datasets)
- **TabPFN Speedup**: 12× faster than XGBoost
- **Few-Shot Advantage**: +11.21% at N=25

### Statistical Significance
- **Friedman Test**: p < 0.01 (significant differences exist)
- **Wilcoxon vs XGBoost**: p = 0.042 (TabPFN significantly better)

---

## 5. Figures to Generate

For the paper, generate these figures:

1. **Figure 1**: ROC-AUC heatmap (models × datasets)
2. **Figure 2**: Learning curves (5 models, 6 training sizes)
3. **Figure 3**: Critical Difference diagram (Nemenyi)
4. **Figure 4**: Timing comparison (bar chart)
5. **Figure 5**: Decision flowchart (when to use which model)

---

## 6. Suggested Paper Structure

### Abstract (250 words)
1. Problem: Can zero-shot tabular models compete with tuned GBDTs?
2. Method: Benchmark on 15 OpenML datasets
3. Key Finding: TabPFN wins at N < 500 with 12× speedup

### Introduction (1-1.5 pages)
- Data scarcity in real-world ML
- Rise of tabular foundation models
- Our contributions (4-5 bullet points)

### Related Work (0.5-1 page)
- TabPFN and similar models
- Traditional GBDT methods
- Few-shot learning

### Experiments (1.5-2 pages)
- Datasets table
- Models table
- Metrics
- Protocol

### Results (2 pages)
- Main benchmark results
- Few-shot analysis ⭐ (key differentiator!)
- Statistical tests
- Calibration

### Discussion (1 page)
- Practical guidelines
- Limitations
- Future work

---

## 7. Key Messages for Reviewers

### Primary Message
> "We present the first systematic few-shot learning analysis of tabular foundation models, showing that TabPFN outperforms tuned gradient boosted trees by up to 11% AUC when training data is limited to fewer than 500 samples."

### Why This Matters
1. **Practical**: Most real-world ML problems have limited labeled data
2. **Novel**: First comprehensive benchmark with statistical rigor
3. **Actionable**: Clear decision framework for practitioners

### Limitations to Acknowledge
- Single seed in some experiments
- Binary classification only
- SAP-RPT-1 few-shot results pending (gated model)

---

## 8. Target Venues (in order of fit)

1. **NeurIPS** - Top ML venue, benchmark papers well-received
2. **ICML** - Strong methodology focus
3. **KDD** - Applied data mining focus
4. **AAAI** - Broad AI, accessible

---

## Quick Reference: Numbers to Cite

| Statistic | Value | For Context |
|-----------|-------|-------------|
| TabPFN Mean AUC | 0.932 | Best performer |
| TabPFN vs XGBoost AUC | +4.4% | Better accuracy |
| TabPFN Speedup | 12× | Much faster |
| Few-shot N=25 advantage | +11.2% | Key finding |
| Crossover point | N ≈ 1,500-3,000 | Practical guidance |
| Win rate | 67% | Dominant performance |

---

*This document is ready for paper writing. All statistics are from experimental results.*
