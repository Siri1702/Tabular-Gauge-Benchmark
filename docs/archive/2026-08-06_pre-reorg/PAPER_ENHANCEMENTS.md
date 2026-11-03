# Research Paper Enhancement Plan

> **Document Purpose**: Track improvements and enhancements for the Tabular-Gauge Benchmark research paper  
> **Last Updated**: July 2026

---

## Completed Work ✅

### Main Benchmark
- [x] 15 OpenML datasets
- [x] 5 models (TabPFN, XGBoost, LightGBM, CatBoost, MLP)
- [x] SAP-RPT-1 main benchmark (full data, 15 datasets)
- [x] Statistical tests (Friedman, Wilcoxon, Nemenyi)
- [x] Publication-ready visualizations

### Few-Shot Analysis (Existing)
- [x] Training sizes: 25, 50, 100, 200, 500, full
- [x] 6 datasets, 5 models
- [x] Single seed analysis

### Few-Shot Analysis (Completed)
- [x] Training sizes: 25, 50, 100, 200, 300, 500, 750, 1000
- [x] 6 datasets, 5 models
- [x] Extended learning curves generated
- [x] Publication-ready visualizations created
- [x] Report updated with latest insights

---

## Completed Enhancements ✅

### Few-Shot Extended Analysis (COMPLETED)
- Added training sizes: 300, 500, 750, 1000
- Generated 4 publication-ready figures:
  - Learning curves with error bars
  - Crossover analysis (TabPFN vs XGBoost)
  - Performance heatmap
  - Timing comparison
- Updated comprehensive analysis report

### Key Findings from Extended Analysis:
- TabPFN wins at ALL 8 training sizes tested!
- Maximum advantage: +11.21% at N=25
- Average advantage: +2.87% across all sizes
- 100% win rate at N=50, 100, 200, 300
- Average rank: TabPFN = 1.29 (best)

---

## Dataset Limitation Analysis

### Current Issue Identified

The extended few-shot analysis has **incomplete data at larger training sizes** due to dataset size limitations:

| Dataset | Total Samples | Max Evaluable N |
|---------|--------------|------------------|
| australian | 6,702 | 1,000+ |
| banknote-auth | 1,097 | 1,000 |
| titanic | 1,047 | 1,000 |
| credit-g | 800 | 800 |
| diabetes | 614 | 614 |
| wdbc | 455 | 455 |

### Evaluation Completeness

| Training Size | Datasets | Completeness | Status |
|--------------|----------|--------------|--------|
| N=25 to N=300 | 6/6 | 100% | Complete |
| N=500 | 5/6 | 83% | Partial (missing wdbc) |
| N=750 | 4/6 | 67% | Partial (missing diabetes, wdbc) |
| N=1000 | 3/6 | 50% | Limited (missing diabetes, credit-g, wdbc) |

### Impact on Results

- **N=25 to N=300**: Full analysis - reliable results
- **N=500**: Mostly complete - some bias toward larger datasets
- **N=750-1000**: Limited - results may be skewed toward australian, titanic, banknote-auth

---

## Priority 1: Statistical Robustness (HIGH ROI - Recommended First)
**Effort: 2-3 hours | Impact: Very High**

| Enhancement | Description | Benefit | Effort | Priority |
|------------|-------------|---------|--------|----------|
| **Multi-seed analysis (3 seeds)** | Run with 3 seeds instead of 1 | Error bars, confidence intervals | 2-3 hours | HIGH |
| **Statistical tests** | Wilcoxon signed-rank test comparing TabPFN vs XGBoost at each N | Prove statistical significance | 1-2 hours | HIGH |
| **Bootstrap CIs** | 95% CI for each model at each training size | Visual confidence bounds | 1-2 hours | MEDIUM |

**Why**: Reviewers always question "is this significant?" - this addresses it directly.

---

### Priority 2: Dataset Augmentation (CRITICAL for Publication)
**Effort: 4-6 hours | Impact: High**

Add datasets with 1,000+ samples to complete analysis at larger training sizes:

| Dataset | OpenML ID | Samples | Domain | Priority |
|---------|-----------|---------|--------|----------|
| **Adult (Census)** | 1590 | ~49K | Demographics | CRITICAL |
| **Mushroom** | 24 | 8,124 | Biology | HIGH |
| **Default Credit** | 1597 | 30,000 | Finance | HIGH |
| **Airlines** | 41163 | 539K | Transport | MEDIUM |

**Target**: At least 4 additional datasets with 1,000+ samples

---

### Priority 3: Enhanced Analysis & Visualizations (MEDIUM ROI)
**Effort: 3-4 hours | Impact: Medium**

| Enhancement | Description | Benefit | Effort | Priority |
|------------|-------------|---------|--------|----------|
| **Critical Difference Diagram** | Nemenyi test visualization | Standard ML benchmark viz | 2 hours | MEDIUM |
| **Win/Loss Matrix** | Per-dataset who beats whom | Granular insights | 1 hour | MEDIUM |
| **Per-dataset breakdown** | Which datasets favor which models | Domain insights | 2 hours | MEDIUM |

---

## Recommended Implementation Order

### Phase 1: Dataset Augmentation (CRITICAL)
**Do this first to ensure complete analysis**

1. **Add 4 new datasets with 1,000+ samples**:
   - Adult (Census Income) - ~49K samples
   - Mushroom - ~8K samples  
   - Default Credit Card - ~30K samples
   - Airlines - ~50K+ samples

2. **Rerun extended analysis** with all 10 datasets

3. **Verify N=1000 now has 10/10 datasets**

### Phase 2: Statistical Validation
4. **Multi-seed analysis** - Add 3 seeds for confidence intervals
5. **Statistical significance tests** - p-values at each training size

### Phase 3: Visualization
6. **Learning curves with confidence bands**
7. **Critical Difference diagram**
8. **Win/loss matrix**

---

## Detailed Step-by-Step Guide

### Step 1: Add New Datasets

```python
# In configs/datasets_few_shot.yaml, add:
new_datasets:
  - name: adult
    openml_id: 1590
    target: class
    min_samples: 1000  # Must have at least 1000 samples
    
  - name: mushroom
    openml_id: 24
    target: class
    min_samples: 1000
    
  - name: default_credit
    openml_id: 1597
    target: default
    min_samples: 1000
    
  - name: airlines
    openml_id: 41163
    target: Class
    min_samples: 1000
```

### Step 2: Run Extended Analysis

```bash
# Run with new datasets
python scripts/run_extended_few_shot.py --datasets 10 --sizes all

# Verify completeness
python -c "import pandas as pd; df = pd.read_csv('results.csv'); print(df.groupby('n_train').size())"
```

### Step 3: Statistical Analysis

```bash
# Run with multiple seeds
python scripts/run_extended_few_shot.py --seeds 3 --datasets 10

# Compute statistics
python scripts/compute_statistics.py
```

### Step 4: Generate Publication Figures

```bash
# Generate all figures
python scripts/generate_extended_figures.py --ci --seeds 3
```

---

## ROI Analysis

| Enhancement | Effort (hrs) | Impact | ROI |
|------------|--------------|--------|-----|
| Add 4 datasets | 4-6 | High | ⭐⭐⭐⭐⭐ |
| Multi-seed + Stats | 3-4 | Very High | ⭐⭐⭐⭐⭐ |
| CD Diagram | 2 | Medium | ⭐⭐⭐⭐ |
| Win/Loss Matrix | 1 | Medium | ⭐⭐⭐ |

---

### Priority 2: Enhanced Analysis

| Enhancement | Description | Benefit | Effort |
|------------|-------------|---------|--------|
| **Timing curves** | Measure time at each training size | Practical guidance | 1-2 hours |
| **Calibration curves** | ECE at each training size | Complete picture | 1-2 hours |
| **Per-dataset analysis** | Show which datasets favor which models | Insights | 2-3 hours |

---

### Priority 3: Additional Datasets

| Dataset | OpenML ID | N | Domain | Priority |
|---------|-----------|---|--------|----------|
| adult | 1590 | ~49K | Demographics | Medium |
| airlines | 41163 | ~63K | Transport | Low |
| mushroom | 24 | 8K | Biology | Low |

---

## Publication-Quality Figures to Generate

After completing enhancements, generate these figures:

| Figure | Description | Format |
|--------|-------------|--------|
| **Learning curves with error bars** | All 5 models, 10 training sizes, std shown | PNG + PDF |
| **AUC heatmap** | Models × Datasets | PNG + PDF |
| **Critical Difference diagram** | Nemenyi test visualization | PNG + PDF |
| **Timing comparison** | Bar chart with log scale | PNG + PDF |
| **Decision flowchart** | When to use which model | PNG + PDF |

---

## Paper Writing Checklist

### Before Writing
- [ ] Complete all few-shot training sizes (25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, full)
- [ ] Generate publication-quality figures
- [ ] Verify all statistical tests
- [ ] Check reproducibility (document seeds, versions)

### Writing Sections
- [ ] Abstract (250 words)
- [ ] Introduction (1-1.5 pages)
- [ ] Related Work (0.5-1 page)
- [ ] Experiments (1.5-2 pages)
- [ ] Results (2 pages)
- [ ] Discussion (1 page)
- [ ] Conclusion (0.5 page)

### Submission Prep
- [ ] Format to venue template
- [ ] Check page limits
- [ ] Prepare supplementary materials
- [ ] Get professor feedback

---

## Key Statistics to Cite (Updated)

| Statistic | Value | For Context |
|-----------|-------|-------------|
| TabPFN Mean AUC | 0.932 | Best performer |
| TabPFN vs XGBoost | +4.4% | Better accuracy |
| TabPFN Speedup | 12× | Much faster |
| Few-shot N=25 advantage | +11.21% | Key finding |
| TabPFN wins at ALL sizes | N=25 to 1000 | Dominant |
| Average advantage | +2.87% | Across all sizes |
| Win rate (N=50-300) | 100% | Strong dominance |
| Average rank | 1.29 | Best performer |

---

## SAP-RPT-1 Status

| Component | Status | Notes |
|-----------|--------|-------|
| Main benchmark | ✅ Complete | 15 datasets, full data |
| Few-shot | ❌ Blocked | Requires HuggingFace auth |

**For Paper**: Include SAP-RPT-1 in main benchmark results. Note few-shot analysis pending in limitations.

---

## Future Work (After First Paper)

1. **SAP-RPT-1 few-shot analysis** (after getting access)
2. **Regression tasks** (extend beyond binary classification)
3. **More datasets** (adult, airlines)
4. **Ensemble methods** (combine TabPFN + GBDT)
5. **Feature engineering analysis** (why TabPFN excels)

---

## Timeline Suggestion

| Week | Phase | Tasks |
|------|-------|-------|
| Week 1 | Complete Data | Run 300/750/1000, add multi-seed |
| Week 2 | Analysis | Statistical tests, error bars |
| Week 3 | Figures | Publication-quality visualizations |
| Week 4-5 | Writing | Draft all paper sections |
| Week 6 | Revision | Get feedback, finalize |
| Week 7 | Submit | Submit to target venue |

---

*This document tracks enhancements for the Tabular-Gauge Benchmark research paper.*
