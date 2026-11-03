# Tabular-Gauge Benchmark - Project Roadmap & Status

> 📅 Last Updated: July 19, 2026
> 🎯 Purpose: Track progress and plan next steps for the benchmark

---

## ✅ Completed Milestones

### Phase 1: Foundation
- [x] Run benchmark on 15 OpenML datasets
- [x] Add SAP-RPT-1 model (contributed by @prabhaingit)
- [x] Statistical tests (Friedman, Nemenyi, Wilcoxon)
- [x] Bootstrap confidence intervals
- [x] Calibration metrics (ECE, MCE)
- [x] Memory profiling
- [x] Updated visualizations (timing, throughput with log scale)

### Phase 2: Few-Shot Analysis ⚠️ PARTIAL
- [x] Test training sizes: N = 25, 50, 100, 200, 500, Full
- [x] 6 datasets tested
- [x] Learning curves generated
- [x] Key finding: TabPFN wins at N < 500
- [ ] **SAP-RPT-1 NOT included** - Needs to be added

### Phase 3: Large-Scale Crossover Analysis ⚠️ PARTIAL
- [x] Adult dataset (48,842 samples)
- [x] Training sizes: 50 to full
- [x] Key finding: Crossover at N ≈ 1,500-3,000
- [ ] **SAP-RPT-1 NOT included** - Needs to be added

### Phase 4: Models & Visualization
- [x] 6 models benchmarked (TabPFN, SAP-RPT-1, XGBoost, LightGBM, CatBoost, MLP)
- [x] 10+ visualizations generated
- [x] Interactive dashboard
- [x] LaTeX tables for papers

---

## 📊 Current Benchmark Status

### Models Tested (6)
| Model | Type | Mean AUC | Speed | Few-Shot | Crossover |
|-------|------|---------|-------|----------|-----------|
| **TabPFN** | Zero-shot Foundation | 0.932 | 5.2s | ✅ Done | ✅ Done |
| **SAP-RPT-1** | Zero-shot Foundation | 0.891 | 9.4s | ❌ Missing | ❌ Missing |
| **XGBoost** | Gradient Boosted | 0.888 | 62.5s | ✅ Done | ✅ Done |
| **LightGBM** | Gradient Boosted | 0.889 | 213s | ✅ Done | ✅ Done |
| **CatBoost** | Gradient Boosted | 0.887 | 331s | ✅ Done | ❌ Missing |
| **MLP** | Neural Network | 0.866 | 33s | ✅ Done | ❌ Missing |

### Datasets (15)
- Small: diabetes (768), wdbc (569), australian (690)
- Medium: credit-g (1000), titanic (891), banknote-auth (1372)
- Large: spambase (4601), adult (~49K in crossover)
- Various domains: healthcare, finance, software, chemistry

---

## 🎯 DETAILED NEXT STEPS

### 🚨 CRITICAL: SAP-RPT-1 Integration (Immediate)

**Why SAP-RPT-1 needs few-shot and crossover analysis:**

1. **Different mechanism than TabPFN**: SAP-RPT-1 uses semantic in-context learning (uses column names), TabPFN uses attention
2. **Unknown few-shot behavior**: We know TabPFN wins at N < 500, but how does SAP-RPT-1 perform?
3. **Different crossover point**: May have different crossover characteristics

#### Step 1.1: Add SAP-RPT-1 to Few-Shot Analysis

```bash
# Run few-shot analysis with SAP-RPT-1 on existing few-shot datasets
python -c "
from experiments.run_benchmark import run_single
from src.models.sap_rpt_model import SAPRptWrapper

# Add to few-shot analysis
datasets = ['diabetes', 'wdbc', 'australian', 'credit-g', 'titanic', 'banknote-auth']
train_sizes = [25, 50, 100, 200, 500, 'full']

# Run SAP-RPT-1 on each
for dataset in datasets:
    for size in train_sizes:
        # Subsample to size
        model = SAPRptWrapper(random_state=42)
        # Run and save results
"
```

**Expected Insight**: How does SAP-RPT-1 perform at N < 500? Does it beat TabPFN?

#### Step 1.2: Add SAP-RPT-1 to Crossover Analysis

```bash
# Run SAP-RPT-1 on Adult dataset with different training sizes
python experiments/run_benchmark.py --datasets adult --models sap_rpt --n_seeds 3
```

**Expected Insight**: At what N does SAP-RPT-1 crossover with XGBoost? Is it different from TabPFN?

---

### 🔴 Priority 1: Complete SAP-RPT-1 Analysis

| Task | Dataset | Training Sizes | Status | Expected Outcome |
|------|---------|----------------|--------|------------------|
| Few-shot | 6 datasets | 25, 50, 100, 200, 500, Full | Not done | May show different pattern than TabPFN |
| Crossover | Adult (48K) | 50 to Full | Not done | May crossover earlier/later than TabPFN |
| Context window | Various | Test context sizes | Not done | Optimal context for SAP-RPT-1 |

**Commands to execute:**
```bash
# 1. First, add SAP-RPT-1 to few-shot (use same datasets)
# 2. Then run crossover with SAP-RPT-1
python experiments/run_benchmark.py --datasets adult --n_seeds 3
# 3. Analyze results
```

---

### 🟠 Priority 2: Complete Crossover for All Models

| Task | Dataset | Models | Status |
|------|---------|--------|--------|
| Adult crossover | Adult (48K) | CatBoost, MLP | Not done |
| Additional crossover | spambase (4.6K) | All 6 models | Partial |
| Additional crossover | mushroom | All 6 models | Partial |

---

### 🟡 Priority 3: Context Window Analysis

**Research Question**: How does context window size affect SAP-RPT-1 performance?

| Context Size | Expected Behavior |
|--------------|-------------------|
| 512 | Fast but less accurate |
| 1024 | Baseline |
| 2048 | Better accuracy |
| 4096 | Even better |
| 8192 | Current default |

**Test this**:
```python
# Test different context sizes
for ctx_size in [512, 1024, 2048, 4096, 8192]:
    model = SAPRptWrapper(max_context_size=ctx_size)
    # Run on benchmark
```

---

### 🟢 Priority 4: Dataset Expansion

#### Small Datasets (N < 500)
| Dataset | OpenML ID | N | Domain |
|---------|--------|---|--------|
| breast | 15 | 286 | Healthcare |
| ionosphere | 59 | 351 | Physics |
| sonar | 40 | 208 | Signal |
| vote | 58 | 435 | Political |

#### Large Datasets (N > 50K)
| Dataset | OpenML ID | N | Domain |
|---------|-----------|--:|--------|
| adult | 1590 | ~49K | Demographics |
| airlines | 41163 | ~63K | Transport |
| covertype | 1597 | ~581K | Ecology |

---

### 🔵 Priority 5: Additional Foundation Models

Research adding these models:
- **FT-Transformer** (from rtdl)
- **TabR** (retrieval-augmented)
- **SAINT** (self-attention)

---

## 📋 IMPLEMENTATION SCHEDULE

### Week 1-2: SAP-RPT-1 Few-Shot
- [ ] Run SAP-RPT-1 on existing 6 few-shot datasets
- [ ] Training sizes: 25, 50, 100, 200, 500, Full
- [ ] Generate updated few-shot analysis report
- [ ] Compare with TabPFN few-shot performance

### Week 3-4: SAP-RPT-1 Crossover
- [ ] Run SAP-RPT-1 on Adult dataset
- [ ] Training sizes: 50, 100, 250, 500, 1000, 2000, 3000, 5000, Full
- [ ] Generate crossover comparison with TabPFN
- [ ] Identify crossover points

### Week 5-6: Context Analysis
- [ ] Test different context sizes for SAP-RPT-1
- [ ] Compare with TabPFN context behavior
- [ ] Document optimal recommendations

### Week 7-8: Dataset Expansion
- [ ] Add small datasets (N < 500)
- [ ] Run full benchmark on new datasets
- [ ] Add large datasets (Adult, Airlines)

---

## 📊 Expected Insights from SAP-RPT-1 Analysis

### Few-Shot Expected Results:
| Scenario | Expected Winner | Rationale |
|----------|------------------|-----------|
| N=25 | TabPFN | Simpler attention mechanism |
| N=50-100 | Either | Depends on semantic value of column names |
| N=200-500 | TabPFN | More context helps TabPFN |
| N>500 | Either | Both converge |

### Crossover Expected Results:
| Model | Expected Crossover Point |
|-------|-------------------------|
| TabPFN | N ≈ 1,500-3,000 |
| SAP-RPT-1 | N ≈ 2,000-4,000 (likely later) |

---

## 🔧 Technical Commands Reference

### Running Few-Shot Analysis
```bash
# Existing few-shot (TabPFN, XGBoost, LightGBM, CatBoost, MLP)
python src/analysis/learning_curves.py --datasets diabetes wdbc australian credit-g titanic banknote-auth --train_sizes 25 50 100 200 500 null --n_seeds 3
```

### Running Crossover Analysis
```bash
# Adult dataset crossover
python src/analysis/learning_curves.py --datasets adult --train_sizes 50 100 250 500 1000 2000 3000 5000 null --n_seeds 3
```

### Running Full Benchmark
```bash
# All models on all datasets
python experiments/run_benchmark.py --all --n_seeds 5
```

---

## 📂 Files to Update After SAP-RPT-1 Analysis

1. **reports/few_shot_analysis_report.md**
   - Add SAP-RPT-1 column to all tables
   - Compare TabPFN vs SAP-RPT-1 few-shot performance
   - Update key findings

2. **reports/summary.md**
   - Add SAP-RPT-1 crossover findings
   - Update decision framework with SAP-RPT-1 specific guidance

3. **PROJECT_ROADMAP.md**
   - Mark SAP-RPT-1 analysis as complete

---

*This roadmap is updated regularly. Check back for progress!*
*Last significant update: July 19, 2026*