# Results Index (Paper-to-Code Traceability)

This matrix maps the results reported in the research manuscript to the actual files and scripts in this repository. 

**Note for Reviewers:** To maintain transparency and prevent overwriting, all research evidence is stored in the `experiments/paper_evidence/` directory, categorized by stage. The "Source Results" below refer to the authoritative research artifacts.

| Paper Item | Experiment | Source Results | Generation Script | Output Artifact | Status |
|---|---|---|---|---|---|
| **Table 4** (Stage I Overall) | Stage I | `experiments/paper_evidence/stage1/paired_stats.csv` | `scripts/build_paper1_tables.py` | `reports/tables/main_results.csv` | ✅ Verified |
| **Fig F1** (Headline AUC) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/fig_F3_headline.py` | `reports/figures/auc_curves.png` | ✅ Verified |
| **Table 6** (AUC across N) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/build_paper1_tables.py` | `reports/tables/few_shot_scaling.csv` | ✅ Verified |
| **Fig F3** (Memory vs N) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/fig_F6_memory.py` | `reports/figures/memory_comparison.png` | ✅ Verified |
| **Fig F4** (Inference vs N) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/fig_F7_inference_time.py` | `reports/figures/timing_comparison.png` | ✅ Verified |
| **Fig F9** (Calibration) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/fig_F9_calibration.py` | `reports/figures/calibration_comparison.png` | ✅ Verified |
| **Crossover Table** | Stage II | `experiments/paper_evidence/stage2/per_dataset_crossover_with_ci.csv` | `scripts/analyze_crossover.py` | `reports/tables/crossover_results.csv` | ✅ Verified |
| **Table 10** (Inference Cost) | Stage II | `experiments/paper_evidence/stage2/multiseed_few_shot_results.csv` | `scripts/build_paper1_tables.py` | `reports/tables/inference_cost.csv` | ✅ Verified |

**Status Legend:**
- ✅ **Verified:** Source data and generation procedure are available and validated.
- ⚠️ **Partial:** Some components are available, but full reproduction is not established.
- ❌ **Missing:** Required materials are unavailable.
