# Reproducing Research Results

This guide provides a practical path to reproduce the results reported in the associated research paper.

## 1. Environment Setup
### Requirements
- Python 3.10+
- GPU (Optional but recommended for TabPFN-v2)

### Installation
```bash
git clone https://github.com/Siri1702/Tabular-Gauge.git
cd Tabular-Gauge
python -m venv venv
source venv/bin/activate # or venv\Scripts\Activate.ps1
pip install -e ".[baselines,foundation]"
```

## 2. Dataset Acquisition
All datasets are sourced from **OpenML**. The toolkit's `DataLoader` fetches them automatically using the IDs specified in `configs/datasets.yaml`. No manual downloads are required.

## 3. Running Experiments

### Option A: The Smoke Test (Quick Verification)
Verify the pipeline works on a single dataset with a single seed:
```bash
python experiments/run_benchmark.py --datasets australian --n_seeds 1
```

### Option B: Full Reproduction
#### Stage I (Model Selection)
Run the head-to-head benchmark:
```bash
# Run on the 15-dataset superset
python experiments/run_benchmark.py --all --n_seeds 5
```

#### Stage II (Few-Shot Scaling)
Run the multi-seed scaling grid:
```bash
python scripts/run_multiseed_few_shot.py
```
*Note: This is computationally expensive and may take several hours depending on hardware.*

## 4. Regenerating Figures and Tables
Once the results are generated in `experiments/results/`, you can recreate the paper's artifacts:

### 1. Summary Tables
```bash
python scripts/build_paper1_tables.py
```

### 2. Publication Figures
```bash
python scripts/plot_publication_figures.py
```

## 5. Expected Outputs
- **Raw Results:** `experiments/results/few_shot/multiseed_few_shot_results.csv`
- **Tables:** `reports/tables/*.csv`
- **Figures:** `reports/figures/*.png`

## 6. Troubleshooting
- **Memory Errors:** If you encounter OOM during Stage II, reduce the number of concurrent workers or use a machine with more RAM.
- **Dependency Mismatches:** Ensure you are using the versions listed in `scripts/requirements_lock.txt`.
