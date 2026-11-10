# Tabular-Gauge

### Understand how tabular ML models perform, scale with data, and trade off predictive quality against computational cost.

**Tabular-Gauge** is an open-source Python toolkit for configurable benchmarking and evaluation of tabular machine learning models. It brings together predictive performance, probability calibration, computational profiling, and data-efficiency analysis in one experiment workflow.

A model's score at one training-set size is useful, but it does not tell the whole story. Tabular-Gauge also helps you investigate how model performance, stability, and computational requirements change as the amount of training data increases.

The goal is to help researchers and practitioners understand not only **which model performs well**, but also **how it behaves across data regimes, how reliable its predictions are, and what resources it requires**.

---

## Why look beyond a single model ranking?

A benchmark at one dataset size answers a specific question: how did these models perform under this particular evaluation setup?

Model selection often involves further questions:

- Which models perform well when labeled data is limited?
- How does performance change as more training observations become available?
- Does the leading model change between smaller and larger training sets?
- How consistent are the results across repeated runs?
- Do predicted probabilities reflect observed outcomes?
- What additional training time, inference time, and memory are required?

These questions matter because the best choice can depend on the amount of data available, the reliability required, and the computational constraints of the use case.

Tabular-Gauge provides a configurable workflow for examining these dimensions together. It is intended to complement existing benchmark ecosystems, not to replace established public leaderboards or claim that any one evaluation dimension is new.

## What you can evaluate

| Evaluation dimension | Questions it helps answer | Examples of available measures and outputs |
|---|---|---|
| **Predictive performance** | How well does the model distinguish between classes and make predictions? | ROC-AUC, Average Precision / PR-AUC, F1, accuracy, and balanced accuracy |
| **Probability reliability** | Are the model's predicted probabilities aligned with observed outcomes? | Brier score, log loss, ECE, MCE, and calibration curves |
| **Computational efficiency** | What resources does the model use during tuning, fitting, and prediction? | Tuning time, fit time, prediction time, total time, peak memory, and throughput |
| **Statistical comparison** | Are observed differences consistent across datasets or repeated runs? | Confidence intervals, model ranking summaries, and statistical comparison reports |
| **Data efficiency and scaling** | How does model behavior change as the training set grows? | Learning curves, fit-time and memory scaling plots, stability heatmaps, and low-data versus full-data winner summaries |

See [METRICS_DOCUMENTATION.md](./METRICS_DOCUMENTATION.md) for metric definitions, interpretation, statistical methods, and hardware considerations.

## Data-efficiency and scaling analysis

A central capability of Tabular-Gauge is its configurable **Data-Efficiency & Scaling Analysis** workflow.

Instead of evaluating a model at only one training-set size, you can run experiments across multiple training sizes and random seeds, then examine how predictive performance and computational requirements evolve.

This helps investigate questions such as:

- **Low-data performance:** Which models perform well when only a limited number of labeled observations are available?
- **Learning behavior:** How does performance change as more training data is introduced?
- **Changes in model rankings:** Does the low-data winner remain the strongest model at the largest evaluated training size?
- **Stability:** Does a model's performance vary substantially across random seeds, particularly in smaller-data regimes?
- **Computational scaling:** How do fitting time and peak memory change as training data increases?

The generated analysis includes:

- **Learning curves** showing the selected evaluation metric against training-set size.
- **Efficiency scaling plots** for fitting time and peak memory.
- **Stability heatmaps** showing variation across seeds for each model and training size.
- **Analytical tables** summarizing performance and stability by training size.
- **A summary report** identifying the low-data winner and the winner at the largest evaluated training size.

The analysis describes the behavior observed under the configured experiment. It does not currently claim to estimate exact model crossover points, recommend an optimal training size, or automatically select a model under a compute budget.

For configuration options and interpretation guidance, see the [Data-Efficiency & Scaling Analysis Guide](./FEW_SHOT_GUIDE.md).

## Models included

The current built-in model wrappers are:

| Model | Model family |
|---|---|
| **TabPFN 3.5** | Tabular foundation model |
| **XGBoost** | Gradient-boosted decision trees |
| **LightGBM** | Gradient-boosted decision trees |
| **CatBoost** | Gradient-boosted decision trees |
| **MLP** | Multi-layer perceptron |

Optional dependencies determine which model wrappers can be run in a given environment. TabPFN 3.5 also requires the applicable Prior Labs account, license acceptance, and API-key configuration; see the [User Guide](./USER_GUIDE.md).

## How the workflow fits together

A typical Tabular-Gauge experiment follows four steps:

1. **Configure:** Choose datasets, enabled models, experiment settings, training sizes, and random seeds.
2. **Run:** Execute the benchmark or data-efficiency experiment.
3. **Generate reports:** Convert result files into figures, tables, and summary reports.
4. **Inspect trade-offs:** Compare predictive performance, probability reliability, stability, and computational requirements.

The standard benchmark and the data-efficiency workflow serve related but distinct purposes: the standard benchmark compares models under the configured evaluation setup, while the scaling workflow repeats evaluation over different training-set sizes.

---

## Getting started

### Requirements

- Python 3.10 or later
- A virtual environment is recommended

### 1. Clone the repository

```bash
git clone https://github.com/Siri1702/Tabular-Gauge-Benchmark.git
cd Tabular-Gauge-Benchmark
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Install the core package:

```bash
pip install -e .
```

Install the optional model dependencies you need:

```bash
# Conventional baselines: XGBoost, LightGBM, CatBoost, and Optuna
pip install -e ".[baselines]"

# Foundation-model dependencies, including TabPFN
pip install -e ".[foundation]"

# Both optional groups
pip install -e ".[baselines,foundation]"
```

You do not need to install every optional group if you only intend to run a subset of models.

### 3. Run a standard benchmark

Run all datasets configured in `configs/datasets.yaml`:

```bash
python experiments/run_benchmark.py --all --force
```

Generate the figures, tables, and summary report:

```bash
python experiments/generate_reports.py
```

The standard report workflow includes model-performance summaries, ranking visualizations, efficiency comparisons, calibration curves, and statistical outputs where the experiment contains enough repeated observations.

### 4. Run a data-efficiency experiment

Run the few-shot analysis on the included `smoke_test` dataset using the default training-size grid:

```bash
python experiments/run_few_shot.py --datasets smoke_test
```

You can also specify training sizes and the number of random seeds:

```bash
python experiments/run_few_shot.py --datasets smoke_test --train_sizes 25 50 100 500 --n_seeds 3
```

Generate the data-efficiency reports using the results CSV produced by the run. Replace `my_test` with the experiment name configured in `configs/experiment.yaml`:

```bash
python experiments/generate_few_shot_reports.py --csv experiments/results/few_shot/my_test/all_few_shot_results.csv
```

The scaling reports are written to `reports/few_shot/{experiment_name}/`.

For additional options, see [FEW_SHOT_GUIDE.md](./FEW_SHOT_GUIDE.md).

---

## Use your own datasets

Tabular-Gauge supports local CSV files and OpenML datasets through the dataset configuration in `configs/datasets.yaml`.

### Local CSV example

Place your CSV file in the repository (for example, under `data/`) and add an entry to the dataset list:

```yaml
datasets:
  - name: my_dataset
    source: csv
    path: data/my_dataset.csv
    target: target
```

Replace `target` with the name of the column you want to predict.

### OpenML example

Add an OpenML dataset entry using its dataset ID:

```yaml
datasets:
  - name: adult
    source: openml
    id: 31
```

Then run the benchmark for the configured dataset:

```bash
python experiments/run_benchmark.py --datasets my_dataset
```

For more details about dataset configuration and preprocessing, see the [User Guide](./USER_GUIDE.md).

## Extend the model lineup

Tabular-Gauge uses model wrappers and a `ModelFactory` registry to instantiate models from configuration. You can extend the toolkit by implementing a wrapper that follows the project's model interface and registering it with the factory.

The model configuration is maintained in `configs/models.yaml`. Refer to [CONTRIBUTING.md](./CONTRIBUTING.md) and the [User Guide](./USER_GUIDE.md) for the extension workflow.

## Understanding the outputs

| Workflow | Results and reports |
|---|---|
| **Standard benchmark** | Raw and aggregated experiment results are stored under `experiments/results/`; generated summaries, figures, and tables are stored under `reports/{experiment_name}/`. |
| **Data-efficiency analysis** | Per-dataset and combined result CSVs are stored under `experiments/results/few_shot/{experiment_name}/`; learning curves, scaling plots, stability heatmaps, tables, and the summary are stored under `reports/few_shot/{experiment_name}/`. |

The exact reports available depend on the selected workflow, metrics, and number of datasets or seeds in the experiment. Statistical summaries and confidence intervals require sufficient repeated observations; a single run should not be treated as strong statistical evidence. For more rigorous comparisons, use multiple seeds and consult the methodology in [METRICS_DOCUMENTATION.md](./METRICS_DOCUMENTATION.md).

## Current scope and limitations

- **Task scope:** The current documented workflow is focused on binary classification.
- **Hardware dependence:** Runtime, latency, memory, and throughput measurements depend on the execution environment. For meaningful comparisons, run models under the same hardware and software conditions and interpret resource measurements in context.
- **Optional model dependencies:** Some models require additional packages or external authentication. Install and configure only the models you intend to run.
- **Future extensions:** Multiclass classification, regression, and additional evaluation capabilities are planned areas of development; they should not be interpreted as supported features in the current release.

## Documentation

- [User Guide](./USER_GUIDE.md) — installation, configuration, benchmark execution, datasets, and model integration.
- [Data-Efficiency & Scaling Analysis Guide](./FEW_SHOT_GUIDE.md) — training-size experiments, learning curves, scaling reports, and interpretation.
- [Metrics & Evaluation Documentation](./METRICS_DOCUMENTATION.md) — metric definitions, statistical methods, and hardware considerations.
- [Documentation Index](./INDEX.md) — navigation across project documentation.
- [Contributing Guide](./CONTRIBUTING.md) — contributing models, datasets, and improvements.
- [Changelog](./CHANGELOG.md) — project changes and release history.
- [Citation](./CITATION.cff) — citation metadata.

## Contributing

Contributions, feedback, and constructive criticism are welcome. You can help by:

- Adding model wrappers or evaluation metrics.
- Contributing datasets and reproducible experiment configurations.
- Improving reports, visualizations, documentation, or tests.
- Reporting bugs and suggesting new evaluation questions.

Please open an issue to discuss substantial changes and refer to [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting a pull request.

## Citation

If you use Tabular-Gauge in research or other work, please refer to [`CITATION.cff`](./CITATION.cff) for the citation metadata.
