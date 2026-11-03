# Contributing to Tabular-Gauge Benchmark

Thank you for your interest in contributing! This project welcomes contributions from researchers and practitioners interested in benchmarking tabular ML approaches.

## 🚀 How to Contribute

### Reporting Issues

- **Bugs**: Open an issue with a clear description, steps to reproduce, and your environment
- **Feature Requests**: Open an issue with `[Feature Request]` prefix
- **Questions**: Open a discussion thread instead of an issue

### Submitting Pull Requests

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes with clear messages (see commit guidelines below)
4. **Push** to your fork: `git push origin feature/your-feature-name`
5. **Open** a Pull Request with a clear description

## 📋 Pull Request Guidelines

### PR Title Format

Use conventional commits style:

```
<type>(<scope>): <description>
```

**Types:**
- `feat`: New feature or model
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring (no behavior change)
- `test`: Adding or updating tests
- `benchmark`: Results or benchmark runs
- `config`: Configuration changes

**Examples:**
```
feat(models): add SAINT model implementation
fix(evaluation): correct Brier score calculation
docs: update dataset list in README
benchmark: run on new OpenML datasets
```

### PR Description

Include in your PR description:

1. **What** — Brief description of changes
2. **Why** — Rationale (optional for docs/refactor)
3. **How** — Key implementation details
4. **Testing** — How you tested your changes

### Review Criteria

PRs are reviewed based on:

- ✅ **Correctness**: Code works as described
- ✅ **Clarity**: Code is readable and well-documented
- ✅ **Consistency**: Follows project conventions
- ✅ **Tests**: Includes appropriate tests (for new features)

## 🧑‍💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/Siri1702/Tabular-Gauge.git
cd Tabular-Gauge

# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\Activate.ps1  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## 🧪 Adding a New Model

To add a new model (e.g., FT-Transformer):

1. **Create** `tabular_gauge/models/ft_transformer_model.py`
2. **Extend** `ModelWrapper` base class:

```python
from tabular_gauge.models.base import ModelWrapper

class FTTransformerWrapper(ModelWrapper):
    def __init__(self, n_trials: int = 50, random_state: int = 42):
        super().__init__(name="FT-Transformer", random_state=random_state)
        self.n_trials = n_trials

    def fit(self, X_train, y_train):
        # Your implementation
        ...

    def predict_proba(self, X_test):
        # Your implementation
        ...
```

3. **Update** `configs/models.yaml` with the new model config.

## 🗄️ Adding a New Dataset

1. **Edit** `configs/datasets.yaml`:

```yaml
- id: <openml_dataset_id>
  name: <dataset_name>
  source: openml
  target: <target_column>
  domain: <domain_category>
  notes: "Brief description"
```

2. **Run** verification:

```bash
python experiments/run_benchmark.py --datasets your-new-dataset
```

## 📝 Coding Standards

### Python Style
- Follow **PEP 8**
- Use **Black** for formatting (line length: 100)
- Use **type hints** where helpful
- Max line length: 100 characters

### Docstrings
Use Google-style docstrings:

```python
def compute_metrics(y_true, y_pred, y_proba):
    """Compute evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.

    Returns:
        dict: Dictionary of metric names to values.

    Example:
        >>> metrics = compute_metrics(y_true, y_pred, y_proba)
        >>> metrics['roc_auc']
        0.85
    """
```

### Variable Naming
- Use descriptive names: `training_time_sec` not `t_time`
- Constants: `MAX_CONTEXT_SIZE = 3000`
- Private methods: `_subsample_balanced()`

## 🧪 Running Tests

```bash
# Run quick benchmark test
python experiments/run_benchmark.py --datasets credit-g --n_seeds 1
```

## 📊 Benchmark Run Guidelines

When adding new benchmark results:

1. **Document** the run in your PR (date, seeds, datasets)
2. **Include** both raw JSON and aggregated CSV
3. **Update** README if results affect reported metrics
4. **Use** the same random seeds for reproducibility

## 💬 Getting Help

- **Discussions**: For questions and community discussion
- **Issues**: For bugs and feature requests

## 🙏 Acknowledgments

Contributions, feedback, and ideas are all welcome! This benchmark is meant to be a community resource for the tabular ML research community.

### Contributor Spotlight

*Contributions to this project, including the initial addition of the SAP-RPT-1 model for research benchmarks, are greatly appreciated.*

---

*By contributing to this project, you agree to follow its code of conduct (be respectful and constructive).*
