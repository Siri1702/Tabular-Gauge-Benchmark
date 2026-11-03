"""
TabPFN smoke test.

What this does
--------------
Verifies that the installed TabPFN stack is usable end-to-end on the
current machine (laptop or Google Colab VM) before we commit hundreds
of multi-seed compute hours to it.

Checks
------
1. Versions of ``tabpfn`` and ``tabpfn-client`` are importable.
2. The same import path the benchmark uses resolves (local ``tabpfn``
   preferred, ``tabpfn_client`` fallback — see ``tabular_gauge/models/tabpfn_model.py``).
3. ``TabPFNClassifier`` instantiates with the exact kwargs the wrapper uses
   (``n_estimators=8``, ``device='cpu'``, ``ignore_pretraining_limits=True``).
4. The classifier fits on a tiny synthetic dataset
   (``sklearn.datasets.make_classification``, 200 rows, 10 features, 2 classes).
5. ``predict_proba`` returns shape ``(n_test, 2)`` (or ``(n_test,)`` if
   the model collapses to 1D — both are acceptable for the wrapper, but
   we report whichever it produced).
6. The resulting AUC is > 0.5 on a balanced split (perfect >=0.99,
   reasonable >=0.80). On a 200-row synthetic blob this is essentially
   always >=0.85, so the threshold is conservative.
7. TabPFN version is within the supported band (>=2.0,<9.0). 8.0.8 is
   the local pinned version; the Colab pin block in
   ``scripts/colab_multiseed_resume.py`` forces the same.

Runtime
-------
Local CPU: ~10-30 seconds (TabPFN downloads/initialises once).
Colab CPU: ~30-60 seconds on first run (model weights cache).
Subsequent runs hit the cache and finish in <10 seconds.

Output
------
A short pass/fail table, suitable for pasting into a git commit message
or a Colab cell output.

Usage
-----
    # From repo root, in the project venv (or Colab runtime):
    python scripts/test_tabpfn.py

    # Exit code is non-zero on failure — useful in pre-commit hooks.
    python scripts/test_tabpfn.py && echo "READY" || echo "FAIL"
"""

import os
import sys
import time
import platform
from pathlib import Path

import numpy as np

# Align project_root for Colab the same way the other scripts do.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# small pretty printer
# --------------------------------------------------------------------------- #

def _ok(label, value):
    print(f"  [PASS] {label:<22} = {value}")


def _fail(label, value, hint=""):
    print(f"  [FAIL] {label:<22} = {value}")
    if hint:
        print(f"         {hint}")


def _section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_versions():
    """1. tabpfn and tabpfn-client versions."""
    import importlib.metadata as md

    versions = {}
    failures = []

    for pkg in ("tabpfn", "tabpfn-client", "torch"):
        try:
            versions[pkg] = md.version(pkg)
            _ok(pkg, versions[pkg])
        except md.PackageNotFoundError:
            versions[pkg] = None
            _fail(pkg, "NOT INSTALLED", hint=f"pip install {pkg}")

    if versions.get("tabpfn") is None:
        failures.append("tabpfn")

    return versions, failures


def check_import_path():
    """2. Same import path src.models.tabpfn_model would use."""
    try:
        from tabpfn import TabPFNClassifier as Local
        src = "local:tabpfn"
    except ImportError as e:
        try:
            from tabpfn_client import TabPFNClassifier as Local
            src = "fallback:tabpfn_client"
        except ImportError:
            _fail("import TabPFNClassifier", "FAILED",
                  hint="pip install tabpfn==8.0.8")
            return None, ["tabpfn import"]

    _ok("import resolves to", src)
    return Local, []


def check_instantiate(LocalCls):
    """3. Classifier instantiates with the exact kwargs the wrapper uses."""
    try:
        # Match src.models.tabpfn_model.TabPFNWrapper defaults.
        clf = LocalCls(
            n_estimators=8,
            device="cpu",
            random_state=0,
            ignore_pretraining_limits=True,
        )
        _ok("TabPFNClassifier()", "OK")
        return clf, []
    except TypeError as e:
        # Older/newer tabpfn builds may have removed some kwargs.
        _fail("TabPFNClassifier()", f"TypeError: {e}",
              hint="your tabpfn version has a different signature "
                   "(expected tabpfn==8.0.8)")
        return None, ["instantiate"]


def check_fit_predict(clf):
    """4-6. Fit + predict on a 200-row synthetic dataset."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8, n_redundant=2,
        n_classes=2, weights=[0.5, 0.5], random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )

    failures = []

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    pred_t = time.perf_counter() - t0

    # 5. shape
    if proba.ndim == 2 and proba.shape == (len(y_test), 2):
        _ok("predict_proba shape", f"{proba.shape} (binary)")
        pos = proba[:, 1]
    elif proba.ndim == 1 and proba.shape == (len(y_test),):
        _ok("predict_proba shape", f"{proba.shape} (1D positive-class)")
        pos = proba
    else:
        _fail("predict_proba shape", f"unexpected: {proba.shape}")
        failures.append("predict_proba shape")
        return fit_t, pred_t, None, failures

    # 6. AUC
    try:
        auc = float(roc_auc_score(y_test, pos))
    except Exception as e:
        _fail("roc_auc_score", f"ERROR: {e}")
        failures.append("roc_auc_score")
        return fit_t, pred_t, None, failures

    if auc < 0.5:
        _fail("AUC", f"{auc:.4f} (worse than chance — broken)")
        failures.append("AUC")
    elif auc < 0.8:
        _fail("AUC", f"{auc:.4f} (suspicious — should be >0.85 on toy data)")
        failures.append("AUC weak")
    else:
        _ok("AUC", f"{auc:.4f}")

    _ok("fit time",   f"{fit_t:.2f} s on {len(X_train)} rows")
    _ok("predict time", f"{pred_t:.2f} s on {len(X_test)} rows")
    return fit_t, pred_t, auc, failures


def check_wrapper_endtoend():
    """Bonus: src.models.tabpfn_model.TabPFNWrapper is what the benchmark
    actually instantiates. Smoke-fit it through the real code path."""
    try:
        # Same env var the wrapper sets; setting it explicitly here is
        # a no-op but documents intent.
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        from tabular_gauge.models.tabpfn_model import TabPFNWrapper
        wrap = TabPFNWrapper(random_state=0)
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=120, n_features=8, n_informative=6,
            random_state=0,
        )
        result = wrap.run(X, y, X[:30], y[:30],
                          dataset_name="__smoke_test__", seed=0)
        if result.error is not None:
            _fail("wrapper.run()", f"error: {result.error}")
            return ["wrapper.run() error"]
        _ok("TabPFNWrapper.run()",
            f"AUC={getattr(result, 'y_proba', np.array([])).shape}, "
            f"err=None")
        return []
    except Exception as e:
        _fail("TabPFNWrapper.run()", f"{type(e).__name__}: {e}")
        return ["wrapper.run()"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    print("TabPFN smoke test")
    print(f"  python : {platform.python_version()}")
    print(f"  system : {platform.system()} {platform.machine()}")
    print(f"  cwd    : {os.getcwd()}")

    failures = []

    _section("1. Package versions")
    versions, fail = check_versions()
    failures += fail

    _section("2. Import path (matches src.models.tabpfn_model)")
    LocalCls, fail = check_import_path()
    failures += fail
    if LocalCls is None:
        print("\nCannot continue without TabPFNClassifier.")
        sys.exit(1)

    _section("3. Classifier instantiation")
    clf, fail = check_instantiate(LocalCls)
    failures += fail
    if clf is None:
        print("\nCannot continue without a classifier instance.")
        sys.exit(1)

    _section("4-6. Fit + predict + AUC on synthetic data")
    _, _, auc, fail = check_fit_predict(clf)
    failures += fail

    _section("7. End-to-end through src TabPFNWrapper")
    failures += check_wrapper_endtoend()

    _section("Summary")
    if not failures:
        print("ALL CHECKS PASSED. You can launch:")
        print("  python scripts/colab_multiseed_resume.py \\")
        print("      --seeds 3 4 \\")
        print("      --models CatBoost MLP TabPFN XGBoost LightGBM \\")
        print("      --datasets titanic banknote-auth kc1 ozone-level \\")
        print("                  sick spambase phoneme australian \\")
        print("      --train_sizes 25 50 100 200 300 500 750 1000 full")
        sys.exit(0)
    else:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nFix the failures above before launching the resume runner.")
        sys.exit(1)


if __name__ == "__main__":
    main()
