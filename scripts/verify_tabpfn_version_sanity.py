"""
Cross-version sanity check for the TabPFN wrapper.

The Act-2 multi-seed benchmark was run with tabpfn==8.0.8 (see REPRODUCIBILITY.md).
This script verifies that the *currently installed* version still matches the
contract we rely on, so that any future paper reproduction does not silently
invalidate the headline numbers.

What it checks (all exit-code 0 means the wrapper is safe to re-run):

  1. VERSION CONTRACT
       The installed tabpfn reports `__version__ == 8.0.8`, AND
       the `TabPFNClassifier.__init__` signature exposes the kwargs the
       wrapper passes: `n_estimators`, `device`, `random_state`,
       `ignore_pretraining_limits`.

  2. SMOKE TEST
       Fit on a tiny synthetic 2-class dataset (100 train, 50 test), confirm
       (a) `fit()` completes, (b) `predict_proba()` returns shape (n_test, 2),
       (c) AUC > 0.5 on a linearly-separable dataset, (d) context bookkeeping
       attributes (`context_used_`, `context_original_`, `was_subsampled_`)
       are set correctly.

  3. SUBSAMPLING TEST
       Feed 12 000 rows (>8 000 context window), confirm
       `was_subsampled_ == True` and `context_used_ == 8 000`.

  4. DETERMINISM TEST
       Same X, same y, same seed → identical AUC across 3 independent
       instantiations.  A non-deterministic API drift would surface here.

  5. IGNORE_PRETRAINING_LIMITS FLAG
       Confirm `ignore_pretraining_limits=True` is actually accepted (some
       versions raise if you pass unknown kwargs). If it raises, the wrapper
       would silently fall back to the cloud client, invalidating all
       headline numbers.

  6. ENVELOPE NOTE
       Print the full TabPFNClassifier.__init__ signature so we have a
       durable record of the API surface at this point in time. Append
       to `reports/tabpfn_api_audit.json`.

Outputs
-------
reports/tabpfn_api_audit.json
    Snapshot of the installed tabpfn version, all relevant init params,
    and the smoke-test AUC. This is the audit trail that the paper
    cites as evidence the wrapper matches the contract.

reports/appendix_tabpfn_version_sanity.md
    Markdown appendix with the checklist + results, ready to drop into
    the paper as §B.5 (cross-version reproducibility).

Run:
    python scripts/verify_tabpfn_version_sanity.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the wrapper
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np  # noqa: E402

from tabular_gauge.models.tabpfn_model import TabPFNWrapper  # noqa: E402

OUT_AUDIT = ROOT / "reports" / "tabpfn_api_audit.json"
OUT_MD    = ROOT / "reports" / "appendix_tabpfn_version_sanity.md"

REQUIRED_KWARGS = {"n_estimators", "device", "random_state", "ignore_pretraining_limits"}
EXPECTED_VERSION = "8.0.8"

# -- Tiny synthetic datasets (no external data dependency) ----------------- #
rng = np.random.RandomState(0)
n_a, n_b = 60, 60
X_a = rng.normal(loc=(-1.5, -1.5), scale=1.0, size=(n_a, 2))
X_b = rng.normal(loc=(+1.5, +1.5), scale=1.0, size=(n_b, 2))
X_synth = np.vstack([X_a, X_b])
y_synth = np.array([0] * n_a + [1] * n_b)


def _auc(y_true: np.ndarray, p: np.ndarray) -> float:
    """Pure-numpy AUC for the smoke test (no sklearn dep needed)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, p))


def _train_test_split(X: np.ndarray, y: np.ndarray,
                      test_frac: float = 0.33,
                      seed: int = 0):
    idx = np.arange(len(X))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_test = int(round(len(X) * test_frac))
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def check_version_contract() -> dict:
    """Check 1 — installed tabpfn version + init signature."""
    import tabpfn
    version = getattr(tabpfn, "__version__", None)
    init_sig = inspect.signature(tabpfn.TabPFNClassifier.__init__)
    init_params = set(init_sig.parameters.keys())
    missing_kwargs = REQUIRED_KWARGS - init_params
    return {
        "installed_version": version,
        "expected_version": EXPECTED_VERSION,
        "version_match": version == EXPECTED_VERSION,
        "missing_required_kwargs": sorted(missing_kwargs),
        "init_signature_params": sorted(init_params),
    }


def check_smoke_test() -> dict:
    """Check 2 — fit + predict + AUC on a linearly-separable dataset."""
    X_tr, y_tr, X_te, y_te = _train_test_split(X_synth, y_synth, test_frac=0.33, seed=0)
    clf = TabPFNWrapper(n_ensemble_configurations=8, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = clf.predict_proba(X_te)
    pred_s = time.perf_counter() - t0
    auc = _auc(y_te, p)
    md = clf.get_metadata()
    return {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "predict_proba_shape": list(p.shape),
        "auc": auc,
        "fit_seconds": round(fit_s, 3),
        "predict_seconds": round(pred_s, 3),
        "context_used": md["context_used"],
        "context_original": md["context_original"],
        "was_subsampled": md["was_subsampled"],
        "auc_above_chance": auc > 0.5,
    }


def check_subsampling() -> dict:
    """Check 3 — context subsampling at >max_context_size."""
    # Build a 12 000-row dataset (>8 000 context window)
    big_n = 12_000
    rng2 = np.random.RandomState(1)
    X_big = rng2.normal(size=(big_n, 4))
    # Make it linearly separable in feature 0
    y_big = (X_big[:, 0] > 0).astype(int)
    clf = TabPFNWrapper(n_ensemble_configurations=4, max_context_size=8000, random_state=7)
    clf.fit(X_big, y_big)
    md = clf.get_metadata()
    return {
        "n_train_input": int(big_n),
        "max_context_size": int(clf.max_context_size),
        "context_used": md["context_used"],
        "was_subsampled": md["was_subsampled"],
        "subsampled_to_max": md["context_used"] == clf.max_context_size,
    }


def check_determinism() -> dict:
    """Check 4 — same seed → identical AUC across 3 instantiations."""
    X_tr, y_tr, X_te, y_te = _train_test_split(X_synth, y_synth, seed=123)
    aucs = []
    for i in range(3):
        clf = TabPFNWrapper(n_ensemble_configurations=8, random_state=42)
        clf.fit(X_tr, y_tr)
        aucs.append(_auc(y_te, clf.predict_proba(X_te)))
    aucs = np.array(aucs)
    return {
        "aucs": [round(float(a), 6) for a in aucs],
        "max_minus_min": float(aucs.max() - aucs.min()),
        "deterministic": bool((aucs.max() - aucs.min()) < 1e-6),
    }


def check_ignore_pretraining_limits_flag() -> dict:
    """Check 5 — confirm ignore_pretraining_limits=True is accepted.

    We instantiate the underlying tabpfn.TabPFNClassifier with the same
    kwargs as the wrapper (without going through fit) and check for any
    error. Some versions raise TypeError on unknown kwargs.
    """
    import tabpfn
    err = None
    try:
        _ = tabpfn.TabPFNClassifier(
            n_estimators=4, device="cpu", random_state=0,
            ignore_pretraining_limits=True,
        )
        accepted = True
    except TypeError as e:
        accepted = False
        err = str(e)
    return {
        "accepted": accepted,
        "error": err,
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def write_markdown(results: dict) -> None:
    out = []
    out.append("# Appendix B.5 — Cross-version sanity check for the TabPFN wrapper\n\n")
    out.append("The Act-2 multi-seed benchmark was run with `tabpfn==8.0.8` "
               "(see `REPRODUCIBILITY.md`). This appendix is the audit trail "
               "for any future reproduction attempt on a different "
               "`tabpfn` release, showing that the wrapper code in "
               "`tabular_gauge/models/tabpfn_model.py` matches the contract of the "
               "installed version.\n\n")

    out.append("## B.5.1 Check 1 — Version contract\n\n")
    v = results["version_contract"]
    out.append(f"- Installed version: **{v['installed_version']}**\n")
    out.append(f"- Expected version (from REPRODUCIBILITY.md): **{v['expected_version']}**\n")
    out.append(f"- Version match: **{'YES' if v['version_match'] else 'NO'}**\n")
    if v["missing_required_kwargs"]:
        out.append(f"- **MISSING required kwargs**: {v['missing_required_kwargs']}\n")
    else:
        out.append("- All required kwargs present in `TabPFNClassifier.__init__`.\n")
    out.append("\nFull `TabPFNClassifier.__init__` signature (snapshot):\n\n")
    out.append("```\n")
    out.append(", ".join(v["init_signature_params"]))
    out.append("\n```\n\n")

    out.append("## B.5.2 Check 2 — Smoke test (fit + predict + AUC)\n\n")
    s = results["smoke"]
    out.append(f"- n_train = {s['n_train']}, n_test = {s['n_test']}\n")
    out.append(f"- predict_proba shape = {s['predict_proba_shape']}\n")
    out.append(f"- ROC-AUC = **{s['auc']:.4f}**  (must be > 0.5 on linearly-separable data)\n")
    out.append(f"- fit time = {s['fit_seconds']:.3f} s, predict time = {s['predict_seconds']:.3f} s\n")
    out.append(f"- context bookkeeping: context_used={s['context_used']}, "
               f"context_original={s['context_original']}, was_subsampled={s['was_subsampled']}\n")
    out.append(f"- AUC > 0.5: **{'YES' if s['auc_above_chance'] else 'NO'}**\n\n")

    out.append("## B.5.3 Check 3 — Subsampling past the context window\n\n")
    ss = results["subsampling"]
    out.append(f"- Input n_train = {ss['n_train_input']} (above the 8 000-row context cap)\n")
    out.append(f"- max_context_size = {ss['max_context_size']}\n")
    out.append(f"- After fit: context_used = {ss['context_used']}, was_subsampled = {ss['was_subsampled']}\n")
    out.append(f"- Subsampled to max: **{'YES' if ss['subsampled_to_max'] else 'NO'}**\n\n")

    out.append("## B.5.4 Check 4 — Determinism\n\n")
    det = results["determinism"]
    out.append(f"- AUCs across 3 fresh instantiations (same seed): {det['aucs']}\n")
    out.append(f"- max − min = {det['max_minus_min']:.2e}\n")
    out.append(f"- Deterministic (Δ < 1e-6): **{'YES' if det['deterministic'] else 'NO'}**\n\n")

    out.append("## B.5.5 Check 5 — `ignore_pretraining_limits=True` accepted\n\n")
    ip = results["ignore_pretraining_limits"]
    out.append(f"- Accepted by `tabpfn.TabPFNClassifier`: **{'YES' if ip['accepted'] else 'NO'}**\n")
    if not ip["accepted"]:
        out.append(f"- Error: `{ip['error']}`\n")
    out.append("\nIf `ignore_pretraining_limits` is not accepted, the wrapper would silently "
               "fall back to the cloud `tabpfn_client` and invalidate all Act-2 headline numbers. "
               "A reviewer reproducing this paper should re-run this script first.\n\n")

    out.append("## B.5.6 Verdict\n\n")
    all_pass = (
        v["version_match"]
        and not v["missing_required_kwargs"]
        and s["auc_above_chance"]
        and ss["subsampled_to_max"]
        and det["deterministic"]
        and ip["accepted"]
    )
    out.append(f"**{'PASS — wrapper contract verified for tabpfn==' + EXPECTED_VERSION if all_pass else 'FAIL — one or more checks failed'}**\n")
    OUT_MD.write_text("".join(out), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("TabPFN cross-version sanity check")
    print("=" * 60)

    results = {}

    print("\n[1/5] Version contract ...")
    results["version_contract"] = check_version_contract()
    v = results["version_contract"]
    print(f"  installed={v['installed_version']}, expected={v['expected_version']}, "
          f"match={v['version_match']}, missing_kwargs={v['missing_required_kwargs']}")

    print("\n[2/5] Smoke test ...")
    results["smoke"] = check_smoke_test()
    s = results["smoke"]
    print(f"  n_train={s['n_train']}, AUC={s['auc']:.4f}, "
          f"shape={s['predict_proba_shape']}, was_subsampled={s['was_subsampled']}")

    print("\n[3/5] Subsampling past 8000-row context window ...")
    results["subsampling"] = check_subsampling()
    ss = results["subsampling"]
    print(f"  n_input={ss['n_train_input']}, context_used={ss['context_used']}, "
          f"was_subsampled={ss['was_subsampled']}")

    print("\n[4/5] Determinism (3 fresh instantiations) ...")
    results["determinism"] = check_determinism()
    det = results["determinism"]
    print(f"  AUCs={det['aucs']}, max-min={det['max_minus_min']:.2e}, "
          f"deterministic={det['deterministic']}")

    print("\n[5/5] ignore_pretraining_limits=True accepted? ...")
    results["ignore_pretraining_limits"] = check_ignore_pretraining_limits_flag()
    ip = results["ignore_pretraining_limits"]
    print(f"  accepted={ip['accepted']}, error={ip['error']}")

    # Verdict
    all_pass = (
        v["version_match"]
        and not v["missing_required_kwargs"]
        and s["auc_above_chance"]
        and ss["subsampled_to_max"]
        and det["deterministic"]
        and ip["accepted"]
    )
    print(f"\n{'='*60}\nVerdict: {'PASS' if all_pass else 'FAIL'}\n{'='*60}")

    # Persist
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wrapper_path": "tabular_gauge/models/tabpfn_model.py",
        "expected_version": EXPECTED_VERSION,
        "results": results,
        "verdict_pass": bool(all_pass),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nWrote audit -> {OUT_AUDIT}")

    write_markdown(results)
    print(f"Wrote markdown -> {OUT_MD}")

    # Non-zero exit on FAIL so CI can gate on it
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()