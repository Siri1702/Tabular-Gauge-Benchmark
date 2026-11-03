"""Capture installed model versions and Stage-II CSV metadata into
two CSVs we can cite from in the paper."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\siri\Desktop\Tabfm-benchmark")
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "dependency_versions.json"
MD_OUT   = OUT_DIR / "appendix_dependency_versions.md"

PKGS = [
    "tabpfn", "tabpfn-client", "tabular-gauge", "torch", "jax", "flax",
    "xgboost", "lightgbm", "catboost",
    "scikit-learn", "scipy", "optuna",
    "openml", "pandas", "numpy",
    "netcal", "matplotlib", "seaborn",
    "psutil",
]

def pip_show(pkg: str):
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "show", pkg],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""

def version_of(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "(not found)"

versions = {}
for p in PKGS:
    info = pip_show(p)
    versions[p] = version_of(info) if info else "(not installed)"

# runtime info
runtime = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "executable": sys.executable,
}

JSON_OUT.write_text(json.dumps(
    {"runtime": runtime, "packages": versions},
    indent=2,
), encoding="utf-8")

with MD_OUT.open("w", encoding="utf-8") as fh:
    fh.write("# Appendix B — Dependency Versions\n\n")
    fh.write("Captured automatically. Reproduce with:\n\n")
    fh.write("```\npython scripts/_audit_versions.py\n```\n\n")
    fh.write(f"- **Python**: `{runtime['python']}`\n")
    fh.write(f"- **Platform**: `{runtime['platform']}`\n\n")
    fh.write("| Package | Version |\n|---|---|\n")
    for p in PKGS:
        fh.write(f"| {p} | `{versions[p]}` |\n")

print("Versions captured.")
print(json.dumps({"runtime": runtime, "packages": versions}, indent=2))
print(f"\nWrote {JSON_OUT}\nWrote {MD_OUT}")
