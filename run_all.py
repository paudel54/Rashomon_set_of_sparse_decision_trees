"""
run_all.py – Initiates the pipeline
Executes all 5 pipeline scripts in sequence.
"""
import subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "Src"

SCRIPTS = [
    ("1️  Data Exploration",             "1_data_exploration.py"),
    ("2️  Rashomon Set Enumeration",      "2_rashomon_set_trees.py"),
    ("3️  Feature Importance (RID)",      "3_feature_importance_rid.py"),
    ("4️  RF vs DT Comparison",           "4_rf_vs_dt_comparison.py"),
    ("5️  TimberTrek Export",             "5_timbertrek_export.py"),
]

print("=" * 60)
print("  RASHOMON SETS – BIOSIGNAL PROJECT PIPELINE")
print("=" * 60)

for label, script in SCRIPTS:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n Script failed: {script}")
        sys.exit(result.returncode)
    print(f"\n  ⏱  Done in {elapsed:.1f}s")

print("\n" + "=" * 60)
print("  ALL SCRIPTS COMPLETE")
print("  Open outputs/ to view all figures and JSON exports.")
print("  Open dashboard.html in your browser for the demo.")
print("=" * 60)
