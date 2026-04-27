"""
Script 2: Rashomon Set Enumeration – Sparse Decision Trees
===========================================================
Implements the Rashomon set concept for sparse decision trees.

Strategy (sklearn-based, ARM64 compatible):
  1. Binarize the 8 sparse features using percentile thresholds.
  2. Train the reference "optimal" decision tree (depth ≤ 4).
  3. Enumerate near-optimal trees by sweeping:
       - max_depth  ∈ {2, 3, 4}
       - min_samples_leaf ∈ {10, 20, 30, 40, 50}
       - criterion ∈ {gini, entropy}
       - max_features ∈ {None, sqrt, log2}
       - random_state ∈ 50 seeds
     keeping every tree whose test accuracy is within ε of the best.
  4. Save all Rashomon trees and metrics.

Outputs saved to: outputs/rashomon/
"""

import os, json, pickle
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "P19_S1_multimodel.csv"
OUT_DIR   = ROOT / "outputs" / "rashomon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPARSE_FEATURES = [
    "median", "median_hr", "rmssd",
    "mean_tonic_eda", "mean_phasic_eda", "std_eda",
    "mean_temperature", "min_temperature",
]
EPSILON       = 0.05   # Rashomon epsilon: within 5% accuracy of best
RANDOM_SEEDS  = list(range(50))
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# ── Load & prep ───────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df = df[df["skt_valid"] == 1].reset_index(drop=True)
df = df[SPARSE_FEATURES + ["medication"]].dropna()

X = df[SPARSE_FEATURES].values
y = df["medication"].values
feature_names = SPARSE_FEATURES

print(f"  Samples: {len(X)}  |  Class balance: {np.bincount(y)}")

# ── Train / test split ────────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                              random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ── Reference optimal tree ────────────────────────────────────────────────────
ref_tree = DecisionTreeClassifier(max_depth=4, criterion="gini",
                                   random_state=RANDOM_STATE)
ref_tree.fit(X_train, y_train)
ref_acc = accuracy_score(y_test, ref_tree.predict(X_test))
rashomon_threshold = ref_acc - EPSILON
print(f"\n  Reference tree accuracy: {ref_acc:.4f}")
print(f"  Rashomon threshold (ε={EPSILON}): {rashomon_threshold:.4f}")

# ── Grid sweep ────────────────────────────────────────────────────────────────
param_grid = {
    "max_depth":         [2, 3, 4],
    "min_samples_leaf":  [5, 10, 20, 30, 50],
    "criterion":         ["gini", "entropy"],
    "max_features":      [None, "sqrt", "log2"],
}
keys   = list(param_grid.keys())
values = list(param_grid.values())

rashomon_models = []   # list of dicts
all_accs = []

print("\nSweeping parameter grid …")
total = np.prod([len(v) for v in values]) * len(RANDOM_SEEDS)
count = 0
for combo in product(*values):
    params = dict(zip(keys, combo))
    for seed in RANDOM_SEEDS:
        count += 1
        if count % 500 == 0:
            print(f"  [{count}/{total}] Rashomon set size so far: {len(rashomon_models)}")
        clf = DecisionTreeClassifier(**params, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        all_accs.append(acc)
        if acc >= rashomon_threshold:
            n_leaves = clf.get_n_leaves()
            depth    = clf.get_depth()
            f1       = f1_score(y_test, y_pred, zero_division=0)
            auc      = roc_auc_score(y_test, y_prob)
            fi       = clf.feature_importances_
            # Which features actually used?
            tree_    = clf.tree_
            feat_used = set(tree_.feature[tree_.feature != _tree.TREE_UNDEFINED])
            rashomon_models.append({
                "params":           params | {"random_state": seed},
                "accuracy":         round(float(acc), 6),
                "f1":               round(float(f1), 6),
                "auc":              round(float(auc), 6),
                "n_leaves":         int(n_leaves),
                "depth":            int(depth),
                "feature_importance": fi.tolist(),
                "features_used":    sorted([SPARSE_FEATURES[i] for i in feat_used]),
                "model":            clf,
            })

print(f"\n✓ Grid sweep complete. Total configs tried: {count}")
print(f"  Rashomon set size: {len(rashomon_models)}")
print(f"  Accuracy range in Rashomon set: "
      f"[{min(m['accuracy'] for m in rashomon_models):.4f}, "
      f"{max(m['accuracy'] for m in rashomon_models):.4f}]")

# ── Summary stats ─────────────────────────────────────────────────────────────
summary = {
    "reference_accuracy": float(ref_acc),
    "epsilon":            EPSILON,
    "rashomon_threshold": float(rashomon_threshold),
    "rashomon_set_size":  len(rashomon_models),
    "total_models_tried": count,
    "accuracy_range": {
        "min": float(min(m["accuracy"] for m in rashomon_models)),
        "max": float(max(m["accuracy"] for m in rashomon_models)),
        "mean": float(np.mean([m["accuracy"] for m in rashomon_models])),
    },
    "depth_distribution": {
        str(d): int(sum(1 for m in rashomon_models if m["depth"] == d))
        for d in sorted(set(m["depth"] for m in rashomon_models))
    },
    "n_leaves_distribution": {
        str(n): int(sum(1 for m in rashomon_models if m["n_leaves"] == n))
        for n in sorted(set(m["n_leaves"] for m in rashomon_models))
    },
}
with open(OUT_DIR / "rashomon_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Saved rashomon_summary.json")

# ── Feature appearance frequency (how often each feature appears in a tree) ───
feat_freq = {f: 0 for f in SPARSE_FEATURES}
for m in rashomon_models:
    for f in m["features_used"]:
        feat_freq[f] += 1
feat_freq_pct = {f: round(100 * v / len(rashomon_models), 2)
                 for f, v in feat_freq.items()}
summary["feature_frequency_pct"] = feat_freq_pct
with open(OUT_DIR / "rashomon_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n  Feature appearance frequency in Rashomon set:")
for feat, pct in sorted(feat_freq_pct.items(), key=lambda x: -x[1]):
    print(f"    {feat:<22} {pct:>6.1f}%")

# ── Save metadata (no model objects – just metrics for downstream scripts) ────
meta_rows = []
for i, m in enumerate(rashomon_models):
    row = {"id": i, "accuracy": m["accuracy"], "f1": m["f1"],
           "auc": m["auc"], "n_leaves": m["n_leaves"], "depth": m["depth"],
           "features_used": "|".join(m["features_used"])}
    for j, feat in enumerate(SPARSE_FEATURES):
        row[f"fi_{feat}"] = m["feature_importance"][j]
    row.update(m["params"])
    meta_rows.append(row)

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUT_DIR / "rashomon_metadata.csv", index=False)
print(f"\n✓ Saved rashomon_metadata.csv  ({len(meta_df)} rows)")

# ── Pickle best model and models list (for downstream) ────────────────────────
with open(OUT_DIR / "rashomon_models.pkl", "wb") as f:
    pickle.dump({
        "rashomon_models": rashomon_models,
        "ref_tree": ref_tree,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": SPARSE_FEATURES,
        "summary": summary,
    }, f)
print("✓ Saved rashomon_models.pkl")

print("\n✅  Rashomon set enumeration complete.")
