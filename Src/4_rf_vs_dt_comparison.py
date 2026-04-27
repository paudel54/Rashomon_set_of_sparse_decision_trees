"""
Script 4: RF vs DT Rashomon Set Comparison
===========================================
Compares:
  A. Decision Tree (DT) Rashomon set  (from script 2)
  B. Random Forest (RF) "Rashomon set" — approximated by:
       - Training 300 RFs with varied hyperparams & seeds
       - Keeping those within ε of the RF reference accuracy
       - Studying the induced importance distributions

Produces comparison metrics and visualisations:
  - Set size comparison
  - Accuracy range comparison
  - Feature importance distributions (DT vs RF)
  - Individual tree accuracy in DT set vs ensemble accuracy in RF set

Outputs: outputs/figures/comparison_*.png, outputs/rashomon/rf_summary.json
"""

import json, pickle
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT     = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "outputs" / "rashomon" / "rashomon_models.pkl"
FIG_DIR  = ROOT / "outputs" / "figures"
RS_DIR   = ROOT / "outputs" / "rashomon"

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D2E",
    "axes.edgecolor":   "#3A3D55",
    "axes.labelcolor":  "#C8C9D8",
    "xtick.color":      "#8B8FA8",
    "ytick.color":      "#8B8FA8",
    "text.color":       "#E0E1EF",
    "grid.color":       "#2E3048",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

EPSILON = 0.05

# ── Load DT Rashomon set ──────────────────────────────────────────────────────
print("Loading DT Rashomon set …")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

rashomon_models = data["rashomon_models"]
X_train = data["X_train"]; X_test = data["X_test"]
y_train = data["y_train"]; y_test = data["y_test"]
features = data["feature_names"]

dt_accs  = np.array([m["accuracy"]          for m in rashomon_models])
dt_fi    = np.array([m["feature_importance"] for m in rashomon_models])
dt_nlvs  = np.array([m["n_leaves"]           for m in rashomon_models])
dt_depth = np.array([m["depth"]             for m in rashomon_models])

print(f"  DT Rashomon set size: {len(rashomon_models)}")

# ── Build RF Rashomon set ────────────────────────────────────────────────────
print("\nBuilding RF Rashomon set …")
rf_param_grid = {
    "n_estimators":     [50, 100, 200],
    "max_depth":        [3, 5, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", "log2"],
}
keys   = list(rf_param_grid.keys())
values = list(rf_param_grid.values())

# Reference RF
rf_ref = RandomForestClassifier(n_estimators=200, max_depth=None,
                                 random_state=42, n_jobs=-1)
rf_ref.fit(X_train, y_train)
rf_ref_acc = accuracy_score(y_test, rf_ref.predict(X_test))
rf_threshold = rf_ref_acc - EPSILON
print(f"  Reference RF accuracy: {rf_ref_acc:.4f}")
print(f"  RF Rashomon threshold: {rf_threshold:.4f}")

rf_rashomon = []
seeds = list(range(30))
total_rf = np.prod([len(v) for v in values]) * len(seeds)
count_rf  = 0
for combo in product(*values):
    params = dict(zip(keys, combo))
    for seed in seeds:
        count_rf += 1
        clf = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        if acc >= rf_threshold:
            rf_rashomon.append({
                "params":   params | {"random_state": seed},
                "accuracy": float(acc),
                "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
                "auc":      float(roc_auc_score(y_test, y_prob)),
                "feature_importance": clf.feature_importances_.tolist(),
            })

print(f"  RF Rashomon set size: {len(rf_rashomon)}")
rf_accs = np.array([m["accuracy"] for m in rf_rashomon])
rf_fi   = np.array([m["feature_importance"] for m in rf_rashomon])

# ── Save RF summary ────────────────────────────────────────────────────────────
rf_summary = {
    "reference_rf_accuracy": float(rf_ref_acc),
    "epsilon":               EPSILON,
    "rf_rashomon_threshold": float(rf_threshold),
    "rf_rashomon_set_size":  len(rf_rashomon),
    "dt_rashomon_set_size":  len(rashomon_models),
    "rf_accuracy_range": {
        "min":  float(rf_accs.min()),
        "max":  float(rf_accs.max()),
        "mean": float(rf_accs.mean()),
    },
    "dt_accuracy_range": {
        "min":  float(dt_accs.min()),
        "max":  float(dt_accs.max()),
        "mean": float(dt_accs.mean()),
    },
}
with open(RS_DIR / "rf_summary.json", "w") as fp:
    json.dump(rf_summary, fp, indent=2)
print("✓ Saved rf_summary.json")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Accuracy distribution comparison DT vs RF
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Rashomon Set Comparison: Decision Tree vs Random Forest",
             fontsize=14, fontweight="bold", color="#E0E1EF")

# DT accuracy distribution
ax = axes[0]
ax.hist(dt_accs, bins=25, color="#5B8DEF", alpha=0.8, edgecolor="white",
        linewidth=0.4, label="DT Rashomon Set")
ax.axvline(rf_ref_acc, color="#F06449", lw=2, linestyle="--",
           label=f"RF Ref ({rf_ref_acc:.3f})")
ax.axvline(dt_accs.mean(), color="#A2D86B", lw=2, linestyle=":",
           label=f"DT Mean ({dt_accs.mean():.3f})")
ax.set_title("Decision Tree Rashomon Set", fontsize=12, color="#E0E1EF")
ax.set_xlabel("Test Accuracy"); ax.set_ylabel("Count")
ax.legend(fontsize=8, facecolor="#1A1D2E", edgecolor="#3A3D55")
ax.grid(True)

# RF accuracy distribution
ax = axes[1]
ax.hist(rf_accs, bins=20, color="#F06449", alpha=0.8, edgecolor="white",
        linewidth=0.4, label="RF Rashomon Set")
ax.axvline(rf_ref_acc, color="#F06449", lw=2, linestyle="--",
           label=f"RF Ref ({rf_ref_acc:.3f})")
ax.axvline(rf_accs.mean(), color="#FFD700", lw=2, linestyle=":",
           label=f"RF Mean ({rf_accs.mean():.3f})")
ax.set_title("Random Forest Rashomon Set", fontsize=12, color="#E0E1EF")
ax.set_xlabel("Test Accuracy"); ax.set_ylabel("Count")
ax.legend(fontsize=8, facecolor="#1A1D2E", edgecolor="#3A3D55")
ax.grid(True)

plt.tight_layout()
plt.savefig(FIG_DIR / "comparison_accuracy_dist.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved comparison_accuracy_dist.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Feature importance distributions DT vs RF side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Feature Importance Distribution in Rashomon Sets",
             fontsize=14, fontweight="bold", color="#E0E1EF")

for ax, fi_mat, label, color in [
    (axes[0], dt_fi, "Decision Trees", "#5B8DEF"),
    (axes[1], rf_fi, "Random Forests", "#F06449"),
]:
    means  = fi_mat.mean(axis=0)
    errors = fi_mat.std(axis=0)
    idx    = np.argsort(means)[::-1]
    bars = ax.bar(range(len(features)), means[idx],
                  yerr=errors[idx], capsize=4,
                  color=color, alpha=0.8, edgecolor="white",
                  linewidth=0.5, error_kw=dict(ecolor="white", lw=1))
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in idx], rotation=35,
                       ha="right", fontsize=9)
    ax.set_title(f"{label} Rashomon Set\n(mean ± std)", fontsize=11, color="#E0E1EF")
    ax.set_ylabel("Feature Importance"); ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(FIG_DIR / "comparison_feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved comparison_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Summary comparison bar chart (metrics)
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "Set Size\n(log scale)":   [np.log10(max(1, len(rashomon_models))),
                                 np.log10(max(1, len(rf_rashomon)))],
    "Mean Accuracy":           [dt_accs.mean(), rf_accs.mean()],
    "Accuracy Range\n(max-min)":
                               [dt_accs.max()-dt_accs.min(),
                                rf_accs.max()-rf_accs.min()],
    "Mean F1":                 [np.mean([m["f1"] for m in rashomon_models]),
                                np.mean([m["f1"] for m in rf_rashomon])],
}

fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Rashomon Set Summary Metrics: DT vs RF",
             fontsize=14, fontweight="bold", color="#E0E1EF")

for ax, (metric, vals) in zip(axes, metrics.items()):
    bars = ax.bar(["DT", "RF"], vals, color=["#5B8DEF", "#F06449"],
                  edgecolor="white", linewidth=0.5, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#E0E1EF")
    ax.set_title(metric, fontsize=10, color="#E0E1EF")
    ax.grid(True, axis="y"); ax.set_ylim(0, max(vals)*1.2 + 0.01)

plt.tight_layout()
plt.savefig(FIG_DIR / "comparison_summary_metrics.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved comparison_summary_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Depth distribution of DT Rashomon set
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("DT Rashomon Set – Structural Characterisation",
             fontsize=13, fontweight="bold", color="#E0E1EF")

depths_u, depths_c = np.unique(dt_depth, return_counts=True)
axes[0].bar(depths_u, depths_c, color="#9B89D4", edgecolor="white",
            linewidth=0.5, alpha=0.85)
axes[0].set_xlabel("Tree Depth"); axes[0].set_ylabel("Count")
axes[0].set_title("Depth Distribution", fontsize=11, color="#E0E1EF")
axes[0].grid(True, axis="y")

leaves_u, leaves_c = np.unique(dt_nlvs, return_counts=True)
axes[1].bar(leaves_u, leaves_c, color="#50C9CE", edgecolor="white",
            linewidth=0.5, alpha=0.85)
axes[1].set_xlabel("Number of Leaves"); axes[1].set_ylabel("Count")
axes[1].set_title("Leaves Distribution", fontsize=11, color="#E0E1EF")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.savefig(FIG_DIR / "comparison_dt_structure.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved comparison_dt_structure.png")

print("\n✅  RF vs DT comparison complete.")
