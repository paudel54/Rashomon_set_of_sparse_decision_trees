"""
Script 3: Feature Importance Distribution (RID-inspired)
=========================================================
Computes the Rashomon Importance Distribution across all trees in the
Rashomon set:
  - For each feature: distribution of importance across all Rashomon trees
  - Stability score (coefficient of variation)
  - Which features are "always important" vs "sometimes important"
  - Marginal importance (how much accuracy drops without that feature)

Outputs: outputs/figures/rid_*.png, outputs/rashomon/rid_results.json
"""

import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

ROOT     = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "outputs" / "rashomon" / "rashomon_models.pkl"
FIG_DIR  = ROOT / "outputs" / "figures"
RS_DIR   = ROOT / "outputs" / "rashomon"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark style ────────────────────────────────────────────────────────────────
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

# ── Load Rashomon set ─────────────────────────────────────────────────────────
print("Loading Rashomon set …")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

rashomon_models = data["rashomon_models"]
X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
features = data["feature_names"]

N = len(rashomon_models)
F = len(features)
print(f"  Rashomon set: {N} trees  |  {F} features")

# ── Build importance matrix (N x F) ──────────────────────────────────────────
fi_matrix = np.array([m["feature_importance"] for m in rashomon_models])
accs      = np.array([m["accuracy"] for m in rashomon_models])

# ── RID metrics ───────────────────────────────────────────────────────────────
def cv(arr):
    """Coefficient of variation (lower = more stable)."""
    return np.std(arr) / (np.mean(arr) + 1e-9)

rid_results = {}
for i, feat in enumerate(features):
    vals = fi_matrix[:, i]
    rid_results[feat] = {
        "mean":         float(np.mean(vals)),
        "std":          float(np.std(vals)),
        "median":       float(np.median(vals)),
        "q05":          float(np.percentile(vals, 5)),
        "q95":          float(np.percentile(vals, 95)),
        "cv":           float(cv(vals)),
        "pct_nonzero":  float(100 * np.mean(vals > 0)),
    }

# ── Marginal accuracy drop (leave-one-feature-out) ────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
print("\nComputing marginal drop (leave-one-out) …")
marginal_drops = {}
for feat_idx, feat in enumerate(features):
    drops = []
    X_lo  = X_test.copy()
    # Permute the feature (breaks its signal)
    rng = np.random.RandomState(99)
    X_lo[:, feat_idx] = rng.permutation(X_lo[:, feat_idx])
    for m in rashomon_models[:200]:   # sample 200 models for speed
        acc_full = m["accuracy"]
        acc_perm = accuracy_score(y_test, m["model"].predict(X_lo))
        drops.append(acc_full - acc_perm)
    marginal_drops[feat] = float(np.mean(drops))

print("  Marginal accuracy drop:")
for f, d in sorted(marginal_drops.items(), key=lambda x: -x[1]):
    print(f"    {f:<22} Δacc = {d:+.4f}")

# Augment rid_results
for feat in features:
    rid_results[feat]["marginal_drop"] = marginal_drops[feat]

with open(RS_DIR / "rid_results.json", "w") as fp:
    json.dump(rid_results, fp, indent=2)
print("\n✓ Saved rid_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Violin plot of importance distributions across Rashomon set
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#5B8DEF","#F06449","#A2D86B","#F7C59F","#9B89D4","#50C9CE","#FFD700","#FF6B9D"]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0F1117")

positions = np.arange(F)
parts = ax.violinplot(
    [fi_matrix[:, i] for i in range(F)],
    positions=positions,
    showmeans=True, showmedians=True, showextrema=True
)
for i, (pc, color) in enumerate(zip(parts["bodies"], PALETTE)):
    pc.set_facecolor(color)
    pc.set_edgecolor("white")
    pc.set_alpha(0.7)
for element in ["cmeans","cmedians","cbars","cmins","cmaxes"]:
    parts[element].set_color("white")
    parts[element].set_linewidth(1.2)

ax.set_xticks(positions)
ax.set_xticklabels(features, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Feature Importance", fontsize=11)
ax.set_title("Rashomon Importance Distribution (RID)\nacross all near-optimal trees",
             fontsize=13, fontweight="bold", color="#E0E1EF")
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(FIG_DIR / "rid_violin.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved rid_violin.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Stability vs Mean Importance scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0F1117")

means = [rid_results[f]["mean"] for f in features]
cvs   = [rid_results[f]["cv"]   for f in features]
marg  = [rid_results[f]["marginal_drop"] for f in features]

scatter = ax.scatter(means, cvs,
                     c=marg, cmap="plasma",
                     s=200, edgecolors="white", linewidths=1, zorder=3)
for i, feat in enumerate(features):
    ax.annotate(feat, (means[i], cvs[i]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, color="#C8C9D8")

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Marginal Accuracy Drop", color="#C8C9D8", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#8B8FA8")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B8FA8")

ax.set_xlabel("Mean Importance (across Rashomon set)", fontsize=11)
ax.set_ylabel("Instability (CV – lower is more stable)", fontsize=11)
ax.set_title("Feature Stability in the Rashomon Set\n(Marginal accuracy impact)",
             fontsize=13, fontweight="bold", color="#E0E1EF")
ax.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "rid_stability_scatter.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved rid_stability_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Feature appearance frequency (% of Rashomon trees that use it)
# ─────────────────────────────────────────────────────────────────────────────
pct_nonzero = [rid_results[f]["pct_nonzero"] for f in features]
sorted_idx  = np.argsort(pct_nonzero)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0F1117")

bars = ax.barh(
    [features[i] for i in sorted_idx],
    [pct_nonzero[i] for i in sorted_idx],
    color=[PALETTE[i % len(PALETTE)] for i in sorted_idx],
    edgecolor="white", linewidth=0.5, alpha=0.85
)
for bar, val in zip(bars, [pct_nonzero[i] for i in sorted_idx]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9, color="#E0E1EF")

ax.set_xlabel("% of Rashomon Trees Using this Feature", fontsize=11)
ax.set_title("Feature Inclusion Frequency across Rashomon Set",
             fontsize=13, fontweight="bold", color="#E0E1EF")
ax.set_xlim(0, 110)
ax.axvline(100, color="#F06449", linewidth=1, linestyle=":")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(FIG_DIR / "rid_feature_frequency.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved rid_feature_frequency.png")

print("\n  RID analysis complete.")
