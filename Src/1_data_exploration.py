"""
Data Exploration & Feature Distribution
==================================================
Loads the biosignal dataset, filters valid windows, selects the 8 sparse
features and visualises their distributions (pre-med vs post-med).

Outputs saved to: outputs/figures/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "P19_S1_multimodel.csv"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SPARSE_FEATURES = [
    "median", "median_hr", "rmssd",
    "mean_tonic_eda", "mean_phasic_eda", "std_eda",
    "mean_temperature", "min_temperature",
]
FEATURE_LABELS = {
    "median":          "ECG Median Amplitude",
    "median_hr":       "Median Heart Rate (bpm)",
    "rmssd":           "HRV – RMSSD (ms)",
    "mean_tonic_eda":  "Mean Tonic EDA (µS)",
    "mean_phasic_eda": "Mean Phasic EDA (µS)",
    "std_eda":         "Std EDA (µS)",
    "mean_temperature":"Mean Skin Temp (°C)",
    "min_temperature": "Min Skin Temp (°C)",
}
FEATURE_GROUPS = {
    "ECG": ["median", "median_hr", "rmssd"],
    "EDA": ["mean_tonic_eda", "mean_phasic_eda", "std_eda"],
    "SKT": ["mean_temperature", "min_temperature"],
}
CLASS_COLORS = {0: "#5B8DEF", 1: "#F06449"}
CLASS_NAMES  = {0: "Pre-Medication", 1: "Post-Medication"}

# ── Style ─────────────────────────────────────────────────────────────────────
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
    "font.family":      "DejaVu Sans",
})

# ── Load & clean ──────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape:  {df.shape}")

# Remove invalid SKT windows
df = df[df["skt_valid"] == 1].reset_index(drop=True)
print(f"  After SKT filter: {df.shape}")
print(f"  Label distribution:\n{df['medication'].value_counts()}\n")

df_feat = df[SPARSE_FEATURES + ["medication"]].dropna()
print(f"  Clean rows: {len(df_feat)}")

# ── Save cleaned feature stats ────────────────────────────────────────────────
stats = df_feat.groupby("medication")[SPARSE_FEATURES].agg(["mean","std"]).round(4)
stats.to_csv(OUT_DIR / "feature_stats.csv")
print(f"✓ Saved feature_stats.csv")

# ── Plot 1: Feature distributions per class (Kernel density estimate plot) (KDE + rug) ──────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Feature Distributions: Pre vs Post Medication\n(Sparse Feature Set – P19 S1)",
             fontsize=15, color="#E0E1EF", fontweight="bold", y=1.01)

axes = axes.flatten()
for i, feat in enumerate(SPARSE_FEATURES):
    ax = axes[i]
    for cls in [0, 1]:
        vals = df_feat[df_feat["medication"] == cls][feat].values
        sns.kdeplot(vals, ax=ax, fill=True, alpha=0.35,
                    color=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                    linewidth=2)
        ax.axvline(np.median(vals), linestyle=":", color=CLASS_COLORS[cls],
                   linewidth=1.5, alpha=0.9)
    ax.set_title(FEATURE_LABELS[feat], fontsize=10, color="#D0D1E0", pad=6)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)
    ax.grid(True)
    if i == 0:
        ax.legend(fontsize=8, facecolor="#1A1D2E", edgecolor="#3A3D55",
                  labelcolor="#E0E1EF")

plt.tight_layout()
save_path = OUT_DIR / "feature_distributions.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"✓ Saved feature_distributions.png")

# ── Plot 2: Correlation heatmap (sparse features) ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0F1117")
corr = df_feat[SPARSE_FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            linewidths=0.5, linecolor="#2E3048",
            annot_kws={"size": 9, "color": "#E0E1EF"},
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix (Sparse 8 Features)",
             fontsize=13, color="#E0E1EF", fontweight="bold", pad=12)
ax.set_xticklabels([FEATURE_LABELS[f] for f in SPARSE_FEATURES],
                   rotation=35, ha="right", fontsize=8, color="#C8C9D8")
ax.set_yticklabels([FEATURE_LABELS[f] for f in SPARSE_FEATURES],
                   rotation=0, fontsize=8, color="#C8C9D8")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_correlation.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved feature_correlation.png")

# ── Plot 3: Box-plots grouped by modal signal ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Feature Spread by Signal Modality", fontsize=14,
             color="#E0E1EF", fontweight="bold")

for ax, (grp, feats) in zip(axes, FEATURE_GROUPS.items()):
    data = []
    labels_plot = []
    colors_plot = []
    for feat in feats:
        for cls in [0, 1]:
            vals = df_feat[df_feat["medication"] == cls][feat].values
            # Normalize within feature for visual comparison
            vals_norm = (vals - vals.mean()) / (vals.std() + 1e-9)
            data.append(vals_norm)
            labels_plot.append(f"{feat}\n({'pre' if cls==0 else 'post'})")
            colors_plot.append(CLASS_COLORS[cls])

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("#8B8FA8")

    ax.set_title(f"{grp} Features", fontsize=11, color="#E0E1EF", pad=8)
    ax.set_xticks(range(1, len(labels_plot) + 1))
    ax.set_xticklabels(labels_plot, fontsize=7, rotation=15)
    ax.set_ylabel("Normalised value", fontsize=9, color="#C8C9D8")
    ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "feature_boxplots.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("✓ Saved feature_boxplots.png")

print("\n  Data exploration complete. All figures saved to outputs/figures/")
