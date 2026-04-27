# Rashomon Sets of Sparse Decision Trees for Biosignal Classification

This repository implements the concept of **Rashomon Sets** for interpretable machine learning, specifically applied to classifying physiological biosignal data (ECG, EDA, and Skin Temperature).

Based on the core ideas from [TreeFARMS (NeurIPS '22)](https://arxiv.org/abs/2209.08040) and [Rashomon Importance Distribution (RID)](https://github.com/jdonnelly36/Rashomon_Importance_Distribution), this project enumerates over 4,100 near-optimal, sparse decision trees to highlight feature stability rather than relying on a single "optimal" model.

---

## Project Overview

When training models on tabular data, there often isn't just one "best" model. Instead, there is a set of models—the **Rashomon Set**—that all perform similarly well (within a margin $\epsilon$ of the best accuracy) but might rely on completely different logic and features.

In this project, we analyze physiological data for a participant (P19_S1) to classify **Pre-Medication vs. Post-Medication** states using 8 sparse, interpretable features.

### Key Outputs:
- **Decision Tree Rashomon Set**: A collection of 4,171 highly accurate but shallow decision trees.
- **Rashomon Importance Distribution (RID)**: Statistical distributions showing the stability of each feature across the entire Rashomon Set.
- **Interactive Dashboard**: A local HTML dashboard to visually explore the tree subset and metrics.
- **TimberTrek Export**: JSON files generated for visualization in the interactive [TimberTrek tool](https://poloclub.github.io/timbertrek).

---

## Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
A single runner script executes all 5 stages of the pipeline: Exploration, Rashomon Enumeration, RID Calculation, Random Forest Comparison, and JSON Exporter.

```bash
python3 run_all.py
```
*This takes a few minutes as it explores 4,500 model configurations using Stratified Shuffle Split cross-validation.*

### 3. View the Interactive Dashboard
The project includes a premium HTML dashboard to interact with the results. Since it loads local JSON files, you must run a simple HTTP server:

```bash
python3 -m http.server 8765
```
Then, open your web browser and navigate to: **[http://localhost:8765/dashboard.html](http://localhost:8765/dashboard.html)**

---

## Repository Structure

```
├── Data/                 # Excluded from git; contains raw 'P19_S1_multimodel.csv'
├── Instructions/         # Background information and variable definitions
├── Src/
│   ├── 1_data_exploration.py       # KDE distributions, correlations & baseline stats
│   ├── 2_rashomon_set_trees.py     # Grid sweeper enforcing the ε-threshold constraint
│   ├── 3_feature_importance_rid.py # Calculates marginal drop & CV instability (RID)
│   ├── 4_rf_vs_dt_comparison.py    # Builds & compares RF Ensembles vs. DTs
│   └── 5_timbertrek_export.py      # Serializes tree parameters & creates visual JSON trie
├── outputs/              # Auto-generated directory
│   ├── figures/          # PNG charts and diagram
│   ├── rashomon/         # CSV Metadata, JSON summaries, and Model Pickle
│   └── timbertrek/       # Export files for the JavaScript visualizers
├── run_all.py            # Entry point for pipeline
└── dashboard.html        # 7-Tab Interactive Presentation Frontend
```

---

## Key Findings (P19 S1 Data)

1. **Massive Multiplicity:** Even constrained to sparse features, there are **4,171 different trees** that achieve $>94.5\%$ test accuracy on the holdout data.
2. **Dominant Features:** The RID marginal drop analysis proves that `mean_tonic_eda` is absolutely vital. It appears in 90.7% of all valid Rashomon models. Removing it causes a catastrophic drop in accuracy across the board.
3. **Redundant Features:** HRV measures like `rmssd` barely feature in the Rashomon set (used in $<10\%$ of trees).
4. **Decision Trees vs. Random Forests:** While both achieve near $100\%$ accuracy, the Random Forest Rashomon sets consist of opaque ensembles, whereas the Decision Tree set offers thousands of clinically interpretable, shallow rule-sets (Depth $\le 4$).

---

## TimberTrek Integration

If you wish to use the official TimberTrek visualizer instead of the local dashboard:
1. Run the pipeline to generate `./outputs/timbertrek/rashomon_trie.json`.
2. Go to [https://poloclub.github.io/timbertrek/](https://poloclub.github.io/timbertrek/).
3. Upload the generated JSON trie file to visually explore and filter the tree space.
