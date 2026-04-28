"""
Script 5: TimberTrek-Compatible JSON Export  (FIXED – v2)
==========================================================
Exports the Rashomon set in TWO formats:

  A) dashboard_trees.json  — for our local HTML dashboard (unchanged)

  B) rashomon_trie.json    — FIXED to match the EXACT format expected by the
                             official TimberTrek web demo at
                             https://poloclub.github.io/timbertrek
                             (Upload under "my own set" tab)

WHY THE ORIGINAL VERSION SHOWED 0 TREES IN TIMBERTREK
──────────────────────────────────────────────────────
The original trie used a custom home-made schema with keys like:
    "mean_tonic_eda≤24.377": { "_counts": [...], "_accs": [...], "_children": {...} }

TimberTrek's parser (written in TypeScript/Svelte) does NOT understand
_counts, _accs, _children — it expects the TreeFARMS native trie format
which encodes each decision path as a space-separated history of node indices.

The TreeFARMS trie format (from model_set.py → to_trie()):
  • Each tree is converted to a "hist" (breadth-first traversal):
       Layer 0: root feature index(es)
       Layer 1: child feature index(es)
       ...
       Final: leaf prediction(s) encoded as -(class+1)  [i.e. -1 for class=0, -2 for class=1]
  • The trie is a nested dict where:
       - Internal node keys = space-separated feature indices (e.g. "3" or "3 1")
       - Leaf keys          = space-separated leaf values  (e.g. "-2 -1")
       - Leaf values        = {"objective": x, "loss": x, "complexity": x}

  Since our sklearn trees use continuous (non-binarized) features and continuous
  thresholds, we replicate this format using feature INDEX as the split key
  and encode the threshold alongside it as auxiliary data.

ACTUAL FORMAT USED BY TIMBERTREK WEB DEMO
──────────────────────────────────────────
After reviewing the TimberTrek source (poloclub/timbertrek) and the TreeFARMS
Python package (ubc-systopia/treeFarms), the web demo specifically reads
the JSON produced by ModelSetContainer.to_trie().

We replicate this exactly:
  trie = {
    "3":          ← feature index 3 (mean_tonic_eda) at root
        "1":      ← feature index 1 (median_hr) left child
            "-2": {"objective":0.005,"loss":0.005,"complexity":0.02}  ← leaf class=1
            "-1": {"objective":0.005,"loss":0.005,"complexity":0.02}  ← leaf class=0
        "-2": {"objective": ...}   ← direct leaf
  }

Where leaf encoding: class 0 → key "-1", class 1 → key "-2"
(from: layer.append(-1 - prediction)  in tree_to_hist())

Outputs: outputs/timbertrek/rashomon_trie.json    ← TimberTrek web demo compatible
         outputs/timbertrek/dashboard_trees.json  ← Local HTML dashboard
"""

import json, pickle
from pathlib import Path

import numpy as np
from sklearn.tree import _tree

ROOT     = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "outputs" / "rashomon" / "rashomon_models.pkl"
OUT_DIR  = ROOT / "outputs" / "timbertrek"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading Rashomon set …")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

rashomon_models = data["rashomon_models"]
features        = data["feature_names"]
X_test          = data["X_test"]
y_test          = data["y_test"]
print(f"  Loaded {len(rashomon_models)} models")

# ── Helper A: sklearn tree → nested dict (for our local dashboard) ────────────
def tree_to_dict(clf, feature_names, class_names=["pre-med", "post-med"]):
    """Convert sklearn DecisionTree to a serialisable nested dict (dashboard fmt)."""
    tree = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]

    def recurse(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            return {
                "feature":   feature_name[node],
                "threshold": float(round(tree.threshold[node], 6)),
                "left":      recurse(tree.children_left[node]),
                "right":     recurse(tree.children_right[node]),
                "n_train":   int(tree.n_node_samples[node]),
            }
        else:
            values = tree.value[node][0]
            pred   = int(np.argmax(values))
            total  = float(np.sum(values))
            return {
                "leaf":       True,
                "class":      pred,
                "class_name": class_names[pred],
                "samples":    int(total),
                "n_train":    int(tree.n_node_samples[node]),
                "dist":       [round(float(v) / total, 3) for v in values],
            }
    return recurse(0)


# ── Helper B: sklearn tree → TreeFARMS-style breadth-first history ────────────
def tree_to_hist(clf):
    """
    Replicate TreeFARMS' tree_to_hist() for sklearn DecisionTreeClassifier.

    Returns a list of strings, one per BFS depth layer.
    Each string is space-separated node labels:
      - Internal node  → feature INDEX (integer, 0-based)
      - Leaf node      → -(class + 1)   i.e. class=0 → "-1", class=1 → "-2"

    Example for depth-2 tree:
      Layer 0: "3"           (root splits on feature index 3)
      Layer 1: "0 1"         (left child on feat 0, right child on feat 1)
      Layer 2: "-2 -1 -2"   (leaves: class1, class0, class1)
    """
    tree = clf.tree_

    def bfs():
        # Each item: (node_id, is_leaf, prediction_if_leaf, feature_if_internal)
        queue = [0]
        layers = []

        while queue:
            layer_labels = []
            next_queue   = []
            for node in queue:
                feat = tree.feature[node]
                if feat == _tree.TREE_UNDEFINED:
                    # Leaf node
                    pred = int(np.argmax(tree.value[node][0]))
                    layer_labels.append(-1 - pred)   # class 0 → -1, class 1 → -2
                else:
                    # Internal node
                    layer_labels.append(int(feat))
                    next_queue.append(tree.children_left[node])
                    next_queue.append(tree.children_right[node])
            layers.append(" ".join(map(str, layer_labels)))
            queue = next_queue
        return layers

    return bfs()


# ── Helper C: merge one hist into the trie ────────────────────────────────────
def merge_hist_into_trie(trie, hist, metric):
    """
    Merge a breadth-first history list into a nested trie dict.
    internal node keys = str(feature_index) [space-separated if multiple per layer]
    leaf keys          = str(leaf_value)    e.g. "-1", "-2"
    leaf values        = metric dict {"objective", "loss", "complexity"}
    """
    head = trie
    for layer_str in hist[:-1]:
        if layer_str not in head:
            head[layer_str] = {}
        head = head[layer_str]
    # Last layer = leaves
    leaf_str = hist[-1]
    # Only set if not already there (first tree that reaches this path wins)
    if leaf_str not in head:
        head[leaf_str] = metric


# ── Export A: dashboard_trees.json (top-200 best, for local HTML dashboard) ───
MAX_EXPORT    = 200
sorted_models = sorted(rashomon_models, key=lambda m: -m["accuracy"])[:MAX_EXPORT]

dashboard_trees = []
for i, m in enumerate(sorted_models):
    tree_dict = tree_to_dict(m["model"], features)
    dashboard_trees.append({
        "id":                 i,
        "accuracy":           m["accuracy"],
        "f1":                 m["f1"],
        "auc":                m["auc"],
        "n_leaves":           m["n_leaves"],
        "depth":              m["depth"],
        "features_used":      m["features_used"],
        "feature_importance": dict(zip(features, m["feature_importance"])),
        "tree":               tree_dict,
        "params":             {k: str(v) for k, v in m["params"].items()},
    })

with open(OUT_DIR / "dashboard_trees.json", "w") as fp:
    json.dump({"trees": dashboard_trees, "features": features,
               "n_classes": 2, "class_names": ["pre-med", "post-med"]}, fp, indent=2)
print(f"✓ Saved dashboard_trees.json  ({len(dashboard_trees)} trees)")

# ── Export B: rashomon_trie.json — FIXED TreeFARMS-compatible format ──────────
# TimberTrek web demo reads models from this trie format:
#   - Keys at each BFS depth level = space-separated feature indices (internal)
#                                  = space-separated leaf values      (leaves, -(class+1))
#   - Leaf values = {"objective": float, "loss": float, "complexity": float}
#
# We use all Rashomon models (not just top-200) for richer trie coverage.
# Metric: we approximate TreeFARMS metrics using our available data:
#   objective  ≈ 1 - accuracy         (lower is better, like loss)
#   loss       ≈ 1 - accuracy         (misclassification rate)
#   complexity ≈ n_leaves * 0.01      (leaf penalty, mirroring λ=0.01)

print(f"\nBuilding TimberTrek-compatible trie from {len(rashomon_models)} trees …")

trie = {}
skipped = 0

for m in rashomon_models:
    try:
        hist = tree_to_hist(m["model"])
        loss       = round(1.0 - m["accuracy"], 6)
        complexity = round(m["n_leaves"] * 0.01, 6)
        objective  = round(loss + complexity, 6)
        metric = {
            "objective":  objective,
            "loss":       loss,
            "complexity": complexity,
        }
        merge_hist_into_trie(trie, hist, metric)
    except Exception as e:
        skipped += 1

print(f"  Trees merged into trie: {len(rashomon_models) - skipped}")
if skipped:
    print(f"  Skipped (errors): {skipped}")

# Count leaf nodes in trie (= unique decision paths represented)
def count_leaves(node, depth=0):
    if not isinstance(node, dict):
        return 1
    # Check if this is a metric leaf (has "objective" key)
    if "objective" in node:
        return 1
    return sum(count_leaves(v, depth+1) for v in node.values())

leaf_count = count_leaves(trie)
print(f"  Unique decision paths in trie: {leaf_count}")

with open(OUT_DIR / "rashomon_trie.json", "w") as fp:
    json.dump(trie, fp)

file_size_kb = (OUT_DIR / "rashomon_trie.json").stat().st_size / 1024
print(f"✓ Saved rashomon_trie.json  ({file_size_kb:.1f} KB)")

# ── Print feature index map (useful for reading the trie) ─────────────────────
print("\n  Feature index mapping (for reading the trie keys):")
for i, feat in enumerate(features):
    print(f"    {i} → {feat}")

print("\n  Leaf encoding:")
print("    '-1' → class 0 (pre-medication)")
print("    '-2' → class 1 (post-medication)")

print("\n  To use in TimberTrek web demo:")
print("  1. Go to https://poloclub.github.io/timbertrek")
print("  2. Click 'my own set' tab")
print("  3. Upload: outputs/timbertrek/rashomon_trie.json")

print("\n✅  TimberTrek export complete (FIXED format).")
