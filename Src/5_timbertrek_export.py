"""
Script 5: TimberTrek-Compatible JSON Export
===========================================
Exports the Rashomon set of decision trees as a JSON file in the
format expected by the TimberTrek interactive visualiser.
https://github.com/poloclub/timbertrek

The trie structure:
{
  "trie": { ... nested feature→threshold→class trie ... },
  "features": ["f0", "f1", ...],
  "num_class": 2,
  "accuracy": 0.9,
  ...
}

Also exports a simplified JSON that can be loaded in the HTML dashboard.

Outputs: outputs/timbertrek/rashomon_trie.json
         outputs/timbertrek/dashboard_trees.json
"""

import json, pickle
from pathlib import Path
from collections import defaultdict

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

# ── Helper: extract tree as nested rules ─────────────────────────────────────
def tree_to_dict(clf, feature_names, class_names=["pre-med", "post-med"]):
    """Convert sklearn DecisionTree to a serialisable nested dict."""
    tree = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]

    def recurse(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name      = feature_name[node]
            threshold = float(round(tree.threshold[node], 6))
            left_child  = tree.children_left[node]
            right_child = tree.children_right[node]
            return {
                "feature":    name,
                "threshold":  threshold,
                "left":       recurse(left_child),   # ≤ threshold
                "right":      recurse(right_child),  # > threshold
                "n_train":    int(tree.n_node_samples[node]),
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
                "dist":       [round(float(v)/total, 3) for v in values],
            }

    return recurse(0)


# ── Export up to top-200 Rashomon trees (file size constraint) ───────────────
MAX_EXPORT = 200
sorted_models = sorted(rashomon_models, key=lambda m: -m["accuracy"])[:MAX_EXPORT]

dashboard_trees = []
for i, m in enumerate(sorted_models):
    tree_dict = tree_to_dict(m["model"], features)
    dashboard_trees.append({
        "id":               i,
        "accuracy":         m["accuracy"],
        "f1":               m["f1"],
        "auc":              m["auc"],
        "n_leaves":         m["n_leaves"],
        "depth":            m["depth"],
        "features_used":    m["features_used"],
        "feature_importance": dict(zip(features, m["feature_importance"])),
        "tree":             tree_dict,
        "params": {
            k: str(v) for k, v in m["params"].items()
        },
    })

with open(OUT_DIR / "dashboard_trees.json", "w") as fp:
    json.dump({"trees": dashboard_trees, "features": features,
               "n_classes": 2, "class_names": ["pre-med", "post-med"]}, fp, indent=2)
print(f"✓ Saved dashboard_trees.json  ({len(dashboard_trees)} trees)")

# ── Build a simple trie for TimberTrek-like visualisation ────────────────────
def merge_into_trie(trie_node, tree_node, acc):
    """Recursively merge tree_node into a trie_node."""
    if tree_node.get("leaf"):
        cls = tree_node["class"]
        key = f"LEAF_{cls}"
        if key not in trie_node:
            trie_node[key] = {"_counts": [0, 0], "_accs": []}
        trie_node[key]["_counts"][cls] += 1
        trie_node[key]["_accs"].append(acc)
        return
    feat  = tree_node["feature"]
    thr   = str(round(tree_node["threshold"], 3))
    key   = f"{feat}≤{thr}"
    if key not in trie_node:
        trie_node[key] = {"_counts": [0, 0], "_accs": [], "_children": {}}
    trie_node[key]["_counts"][0] += 1
    trie_node[key]["_accs"].append(acc)
    merge_into_trie(trie_node[key]["_children"], tree_node["left"],  acc)
    merge_into_trie(trie_node[key]["_children"], tree_node["right"], acc)

trie = {}
for t in dashboard_trees:
    merge_into_trie(trie, t["tree"], t["accuracy"])

with open(OUT_DIR / "rashomon_trie.json", "w") as fp:
    json.dump(trie, fp)
print("✓ Saved rashomon_trie.json (merged trie)")

print("\n✅  TimberTrek export complete.")
