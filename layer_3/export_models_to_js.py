"""
layer_3/export_models_to_js.py
------------------------------
Compiles scikit-learn trained Random Forest models (Pipelines with StandardScaler + RandomForestClassifier)
into pure, zero-dependency, ultra-fast JavaScript decision trees.

Outputs:
  layer_3/compiled_models.js
"""

import os
import sys
import json
import joblib
import numpy as np

LAYER3_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(LAYER3_DIR)
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
JS_OUT_PATH = os.path.join(LAYER3_DIR, "compiled_models.js")


def serialize_tree(tree):
    """Convert a sklearn DecisionTree into a compact dictionary."""
    return {
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": [round(float(t), 6) for t in tree.threshold],
        # Normalize node values into class probability distributions
        "value": [
            [round(float(v), 6) for v in (val[0] / max(np.sum(val[0]), 1e-9))]
            for val in tree.value
        ],
    }


def export_pipeline_to_dict(pipeline_path, meta_path):
    if not os.path.exists(pipeline_path):
        return None

    pipe = joblib.load(pipeline_path)
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    scaler = pipe.named_steps.get("scaler", None)
    rf     = pipe.named_steps.get("rf", pipe.named_steps.get("clf", None))

    if rf is None and hasattr(pipe, "estimators_"):
        rf = pipe

    mean_vec  = scaler.mean_.tolist() if scaler is not None else []
    scale_vec = scaler.scale_.tolist() if scaler is not None else []

    trees = [serialize_tree(est.tree_) for est in rf.estimators_]

    return {
        "feature_names": meta.get("feature_names", []),
        "mean": [round(float(x), 6) for x in mean_vec],
        "scale": [round(float(x), 6) for x in scale_vec],
        "n_classes": rf.n_classes_ if hasattr(rf, "n_classes_") else 3,
        "n_trees": len(trees),
        "trees": trees,
    }


def main():
    print("=" * 60)
    print("  EXPORTING SCIKIT-LEARN MODELS TO PURE JAVASCRIPT")
    print("=" * 60)

    # 1. Export Path 1 (sample_model.pkl)
    p1_path = os.path.join(MODEL_DIR, "sample_model.pkl")
    p1_meta = os.path.join(MODEL_DIR, "sample_model_meta.pkl")
    p1_data = export_pipeline_to_dict(p1_path, p1_meta)
    print(f"Path 1 (sample_model): {p1_data['n_trees']} trees, {len(p1_data['feature_names'])} features")

    # 2. Export Path 2 (best_model.pkl)
    p2_path = os.path.join(MODEL_DIR, "best_model.pkl")
    p2_meta = os.path.join(MODEL_DIR, "model_meta.pkl")
    p2_data = export_pipeline_to_dict(p2_path, p2_meta)
    print(f"Path 2 (best_model)  : {p2_data['n_trees']} trees, {len(p2_data['feature_names'])} features")

    js_code = f"""/**
 * layer_3/compiled_models.js
 * --------------------------
 * Auto-generated JavaScript engine running the EXACT same trained Random Forest models
 * (StandardScaler + RandomForestClassifier) as the Raspberry Pi Python backend.
 */

const P1_MODEL_DATA = {json.dumps(p1_data)};
const P2_MODEL_DATA = {json.dumps(p2_data)};

function evalTree(tree, x) {{
  let node = 0;
  while (tree.feature[node] !== -2) {{
    const featIdx = tree.feature[node];
    const thresh  = tree.threshold[node];
    if (x[featIdx] <= thresh) {{
      node = tree.left[node];
    }} else {{
      node = tree.right[node];
    }}
  }}
  return tree.value[node];
}}

function predictForestProba(modelData, rawFeatureVector) {{
  // 1. Apply StandardScaler: (x - mean) / scale
  const scaled = new Array(rawFeatureVector.length);
  for (let i = 0; i < rawFeatureVector.length; i++) {{
    if (modelData.scale && modelData.scale[i]) {{
      scaled[i] = (rawFeatureVector[i] - modelData.mean[i]) / modelData.scale[i];
    }} else {{
      scaled[i] = rawFeatureVector[i];
    }}
  }}

  // 2. Average probability predictions across all trees
  const avgProba = new Array(modelData.n_classes).fill(0);
  const nTrees = modelData.trees.length;

  for (let t = 0; t < nTrees; t++) {{
    const treeProba = evalTree(modelData.trees[t], scaled);
    for (let c = 0; c < modelData.n_classes; c++) {{
      avgProba[c] += treeProba[c];
    }}
  }}

  for (let c = 0; c < modelData.n_classes; c++) {{
    avgProba[c] /= nTrees;
  }}

  let maxIdx = 0;
  for (let c = 1; c < modelData.n_classes; c++) {{
    if (avgProba[c] > avgProba[maxIdx]) maxIdx = c;
  }}

  return {{
    label: maxIdx,
    proba: avgProba,
    norm_prob: avgProba[0] || 0.0,
    near_prob: avgProba[1] || 0.0,
    crash_prob: avgProba[2] || 0.0,
  }};
}}

// Public API
function predictSampleRF_JS(sampleDict) {{
  const featNames = P1_MODEL_DATA.feature_names;
  const rawVec = featNames.map(f => (sampleDict[f] !== undefined ? sampleDict[f] : 0.0));
  return predictForestProba(P1_MODEL_DATA, rawVec);
}}

function predictWindowRF_JS(featsDict) {{
  const featNames = P2_MODEL_DATA.feature_names;
  const rawVec = featNames.map(f => (featsDict[f] !== undefined ? featsDict[f] : 0.0));
  return predictForestProba(P2_MODEL_DATA, rawVec);
}}

if (typeof module !== 'undefined' && module.exports) {{
  module.exports = {{
    P1_MODEL_DATA,
    P2_MODEL_DATA,
    predictSampleRF_JS,
    predictWindowRF_JS,
    predictForestProba
  }};
}}
"""

    with open(JS_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(js_code)

    print(f"\nSuccessfully compiled models to JS: {JS_OUT_PATH}")
    print(f"File size: {os.path.getsize(JS_OUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
