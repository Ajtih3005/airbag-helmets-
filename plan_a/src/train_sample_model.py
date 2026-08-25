"""
train_sample_model.py
----------------------
Trains a FAST per-sample ML classifier for the Smart Airbag Helmet.

This is Path-1 (Immediate ML) — trained on raw single-sample values with
NO windowing. It runs on every new sensor reading (1ms), giving an immediate
prediction before the 50ms sliding window is even full.

Input features (9):
    ax, ay, az      - MPU6050 accelerometer (low-g)
    gx, gy, gz      - MPU6050 gyroscope
    hg_ax, hg_ay, hg_az - ADXL377 high-g accelerometer

Output:
    models/sample_model.pkl      - trained classifier (Pipeline: Scaler + RF)
    models/sample_model_meta.pkl - metadata (feature names, label map)

Usage:
    python src/train_sample_model.py
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics         import accuracy_score, classification_report, f1_score

# -------------------------------------------------------
#  PATHS
# -------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH     = os.path.join(project_root, "data", "synthetic", "helmet_imu_raw.csv")
MODEL_DIR    = os.path.join(project_root, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "sample_model.pkl")
META_PATH    = os.path.join(MODEL_DIR, "sample_model_meta.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
#  FEATURE & LABEL SETUP
# -------------------------------------------------------
SAMPLE_FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "hg_ax", "hg_ay", "hg_az"]
LABEL_NAMES     = {0: "Normal", 1: "Near-Crash", 2: "Crash"}

# -------------------------------------------------------
#  LOAD DATA
# -------------------------------------------------------
print("=" * 55)
print("  FAST PER-SAMPLE ML MODEL TRAINER")
print("=" * 55)
print(f"\nLoading raw dataset: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"  Total samples : {len(df):,}")
print(f"  Sessions      : {df['session_id'].nunique()}")
print(f"\nLabel distribution:")
for lbl, name in LABEL_NAMES.items():
    cnt = (df["label"] == lbl).sum()
    pct = cnt / len(df) * 100
    print(f"  {name:<15} {cnt:>8,}  ({pct:.1f}%)")

# -------------------------------------------------------
#  FEATURES & LABELS
# -------------------------------------------------------
X      = df[SAMPLE_FEATURES]
y      = df["label"]
groups = df["session_id"]

# -------------------------------------------------------
#  SESSION-AWARE TRAIN / TEST SPLIT (no data leakage)
# -------------------------------------------------------
print("\nSplitting by session (no data leakage)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"  Train samples : {len(X_train):,}")
print(f"  Test samples  : {len(X_test):,}")

# -------------------------------------------------------
#  BUILD & TRAIN MODEL (Pipeline: StandardScaler + RandomForest)
# -------------------------------------------------------
print("\nTraining RandomForest on raw 9-feature samples...")
t0 = time.time()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators  = 100,    # lighter than the window model (faster inference)
        max_depth     = 10,     # shallower — fewer features to split on
        min_samples_leaf = 5,
        n_jobs        = -1,
        random_state  = 42,
        class_weight  = "balanced",  # compensate for Normal dominance
    )),
])

model.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"  Training done in {elapsed:.1f}s")

# -------------------------------------------------------
#  EVALUATE
# -------------------------------------------------------
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="macro")

print(f"\nTest Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro F1      : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values())))

# -------------------------------------------------------
#  SAVE MODEL
# -------------------------------------------------------
joblib.dump(model, MODEL_PATH)
meta = {
    "feature_names"  : SAMPLE_FEATURES,
    "label_names"    : LABEL_NAMES,
    "model_type"     : "per_sample_rf",
    "n_features"     : len(SAMPLE_FEATURES),
    "accuracy"       : float(acc),
    "macro_f1"       : float(f1),
}
joblib.dump(meta, META_PATH)

print(f"\nModel saved  -> {MODEL_PATH}")
print(f"Meta saved   -> {META_PATH}")
print("=" * 55)
print("  Done. Run raspberry_pi_interface.py to use both models.")
print("=" * 55)
