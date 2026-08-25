"""
03_ml_pipeline/ml_inference_engine.py
---------------------------------------
Machine Learning Inference Engine.
Loads trained Random Forest/XGBoost model and feature scaler, evaluates window features.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging

log = logging.getLogger("ml_engine")

class MLInferenceEngine:
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "best_model.pkl")
        self.meta_path = os.path.join(model_dir, "model_meta.pkl")
        self.model = None
        self.meta = {}

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.meta_path):
                    self.meta = joblib.load(self.meta_path)
                log.info(f"[ML] Model loaded successfully from {self.model_path}")
            except Exception as e:
                log.error(f"[ML ERROR] Failed to load model: {e}")
        else:
            log.warning(f"[ML WARNING] Model file not found at {self.model_path}. ML inference will yield 0.0 proba.")

    def predict_window(self, feature_dict):
        """
        Takes extracted 22-dimensional feature dict.
        Returns: (pred_label: int, crash_prob: float, near_crash_prob: float)
        Labels: 0 = Normal, 1 = Near-Crash, 2 = Crash
        """
        if not self.model:
            return 0, 0.0, 0.0

        try:
            feat_names = self.meta.get("feature_names", list(feature_dict.keys()))
            feat_df = pd.DataFrame([[feature_dict.get(f, 0.0) for f in feat_names]], columns=feat_names)
            
            pred_label = int(self.model.predict(feat_df)[0])
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(feat_df)[0]
                crash_prob = float(proba[2]) if len(proba) > 2 else (1.0 if pred_label == 2 else 0.0)
                nc_prob = float(proba[1]) if len(proba) > 1 else 0.0
            else:
                crash_prob = 1.0 if pred_label == 2 else 0.0
                nc_prob = 1.0 if pred_label == 1 else 0.0

            return pred_label, crash_prob, nc_prob
        except Exception as e:
            log.error(f"[ML INFERENCE ERROR] {e}")
            return 0, 0.0, 0.0
