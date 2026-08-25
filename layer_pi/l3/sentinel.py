"""
layer_3/sentinel.py
-------------------
Path 1: Per-Sample Sentinel & Deterioration Analyzer.
Tracks rolling 20ms trend to accelerate gate counters on rapid deterioration
and suppress false alarms on recovery.
"""

from collections import deque
import numpy as np
import pandas as pd


class DeteriorationAnalyzer:
    """
    Tracks Path-1 per-sample predictions over a rolling 20ms window 
    to detect if rider dynamics are deteriorating or recovering.
    """
    def __init__(self, history_len=20):
        self.history = deque(maxlen=history_len)

    def update(self, p1_label: int, crash_prob: float, near_prob: float):
        weight = 1.0 if p1_label == 2 else (0.5 if p1_label == 1 else 0.0)
        self.history.append(weight)

        n = len(self.history)
        det_score = sum(self.history) / n

        if n >= 4:
            half = n // 2
            recent_avg = sum(list(self.history)[half:]) / (n - half)
            older_avg  = sum(list(self.history)[:half]) / half
            trend = recent_avg - older_avg
        else:
            trend = 0.0

        return float(det_score), float(trend)


class Path1Sentinel:
    """
    Fast Path 1 Sentinel: Evaluates dynamic motion with model prediction when motion deviates.
    """
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.analyzer = DeteriorationAnalyzer(history_len=20)
        self._last_result = {"label": 0, "crash_prob": 0.0, "near_prob": 0.0, "norm_prob": 1.0, "det_score": 0.0, "trend": 0.0}
        self._sample_count = 0

    def predict_sample(self, sample_dict: dict):
        self._sample_count += 1
        
        # Fast physics gate: check if motion is stable normal cruising (g ~ 1.0, low gyro)
        ax, ay, az = sample_dict.get("ax", 0.0), sample_dict.get("ay", 0.0), sample_dict.get("az", 9.81)
        gx, gy, gz = sample_dict.get("gx", 0.0), sample_dict.get("gy", 0.0), sample_dict.get("gz", 0.0)
        hg_az      = sample_dict.get("hg_az", az)
        
        total_g  = (ax**2 + ay**2 + az**2)**0.5 / 9.81
        gyro_mag = (gx**2 + gy**2 + gz**2)**0.5
        impact_g = abs(hg_az) / 9.81

        # Fast path: if motion is tranquil normal (0.8g to 1.3g, <25 deg/s gyro), label is Normal (0)
        if 0.75 <= total_g <= 1.35 and gyro_mag < 25.0 and impact_g < 2.0:
            det_score, trend = self.analyzer.update(0, 0.0, 0.0)
            return {
                "label": 0, "crash_prob": 0.0, "near_prob": 0.0, "norm_prob": 1.0,
                "det_score": det_score, "trend": trend
            }

        # Dynamic motion: evaluate model (every 5 samples or on high impact)
        if self._sample_count % 5 == 0 or impact_g > 3.0 or gyro_mag > 60.0:
            row = np.array([[sample_dict.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(row)[0]
                while len(proba) < 3: proba = np.append(proba, 0.0)
                crash_p = float(proba[2])
                near_p  = float(proba[1])
                norm_p  = float(proba[0])
                label   = 2 if crash_p > 0.50 else (1 if near_p > 0.55 else 0)
            else:
                label = int(self.model.predict(row)[0])
                crash_p = 1.0 if label == 2 else 0.0
                near_p  = 1.0 if label == 1 else 0.0
                norm_p  = 1.0 if label == 0 else 0.0
        else:
            # Re-use last probability with current physics
            crash_p = self._last_result["crash_prob"]
            near_p  = self._last_result["near_prob"]
            norm_p  = self._last_result["norm_prob"]
            label   = self._last_result["label"]

        # High impact override
        if impact_g > 15.0 or gyro_mag > 250.0:
            label = 2
            crash_p = max(crash_p, 0.95)

        det_score, trend = self.analyzer.update(label, crash_p, near_p)
        res = {
            "label": label,
            "crash_prob": crash_p,
            "near_prob": near_p,
            "norm_prob": norm_p,
            "det_score": det_score,
            "trend": trend,
        }
        self._last_result = res
        return res
