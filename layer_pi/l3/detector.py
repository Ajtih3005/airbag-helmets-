"""
layer_3/detector.py
-------------------
Unified 3-Path Machine Learning Detection Engine for Raspberry Pi.
Coordinates:
  - PATH 1: Per-Sample Sentinel (sample_model.pkl, 0ms wait)
  - PATH 2: Window RF Classifier (best_model.pkl, 50ms window)
  - PATH 3: Black Box RAM Validation (60s rolling history)
  - Dual-Layer Arbiter & Safety Logic
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from collections import deque
import joblib
import numpy as np
import pandas as pd

# Path setup - l3/ is the package dir, parent is layer_pi/
import os, sys
L3_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(L3_DIR)           # layer_pi/
MODEL_DIR = os.path.join(ROOT_DIR, "models")

try:
    # Relative imports work when imported as a package (from l3.detector import ...)
    from .feature_engineering import extract_features_window, WINDOW_SIZE, STRIDE
    from .sentinel import Path1Sentinel
    from .arbiter import DualLayerArbiter, FastPhysicsTrigger
except ImportError:
    # Fallback: add l3/ to path for direct execution
    sys.path.insert(0, L3_DIR)
    from feature_engineering import extract_features_window, WINDOW_SIZE, STRIDE
    from sentinel import Path1Sentinel
    from arbiter import DualLayerArbiter, FastPhysicsTrigger

CRASH_PROB_THRESHOLD    = 0.70
CONSECUTIVE_WINDOWS_REQ = 3
BLACKBOX_SECONDS        = 60
SAMPLE_RATE_HZ          = 1000
BLACKBOX_MAX            = SAMPLE_RATE_HZ * BLACKBOX_SECONDS
BLACKBOX_MIN_RATIO      = 0.05
LABEL_NAME              = {0: "Normal", 1: "Near-Crash", 2: "CRASH"}


class Layer3Detector:
    """
    Main Detection Engine for Layer 3.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.sample_model = None
        self.sample_meta  = {}
        self.window_model = None
        self.window_meta  = {}
        
        self.load_models()

        self.sentinel = Path1Sentinel(
            self.sample_model,
            self.sample_meta.get("feature_names", ["ax","ay","az","gx","gy","gz","hg_ax","hg_ay","hg_az"])
        ) if self.sample_model else None

        self.window_features = self.window_meta.get("feature_names", [])

        self.window_buffer      = deque(maxlen=WINDOW_SIZE * 2)
        self.blackbox           = deque(maxlen=BLACKBOX_MAX)
        self.consecutive_crash  = 0
        self.deploy_fired       = False
        self.deploy_latency_ms  = 0.0
        self.samples_since_pred = 0
        self.total_samples      = 0
        self.total_windows      = 0

        self.arbiter = DualLayerArbiter(
            crash_prob_thresh=CRASH_PROB_THRESHOLD,
            consecutive_req=CONSECUTIVE_WINDOWS_REQ
        )
        self.physics_trigger = FastPhysicsTrigger()

    def load_models(self):
        # Look for models in parent dir (layer_pi/models)
        candidate_dirs = [
            os.path.join(ROOT_DIR, "models"),
            os.path.join(L3_DIR, "models"),
            self.model_dir
        ]
        target_dir = None
        for d in candidate_dirs:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "best_model.pkl")):
                target_dir = d
                break
        if not target_dir:
            target_dir = candidate_dirs[0]

        s_path = os.path.join(target_dir, "sample_model.pkl")
        sm_path = os.path.join(target_dir, "sample_model_meta.pkl")
        if os.path.exists(s_path):
            self.sample_model = joblib.load(s_path)
            self.sample_meta  = joblib.load(sm_path) if os.path.exists(sm_path) else {}
            if hasattr(self.sample_model, "set_params"):
                try: self.sample_model.set_params(rf__n_jobs=1)
                except: pass

        w_path = os.path.join(target_dir, "best_model.pkl")
        wm_path = os.path.join(target_dir, "model_meta.pkl")
        if os.path.exists(w_path):
            self.window_model = joblib.load(w_path)
            self.window_meta  = joblib.load(wm_path) if os.path.exists(wm_path) else {}
            if hasattr(self.window_model, "set_params"):
                try: self.window_model.set_params(rf__n_jobs=1)
                except: pass

        loaded = []
        if self.sample_model: loaded.append("Path 1 Sentinel")
        if self.window_model: loaded.append("Path 2 Random Forest")
        print(f"  [DETECTOR] Loaded Models from: {target_dir} -> {', '.join(loaded) if loaded else 'NONE FOUND!'}")

    def process_sample(self, sample: dict, current_time_s: float = None):
        """
        Process a single 1ms sensor reading through all 3 paths.
        Returns detailed decision packet.
        """
        self.total_samples += 1
        self.samples_since_pred += 1
        self.window_buffer.append(sample)
        
        t_now = current_time_s if current_time_s is not None else self.total_samples / 1000.0

        # --- PATH 1: Per-Sample Sentinel (0ms wait) ---------------------------
        p1_res = {"label": 0, "crash_prob": 0.0, "near_prob": 0.0, "det_score": 0.0, "trend": 0.0}
        if self.sentinel:
            p1_res = self.sentinel.predict_sample(sample)

        p1_label = p1_res["label"]
        p1_crash = p1_res["crash_prob"]
        det_score = p1_res["det_score"]
        trend = p1_res["trend"]

        # Window stride — how many samples between Path 2 evaluations (200ms)
        WIN_STRIDE = 200

        # Sentinel Gate Acceleration / Recovery Rule (Plan A specification)
        # Only boost gate when Sentinel actually predicts crash/near-crash AND trend is rising
        if p1_label in (1, 2) and (trend >= 0.20 or det_score >= 0.35):
            if self.consecutive_crash == 0:
                self.consecutive_crash = 2  # Boost gate to 2/3 — forces Path 2 evaluation next cycle
                self.samples_since_pred = WIN_STRIDE  # Trigger immediate Path 2 evaluation

        elif p1_label == 0 and trend <= -0.20 and self.consecutive_crash > 0:
            self.consecutive_crash = 0  # Recovery: sensor stabilized, reset gate

        # --- PATH 2: Window Classifier (50ms context) -------------------------
        p2_label = p2_crash = p2_near = None
        p2_evaluated = False

        if self.window_model and len(self.window_buffer) >= WINDOW_SIZE and self.samples_since_pred >= WIN_STRIDE:
            self.samples_since_pred = 0
            self.total_windows += 1
            p2_evaluated = True

            win_df = pd.DataFrame(list(self.window_buffer)[-WINDOW_SIZE:])
            feats = extract_features_window(win_df)
            # Pass named DataFrame so StandardScaler (fitted with feature names) doesn't warn
            feat_df = pd.DataFrame([[feats.get(f, 0.0) for f in self.window_features]],
                                   columns=self.window_features)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(self.window_model, "predict_proba"):
                    p2_proba = self.window_model.predict_proba(feat_df)[0]
                    while len(p2_proba) < 3:
                        p2_proba = np.append(p2_proba, 0.0)
                    p2_crash = float(p2_proba[2])
                    p2_near  = float(p2_proba[1])
                    p2_label = int(np.argmax(p2_proba))
                else:
                    p2_label = int(self.window_model.predict(feat_df)[0])
                    p2_crash = 1.0 if p2_label == 2 else 0.0
                    p2_near  = 1.0 if p2_label == 1 else 0.0

            if p2_crash >= CRASH_PROB_THRESHOLD:
                self.consecutive_crash += 1
            else:
                if not (p1_label == 2 and p1_crash >= CRASH_PROB_THRESHOLD):
                    self.consecutive_crash = 0

        # --- Fast Physics Check ---
        phys_result = self.physics_trigger.evaluate_sample(sample, t_now)

        # Store in Black Box RAM
        bb_entry = {
            **sample,
            "t_ms": self.total_samples,
            "p1_label": p1_label,
            "p1_crash": p1_crash,
            "p2_label": p2_label,
            "p2_crash": p2_crash,
        }
        self.blackbox.append(bb_entry)

        # --- PATH 3: Black Box Validation on Gate 3/3 --------------------------
        bb_ratio = 0.0
        new_deploy = False
        reason = ""

        if self.consecutive_crash >= CONSECUTIVE_WINDOWS_REQ and not self.deploy_fired:
            bb_preds = [e for e in self.blackbox if e.get("p1_label") is not None or e.get("p2_label") is not None]
            crash_nc = sum(1 for e in bb_preds if e.get("p1_label") in (1,2) or e.get("p2_label") in (1,2))
            bb_ratio = crash_nc / max(len(bb_preds), 1)

            if bb_ratio >= BLACKBOX_MIN_RATIO:
                self.deploy_fired = True
                new_deploy = True
                reason = "3-Path ML Confirmed Crash"
            else:
                self.consecutive_crash = 0
                reason = "False Positive suppressed by Black Box"

        return {
            "sample_num": self.total_samples,
            "t_ms": self.total_samples,
            "p1_label": p1_label,
            "p1_name": LABEL_NAME.get(p1_label, "Normal"),
            "p1_crash": p1_crash,
            "det_score": det_score,
            "trend": trend,
            "p2_evaluated": p2_evaluated,
            "p2_label": p2_label,
            "p2_name": LABEL_NAME.get(p2_label, "None") if p2_label is not None else None,
            "p2_crash": p2_crash,
            "gate_count": self.consecutive_crash,
            "blackbox_ratio": bb_ratio,
            "deploy_fired": self.deploy_fired,
            "new_deploy": new_deploy,
            "decision_reason": reason,
        }

    def reset(self):
        self.window_buffer.clear()
        self.blackbox.clear()
        self.consecutive_crash = 0
        self.deploy_fired = False
        self.total_samples = 0
        self.total_windows = 0
        self.samples_since_pred = 0
