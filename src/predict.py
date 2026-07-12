"""
predict.py
----------
Live prediction pipeline for Smart Airbag Helmet.

Simulates streaming IMU sensor data → sliding window → model → prediction.
This is the demo-ready component that mimics real ESP32 data ingestion.

Usage:
    python src/predict.py                    # simulate with random data
    python src/predict.py --session 42       # replay a session from the dataset
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
#  LABEL CONFIG
# ─────────────────────────────────────────────
LABEL_MAP = {
    0: ("Normal",      "🟢", "#2ecc71"),
    1: ("Pothole",     "🟡", "#f1c40f"),
    2: ("SuddenBrake", "🟠", "#e67e22"),
    3: ("Crash",       "🔴", "#e74c3c"),
}

SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 20
STRIDE       = 10


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_dir: str = "../models"):
    """Load best_model.pkl + metadata."""
    model_path = os.path.join(model_dir, "best_model.pkl")
    meta_path  = os.path.join(model_dir, "model_meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}.\n"
            "Run train_model.py first to generate the model."
        )

    model = joblib.load(model_path)
    meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    print(f"✅ Loaded: {meta.get('model_name', 'Unknown')}")
    print(f"   Accuracy : {meta.get('accuracy', '?'):.4f}")
    print(f"   F1 macro : {meta.get('f1_macro', '?'):.4f}")
    print(f"   Features : {len(meta.get('feature_names', []))}")

    return model, meta


# ─────────────────────────────────────────────
#  SINGLE WINDOW PREDICTION
# ─────────────────────────────────────────────

def predict_window(model, window_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Given a 20-row window DataFrame, extract features and predict.

    Returns dict with label, probability, and feature vector.
    """
    from src.feature_engineering import extract_features_window

    feats = extract_features_window(window_df)
    feat_vec = np.array([[feats[f] for f in feature_names]])

    pred_label = int(model.predict(feat_vec)[0])

    # Get probability if model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(feat_vec)[0]
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf', None), 'predict_proba'):
        proba = model.predict_proba(feat_vec)[0]
    else:
        proba = np.zeros(4)
        proba[pred_label] = 1.0

    label_name, emoji, color = LABEL_MAP[pred_label]
    crash_prob = float(proba[3]) if len(proba) > 3 else 0.0

    return {
        "label"      : pred_label,
        "label_name" : label_name,
        "emoji"      : emoji,
        "proba"      : proba,
        "crash_prob" : crash_prob,
        "deploy"     : crash_prob > 0.65,   # deployment threshold
    }


# ─────────────────────────────────────────────
#  LIVE SIMULATOR (streaming)
# ─────────────────────────────────────────────

def run_live_simulation(
    model,
    meta: dict,
    source: str = "synthetic",
    dataset_path: str = None,
    session_id: int  = None,
    delay: float     = 0.05,
    verbose: bool    = True
):
    """
    Simulate streaming IMU data through the prediction pipeline.

    source: 'synthetic' → generate random data
            'replay'    → replay a session from CSV
    """
    feature_names = meta.get('feature_names', [])
    buffer = []   # Rolling window buffer

    print("\n" + "="*55)
    print("  LIVE PREDICTION SIMULATOR")
    print("  Pre-Impact Rider State Classifier")
    print("="*55)
    print(f"  Window size : {WINDOW_SIZE} samples")
    print(f"  Stride      : {STRIDE}")
    print(f"  Threshold   : crash_prob > 0.65 → DEPLOY")
    print("="*55 + "\n")

    if source == "replay" and dataset_path:
        df      = pd.read_csv(dataset_path)
        sess_df = df[df['session_id'] == session_id].reset_index(drop=True)
        rows    = [sess_df.iloc[i] for i in range(len(sess_df))]
        actual_labels = sess_df['label'].tolist()
        print(f"  Replaying session {session_id} ({len(rows)} timesteps)\n")
    else:
        # Synthetic stream: Normal with random crash events
        from src.data_generator import _normal, _crash, _pothole, _sudden_brake
        rng  = np.random.default_rng(777)
        n_ts = 200
        rows = []
        actual_labels = []
        # Build a scenario: normal → brake → crash
        for phase, (gen, lbl, n) in enumerate([
            (_normal, 0, 80), (_sudden_brake, 2, 40), (_crash, 3, 30), (_normal, 0, 50)
        ]):
            sig = gen(n, rng)
            for i in range(n):
                rows.append({col: sig[col][i] for col in SENSOR_COLS})
                actual_labels.append(lbl)
        print(f"  Simulated stream: Normal → SuddenBrake → Crash → Normal ({len(rows)} steps)\n")

    step         = 0
    deploy_fired = False

    for i, row_data in enumerate(rows):
        # Add to buffer
        if isinstance(row_data, dict):
            buffer.append(row_data)
        else:
            buffer.append({col: row_data[col] for col in SENSOR_COLS})

        # Only predict once we have a full window
        if len(buffer) >= WINDOW_SIZE and (len(buffer) - WINDOW_SIZE) % STRIDE == 0:
            window_df = pd.DataFrame(buffer[-WINDOW_SIZE:])
            result    = predict_window(model, window_df, feature_names)

            true_label = actual_labels[i] if i < len(actual_labels) else -1
            true_name  = LABEL_MAP.get(true_label, ('?', '?', '?'))[0]
            correct    = "✓" if result['label'] == true_label else "✗"

            crash_bar = "█" * int(result['crash_prob'] * 20) + "░" * (20 - int(result['crash_prob'] * 20))

            if verbose:
                print(f"  t={i:>3}  {result['emoji']} {result['label_name']:<13} "
                      f"| Crash: [{crash_bar}] {result['crash_prob']:.2%} "
                      f"| True: {true_name:<13} {correct}")

            # Airbag trigger
            if result['deploy'] and not deploy_fired:
                print(f"\n  🚨🚨🚨  AIRBAG DEPLOY TRIGGERED at t={i}  🚨🚨🚨")
                print(f"  Crash probability: {result['crash_prob']:.2%}")
                print(f"  (In hardware: ESP32 receives signal → servo → airbag)\n")
                deploy_fired = True

            step += 1
            time.sleep(delay)

    print("\n" + "="*55)
    print(f"  Simulation complete. Windows predicted: {step}")
    print("="*55)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Airbag Helmet — Live Predictor")
    parser.add_argument('--model-dir',   default='models',                          help='Path to models directory')
    parser.add_argument('--dataset',     default='data/synthetic/helmet_imu_raw.csv', help='Path to raw dataset')
    parser.add_argument('--session',     type=int, default=None,                    help='Session ID to replay')
    parser.add_argument('--delay',       type=float, default=0.02,                  help='Delay between steps (seconds)')
    parser.add_argument('--no-verbose',  action='store_true',                       help='Suppress per-step output')
    args = parser.parse_args()

    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    model, meta = load_model(args.model_dir)

    if args.session is not None and os.path.exists(args.dataset):
        run_live_simulation(model, meta,
                            source='replay',
                            dataset_path=args.dataset,
                            session_id=args.session,
                            delay=args.delay,
                            verbose=not args.no_verbose)
    else:
        run_live_simulation(model, meta,
                            source='synthetic',
                            delay=args.delay,
                            verbose=not args.no_verbose)
