"""
feature_engineering.py
-----------------------
Sliding window + feature extraction for Smart Airbag Helmet project.

Pipeline:
    Raw time-series (session_id, timestamp, ax..gz, label)
        ↓
    Sliding window over each session
        ↓
    Feature vector per window (mean, std, max, min, magnitude, jerk, tilt, etc.)
        ↓
    Majority-vote label per window
        ↓
    Flat feature matrix ready for ML
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 20     # timesteps per window
STRIDE       = 10     # step between windows


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION (single window)
# ─────────────────────────────────────────────

def extract_features_window(window: pd.DataFrame) -> dict:
    """
    Extract feature vector from a single window DataFrame.

    Features extracted:
        Per-axis (ax,ay,az,gx,gy,gz):
            mean, std, max, min, range, variance, rms, iqr, skewness
        Derived:
            accel_magnitude (mean over window)
            gyro_magnitude  (mean over window)
            sma             (Signal Magnitude Area)
            jerk_mean       (mean of |Δaccel| per step)
            jerk_max        (max  of |Δaccel|)
            tilt_angle      (mean tilt from az/accel_mag)
            pitch_mean      (mean gy)
            roll_mean       (mean gx)
    """
    feats = {}

    ax = window['ax'].values
    ay = window['ay'].values
    az = window['az'].values
    gx = window['gx'].values
    gy = window['gy'].values
    gz = window['gz'].values

    # ── Per-axis stats ──────────────────────────────────────────────
    for name, col in zip(['ax','ay','az','gx','gy','gz'],
                         [ax, ay, az, gx, gy, gz]):
        feats[f'{name}_mean']   = np.mean(col)
        feats[f'{name}_std']    = np.std(col)
        feats[f'{name}_max']    = np.max(col)
        feats[f'{name}_min']    = np.min(col)
        feats[f'{name}_range']  = np.max(col) - np.min(col)
        feats[f'{name}_var']    = np.var(col)
        feats[f'{name}_rms']    = np.sqrt(np.mean(col ** 2))
        q75, q25 = np.percentile(col, [75, 25])
        feats[f'{name}_iqr']    = q75 - q25
        # Skewness (manual — avoid scipy dependency for now)
        if np.std(col) > 1e-8:
            feats[f'{name}_skew'] = np.mean(((col - np.mean(col)) / np.std(col)) ** 3)
        else:
            feats[f'{name}_skew'] = 0.0

    # ── Derived: Magnitudes ─────────────────────────────────────────
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)

    feats['accel_mag_mean'] = np.mean(accel_mag)
    feats['accel_mag_std']  = np.std(accel_mag)
    feats['accel_mag_max']  = np.max(accel_mag)

    feats['gyro_mag_mean']  = np.mean(gyro_mag)
    feats['gyro_mag_std']   = np.std(gyro_mag)
    feats['gyro_mag_max']   = np.max(gyro_mag)

    # ── Signal Magnitude Area (SMA) ─────────────────────────────────
    # SMA = (1/N) * Σ(|ax| + |ay| + |az|)
    feats['sma_accel'] = np.mean(np.abs(ax) + np.abs(ay) + np.abs(az))
    feats['sma_gyro']  = np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz))

    # ── Jerk (rate of change of acceleration) ───────────────────────
    # Jerk captures the "sharpness" of an impact — key for crash detection
    jerk_x = np.diff(ax)
    jerk_y = np.diff(ay)
    jerk_z = np.diff(az)
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

    feats['jerk_mean'] = np.mean(jerk_mag) if len(jerk_mag) > 0 else 0.0
    feats['jerk_max']  = np.max(jerk_mag)  if len(jerk_mag) > 0 else 0.0
    feats['jerk_std']  = np.std(jerk_mag)  if len(jerk_mag) > 0 else 0.0

    # ── Tilt Angle ──────────────────────────────────────────────────
    # Tilt = arccos(az / accel_mag) — angle from vertical
    # Useful to detect fall/lean events
    safe_mag = np.where(accel_mag > 1e-6, accel_mag, 1e-6)
    cos_tilt  = np.clip(az / safe_mag, -1.0, 1.0)
    tilt_rad  = np.arccos(cos_tilt)
    feats['tilt_mean_deg'] = np.degrees(np.mean(tilt_rad))
    feats['tilt_max_deg']  = np.degrees(np.max(tilt_rad))
    feats['tilt_std_deg']  = np.degrees(np.std(tilt_rad))

    # ── Cross-axis correlation features ────────────────────────────
    # Crash: all axes correlated due to chaotic motion
    if np.std(ax) > 1e-8 and np.std(ay) > 1e-8:
        feats['corr_ax_ay'] = float(np.corrcoef(ax, ay)[0, 1])
    else:
        feats['corr_ax_ay'] = 0.0

    if np.std(ax) > 1e-8 and np.std(az) > 1e-8:
        feats['corr_ax_az'] = float(np.corrcoef(ax, az)[0, 1])
    else:
        feats['corr_ax_az'] = 0.0

    return feats


# ─────────────────────────────────────────────
#  SLIDING WINDOW OVER FULL DATASET
# ─────────────────────────────────────────────

def apply_sliding_window(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int      = STRIDE,
    verbose: bool    = True
) -> pd.DataFrame:
    """
    Apply sliding window over each session and extract features.

    Parameters
    ----------
    df          : Raw dataset with columns [session_id, timestamp, ax..gz, label]
    window_size : Number of timesteps per window
    stride      : Step between consecutive windows
    verbose     : Print progress

    Returns
    -------
    feature_df : Flat DataFrame, one row per window
                 Columns: all extracted features + 'label' (majority vote)
    """
    all_features = []
    sessions = df['session_id'].unique()

    for sess_id in sessions:
        sess_df = df[df['session_id'] == sess_id].reset_index(drop=True)
        n = len(sess_df)

        for start in range(0, n - window_size + 1, stride):
            end    = start + window_size
            window = sess_df.iloc[start:end]

            # Extract features
            feats = extract_features_window(window)

            # Label = majority vote within window
            majority_label = int(window['label'].value_counts().idxmax())
            feats['label']      = majority_label
            feats['session_id'] = int(sess_id)

            all_features.append(feats)

    feature_df = pd.DataFrame(all_features)

    if verbose:
        print(f"Windows extracted  : {len(feature_df)}")
        print(f"Features per window: {feature_df.shape[1] - 2}")  # exclude label, session_id
        print(f"\nWindow label distribution:")
        counts = feature_df['label'].value_counts().sort_index()
        label_map = {0: 'Normal', 1: 'Pothole', 2: 'SuddenBrake', 3: 'Crash'}
        for lbl, cnt in counts.items():
            print(f"  {label_map[lbl]:<15} {cnt:>5}")

    return feature_df


# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT (session-aware)
# ─────────────────────────────────────────────

def split_dataset(
    feature_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Session-aware train/test split.

    We split by session_id (not by rows) to avoid data leakage —
    consecutive windows from the same session are highly correlated.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import GroupShuffleSplit

    feature_cols = [c for c in feature_df.columns if c not in ('label', 'session_id')]
    X = feature_df[feature_cols]
    y = feature_df['label']
    groups = feature_df['session_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    raw_path  = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "helmet_imu_raw.csv")
    feat_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")

    print("Loading raw data...")
    df = pd.read_csv(raw_path)

    print("Applying sliding window...")
    feature_df = apply_sliding_window(df, window_size=20, stride=10, verbose=True)

    feature_df.to_csv(feat_path, index=False)
    print(f"\nSaved features to: {feat_path}")
