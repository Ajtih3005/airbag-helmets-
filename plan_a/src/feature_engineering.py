"""
feature_engineering.py
-----------------------
Sliding window + feature extraction for Smart Airbag Helmet project.

Timing (aligned with spec):
    SAMPLE_RATE_HZ = 1000   (1 kHz)
    WINDOW_SIZE    = 200    (200 ms at 1000 Hz)
    STRIDE         = 100    (100 ms stride = 50% overlap)

Sensors:
    MPU6050  : ax, ay, az, gx, gy, gz
    ADXL377  : hg_ax, hg_ay, hg_az  (high-g channel)

Labels (3-class):
    0 - Normal
    1 - Near-Crash
    2 - Crash
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Import timing constants from data_generator (single source of truth)
try:
    from src.data_generator import SAMPLE_RATE_HZ, WINDOW_SIZE, STRIDE, WINDOW_MS
except ImportError:
    try:
        from data_generator import SAMPLE_RATE_HZ, WINDOW_SIZE, STRIDE, WINDOW_MS
    except ImportError:
        SAMPLE_RATE_HZ = 1000
        WINDOW_MS      = 200
        WINDOW_SIZE    = 200
        STRIDE         = 100

SENSOR_COLS    = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
HG_SENSOR_COLS = ['hg_ax', 'hg_ay', 'hg_az']
ALL_SENSOR_COLS = SENSOR_COLS + HG_SENSOR_COLS

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "Crash"}


# -------------------------------------------------
#  FEATURE EXTRACTION (single window)
# -------------------------------------------------

def extract_features_window(window: pd.DataFrame) -> dict:
    """
    Extract feature vector from a single 200-sample window.

    Features:
        Per MPU6050 axis (ax,ay,az,gx,gy,gz):
            mean, std, max, min, range, rms, iqr, skewness
        Per ADXL377 axis (hg_ax, hg_ay, hg_az):
            mean, std, max, rms            (high-g key for crash discrimination)
        Derived:
            accel_mag_*   (mean/std/max of sqrt(ax^2+ay^2+az^2))
            gyro_mag_*    (mean/std/max of sqrt(gx^2+gy^2+gz^2))
            hg_mag_*      (mean/std/max of ADXL377 vector magnitude)
            sma_accel     (signal magnitude area)
            sma_gyro
            sma_hg        (high-g SMA -- big spike in crash)
            jerk_mean/max/std  (rate of change of acceleration)
            tilt_mean/max/std  (angle from vertical in degrees)
            corr_ax_ay, corr_ax_az  (cross-axis correlations)
            hg_peak_ratio  (ADXL377 max / MPU6050 max -- crash discriminator)
    """
    feats = {}

    ax    = window['ax'].values
    ay    = window['ay'].values
    az    = window['az'].values
    gx    = window['gx'].values
    gy    = window['gy'].values
    gz    = window['gz'].values

    # ADXL377 columns (may not exist in old datasets — graceful fallback)
    has_hg = all(c in window.columns for c in HG_SENSOR_COLS)
    if has_hg:
        hg_ax = window['hg_ax'].values
        hg_ay = window['hg_ay'].values
        hg_az = window['hg_az'].values
    else:
        hg_ax = ax.copy()
        hg_ay = ay.copy()
        hg_az = az.copy()

    # ---- Per-axis stats: MPU6050 ----------------------------------------
    for name, col in zip(['ax','ay','az','gx','gy','gz'],
                         [ax, ay, az, gx, gy, gz]):
        feats[f'{name}_mean']  = np.mean(col)
        feats[f'{name}_std']   = np.std(col)
        feats[f'{name}_max']   = np.max(col)
        feats[f'{name}_min']   = np.min(col)
        feats[f'{name}_range'] = np.max(col) - np.min(col)
        feats[f'{name}_rms']   = np.sqrt(np.mean(col ** 2))
        q75, q25 = np.percentile(col, [75, 25])
        feats[f'{name}_iqr']   = q75 - q25
        if np.std(col) > 1e-8:
            feats[f'{name}_skew'] = float(np.mean(((col - np.mean(col)) / np.std(col)) ** 3))
        else:
            feats[f'{name}_skew'] = 0.0

    # ---- Per-axis stats: ADXL377 (high-g) ---------------------------------
    for name, col in zip(['hg_ax','hg_ay','hg_az'], [hg_ax, hg_ay, hg_az]):
        feats[f'{name}_mean'] = np.mean(col)
        feats[f'{name}_std']  = np.std(col)
        feats[f'{name}_max']  = np.max(np.abs(col))
        feats[f'{name}_rms']  = np.sqrt(np.mean(col ** 2))

    # ---- Derived: Magnitudes -----------------------------------------------
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
    hg_mag    = np.sqrt(hg_ax**2 + hg_ay**2 + hg_az**2)

    feats['accel_mag_mean'] = np.mean(accel_mag)
    feats['accel_mag_std']  = np.std(accel_mag)
    feats['accel_mag_max']  = np.max(accel_mag)

    feats['gyro_mag_mean']  = np.mean(gyro_mag)
    feats['gyro_mag_std']   = np.std(gyro_mag)
    feats['gyro_mag_max']   = np.max(gyro_mag)

    feats['hg_mag_mean']    = np.mean(hg_mag)
    feats['hg_mag_std']     = np.std(hg_mag)
    feats['hg_mag_max']     = np.max(hg_mag)

    # ---- Signal Magnitude Area ---------------------------------------------
    feats['sma_accel'] = np.mean(np.abs(ax) + np.abs(ay) + np.abs(az))
    feats['sma_gyro']  = np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz))
    feats['sma_hg']    = np.mean(np.abs(hg_ax) + np.abs(hg_ay) + np.abs(hg_az))

    # ---- Jerk (rate of change of acceleration) ----------------------------
    jerk_x   = np.diff(ax)
    jerk_y   = np.diff(ay)
    jerk_z   = np.diff(az)
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

    feats['jerk_mean'] = float(np.mean(jerk_mag)) if len(jerk_mag) > 0 else 0.0
    feats['jerk_max']  = float(np.max(jerk_mag))  if len(jerk_mag) > 0 else 0.0
    feats['jerk_std']  = float(np.std(jerk_mag))  if len(jerk_mag) > 0 else 0.0

    # ---- Tilt Angle --------------------------------------------------------
    safe_mag = np.where(accel_mag > 1e-6, accel_mag, 1e-6)
    cos_tilt = np.clip(az / safe_mag, -1.0, 1.0)
    tilt_rad = np.arccos(cos_tilt)
    feats['tilt_mean_deg'] = float(np.degrees(np.mean(tilt_rad)))
    feats['tilt_max_deg']  = float(np.degrees(np.max(tilt_rad)))
    feats['tilt_std_deg']  = float(np.degrees(np.std(tilt_rad)))

    # ---- Cross-axis correlations -------------------------------------------
    if np.std(ax) > 1e-8 and np.std(ay) > 1e-8:
        feats['corr_ax_ay'] = float(np.corrcoef(ax, ay)[0, 1])
    else:
        feats['corr_ax_ay'] = 0.0
    if np.std(ax) > 1e-8 and np.std(az) > 1e-8:
        feats['corr_ax_az'] = float(np.corrcoef(ax, az)[0, 1])
    else:
        feats['corr_ax_az'] = 0.0

    # ---- High-g peak ratio (ADXL377 max vs MPU6050 max) -------------------
    # This ratio spikes sharply in crashes (ADXL377 captures what MPU6050 saturates)
    accel_max = float(np.max(accel_mag))
    hg_max    = float(np.max(hg_mag))
    feats['hg_peak_ratio'] = hg_max / max(accel_max, 1e-6)

    return feats


# -------------------------------------------------
#  SLIDING WINDOW OVER FULL DATASET
# -------------------------------------------------

def apply_sliding_window(df: pd.DataFrame,
                         window_size: int = WINDOW_SIZE,
                         stride: int      = STRIDE,
                         verbose: bool    = True) -> pd.DataFrame:
    """
    Apply sliding window (200 samples = 200 ms @ 1000 Hz) over each session.

    Returns a flat DataFrame — one row per window — ready for ML training.
    Label = majority vote across the window.
    """
    all_features = []
    sessions = df['session_id'].unique()

    for sess_id in sessions:
        sess_df = df[df['session_id'] == sess_id].reset_index(drop=True)
        n = len(sess_df)

        for start in range(0, n - window_size + 1, stride):
            end    = start + window_size
            window = sess_df.iloc[start:end]
            feats  = extract_features_window(window)

            majority_label = int(window['label'].value_counts().idxmax())
            feats['label']      = majority_label
            feats['session_id'] = int(sess_id)
            all_features.append(feats)

    feature_df = pd.DataFrame(all_features)

    if verbose:
        print(f"Windows extracted   : {len(feature_df)}")
        print(f"Features per window : {feature_df.shape[1] - 2}")
        print(f"Window duration     : {WINDOW_MS} ms  ({window_size} samples @ {SAMPLE_RATE_HZ} Hz)")
        print(f"\nWindow label distribution:")
        counts = feature_df['label'].value_counts().sort_index()
        for lbl, cnt in counts.items():
            print(f"  {LABEL_MAP.get(lbl, str(lbl)):<15} {cnt:>6}")

    return feature_df


# -------------------------------------------------
#  TRAIN / TEST SPLIT (session-aware)
# -------------------------------------------------

def split_dataset(feature_df: pd.DataFrame,
                  test_size: float = 0.2,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Session-aware train/test split to prevent data leakage.
    Splits by session_id so consecutive windows from the same session
    don't bleed between train and test sets.
    """
    from sklearn.model_selection import GroupShuffleSplit

    feature_cols = [c for c in feature_df.columns if c not in ('label', 'session_id')]
    X      = feature_df[feature_cols]
    y      = feature_df['label']
    groups = feature_df['session_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


# -------------------------------------------------
#  CLI entry point
# -------------------------------------------------
if __name__ == "__main__":
    import os
    raw_path  = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "helmet_imu_raw.csv")
    feat_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")
    os.makedirs(os.path.dirname(os.path.abspath(feat_path)), exist_ok=True)
    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    print("Applying sliding window...")
    feature_df = apply_sliding_window(df, window_size=WINDOW_SIZE, stride=STRIDE, verbose=True)
    feature_df.to_csv(feat_path, index=False)
    print(f"\nSaved features to: {feat_path}")
