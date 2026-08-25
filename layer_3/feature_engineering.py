"""
layer_3/feature_engineering.py
--------------------------------
Sliding window + 81-feature extraction for Layer 3 ML classification.
Sourced from plan_a/src/feature_engineering.py.
"""

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 1000
WINDOW_MS      = 50
WINDOW_SIZE    = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # 50 samples
STRIDE_MS      = 25                                       # 25ms stride = 40Hz window evaluations
STRIDE         = int(SAMPLE_RATE_HZ * STRIDE_MS / 1000)   # 25 samples

SENSOR_COLS     = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
HG_SENSOR_COLS  = ['hg_ax', 'hg_ay', 'hg_az']
ALL_SENSOR_COLS = SENSOR_COLS + HG_SENSOR_COLS

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "Crash"}


def extract_features_window(window: pd.DataFrame) -> dict:
    """
    Extract 81 features from a single window.
    """
    feats = {}

    ax = window['ax'].values
    ay = window['ay'].values
    az = window['az'].values
    gx = window['gx'].values
    gy = window['gy'].values
    gz = window['gz'].values

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
        feats[f'{name}_mean']  = float(np.mean(col))
        feats[f'{name}_std']   = float(np.std(col))
        feats[f'{name}_max']   = float(np.max(col))
        feats[f'{name}_min']   = float(np.min(col))
        feats[f'{name}_range'] = float(np.max(col) - np.min(col))
        feats[f'{name}_rms']   = float(np.sqrt(np.mean(col ** 2)))
        q75, q25 = np.percentile(col, [75, 25])
        feats[f'{name}_iqr']   = float(q75 - q25)
        if np.std(col) > 1e-8:
            feats[f'{name}_skew'] = float(np.mean(((col - np.mean(col)) / np.std(col)) ** 3))
        else:
            feats[f'{name}_skew'] = 0.0

    # ---- Per-axis stats: ADXL377 (high-g) ---------------------------------
    for name, col in zip(['hg_ax','hg_ay','hg_az'], [hg_ax, hg_ay, hg_az]):
        feats[f'{name}_mean'] = float(np.mean(col))
        feats[f'{name}_std']  = float(np.std(col))
        feats[f'{name}_max']  = float(np.max(np.abs(col)))
        feats[f'{name}_rms']  = float(np.sqrt(np.mean(col ** 2)))

    # ---- Derived: Magnitudes -----------------------------------------------
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
    hg_mag    = np.sqrt(hg_ax**2 + hg_ay**2 + hg_az**2)

    feats['accel_mag_mean'] = float(np.mean(accel_mag))
    feats['accel_mag_std']  = float(np.std(accel_mag))
    feats['accel_mag_max']  = float(np.max(accel_mag))

    feats['gyro_mag_mean']  = float(np.mean(gyro_mag))
    feats['gyro_mag_std']   = float(np.std(gyro_mag))
    feats['gyro_mag_max']   = float(np.max(gyro_mag))

    feats['hg_mag_mean']    = float(np.mean(hg_mag))
    feats['hg_mag_std']     = float(np.std(hg_mag))
    feats['hg_mag_max']     = float(np.max(hg_mag))

    # ---- Signal Magnitude Area ---------------------------------------------
    feats['sma_accel'] = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    feats['sma_gyro']  = float(np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz)))
    feats['sma_hg']    = float(np.mean(np.abs(hg_ax) + np.abs(hg_ay) + np.abs(hg_az)))

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

    # ---- Cross-Axis Correlations -------------------------------------------
    def _safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    feats['corr_ax_ay'] = _safe_corr(ax, ay)
    feats['corr_ax_az'] = _safe_corr(ax, az)
    feats['corr_ay_az'] = _safe_corr(ay, az)
    feats['corr_gx_gy'] = _safe_corr(gx, gy)
    feats['corr_gx_gz'] = _safe_corr(gx, gz)
    feats['corr_gy_gz'] = _safe_corr(gy, gz)

    # ---- High-G Specific Ratios --------------------------------------------
    mpu_peak = max(float(np.max(np.abs(ax))), float(np.max(np.abs(ay))), 1e-6)
    hg_peak  = float(np.max(np.abs(hg_ax)))
    feats['hg_peak_ratio'] = hg_peak / mpu_peak

    return feats
