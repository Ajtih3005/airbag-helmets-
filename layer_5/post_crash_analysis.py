"""
layer_5/post_crash_analysis.py
--------------------------------
Post-Crash Black Box Analysis & Crash Reconstruction Report — Layer 5.
Runs entirely on Raspberry Pi after airbag deployment.

Outputs:
    1. logs/crash_blackbox_<timestamp>.csv     — Full 60s raw telemetry dump
    2. logs/crash_report_<timestamp>.txt       — Human-readable crash report
    3. AIRBAG_DEPLOYED.lock                    — One-shot safety lock file
"""

import os
import sys
import time
import math
import logging
import json
from datetime import datetime

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR  = os.path.join(ROOT_DIR, "logs")
LOCK_FILE = os.path.join(ROOT_DIR, "AIRBAG_DEPLOYED.lock")

log = logging.getLogger("layer5")

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "CRASH"}
GRAVITY   = 9.81  # m/s²


# ─── Crash Metrics ─────────────────────────────────────────────────────────────

def compute_crash_metrics(blackbox_entries):
    """
    Analyse the Black Box RAM buffer to extract:
      - Peak g-force (MPU6050 and ADXL377 channels)
      - Peak gyro rotation rate
      - Max jerk
      - Estimated tumble duration
      - Crash severity classification
    """
    if not blackbox_entries:
        return {}

    peak_accel_g   = 0.0
    peak_hg_g      = 0.0
    peak_gyro      = 0.0
    prev_ax = prev_ay = prev_az = None
    max_jerk       = 0.0
    crash_samples  = 0
    total_samples  = len(blackbox_entries)

    for entry in blackbox_entries:
        ax = float(entry.get("ax", 0.0))
        ay = float(entry.get("ay", 0.0))
        az = float(entry.get("az", 0.0))
        gx = float(entry.get("gx", 0.0))
        gy = float(entry.get("gy", 0.0))
        gz = float(entry.get("gz", 0.0))
        hg_ax = float(entry.get("hg_ax", ax))
        hg_ay = float(entry.get("hg_ay", ay))
        hg_az = float(entry.get("hg_az", az))

        accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
        accel_g   = accel_mag / GRAVITY
        hg_mag    = math.sqrt(hg_ax**2 + hg_ay**2 + hg_az**2) / GRAVITY
        gyro_mag  = math.sqrt(gx**2 + gy**2 + gz**2)

        if accel_g > peak_accel_g: peak_accel_g = accel_g
        if hg_mag  > peak_hg_g:   peak_hg_g    = hg_mag
        if gyro_mag > peak_gyro:  peak_gyro    = gyro_mag

        if prev_ax is not None:
            jerk = math.sqrt((ax - prev_ax)**2 + (ay - prev_ay)**2 + (az - prev_az)**2)
            if jerk > max_jerk: max_jerk = jerk

        prev_ax, prev_ay, prev_az = ax, ay, az

        if entry.get("p1_label") == 2 or entry.get("p2_label") == 2:
            crash_samples += 1

    tumble_duration_ms = crash_samples  # 1 sample = 1 ms at 1000 Hz

    # Severity classification
    if peak_hg_g >= 100 or peak_accel_g >= 50:
        severity = "SEVERE (Life-threatening impact)"
    elif peak_hg_g >= 50 or peak_accel_g >= 25:
        severity = "HIGH (Major crash)"
    elif peak_hg_g >= 20 or peak_accel_g >= 10:
        severity = "MODERATE (Significant impact)"
    else:
        severity = "LOW (Minor incident)"

    return {
        "total_samples_in_blackbox": total_samples,
        "total_duration_s": round(total_samples / 1000.0, 2),
        "peak_accel_mpu_g": round(peak_accel_g, 2),
        "peak_accel_hg_g": round(peak_hg_g, 2),
        "peak_gyro_deg_per_s": round(peak_gyro, 1),
        "max_jerk_ms2": round(max_jerk * 1000, 2),
        "crash_duration_ms": tumble_duration_ms,
        "severity": severity,
    }


# ─── CSV Dump ──────────────────────────────────────────────────────────────────

def dump_blackbox_csv(blackbox_entries, timestamp_str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, f"crash_blackbox_{timestamp_str}.csv")

    try:
        import pandas as pd
        df = pd.DataFrame(blackbox_entries)
        df.to_csv(csv_path, index=False)
        size_kb = os.path.getsize(csv_path) / 1024
        print(f"[LAYER 5] Black Box CSV saved -> {csv_path} ({size_kb:.1f} KB)")
    except ImportError:
        # Fallback: write raw CSV without pandas
        if blackbox_entries:
            headers = list(blackbox_entries[0].keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in blackbox_entries:
                    f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
            print(f"[LAYER 5] Black Box CSV saved (no pandas) -> {csv_path}")

    return csv_path


# ─── Text Crash Report ─────────────────────────────────────────────────────────

def generate_crash_report(metrics, deploy_ts, crash_onset_ts, latency_ms, reason, timestamp_str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    report_path = os.path.join(LOGS_DIR, f"crash_report_{timestamp_str}.txt")

    dt = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
================================================================================
          SMART AIRBAG HELMET — POST-CRASH ANALYSIS REPORT
================================================================================
  Generated       : {dt}
  Session ID      : {timestamp_str}
================================================================================

  DEPLOYMENT SUMMARY
  ------------------
  Deployment Status    : AIRBAG DEPLOYED
  Reason               : {reason}
  Crash Onset          : {crash_onset_ts} ms
  Airbag Deployed At   : {deploy_ts} ms
  Detection Latency    : {latency_ms} ms
  {'WITHIN SPEC (<20ms)' if latency_ms <= 20 else 'EXCEEDS SPEC (>20ms)'}

================================================================================
  CRASH METRICS (from 60s Black Box RAM Buffer)
================================================================================
  Duration in Buffer   : {metrics.get('total_duration_s', 0):.2f} s  ({metrics.get('total_samples_in_blackbox', 0):,} samples)
  Peak Accel (MPU6050) : {metrics.get('peak_accel_mpu_g', 0):.2f} g
  Peak Accel (ADXL377) : {metrics.get('peak_accel_hg_g', 0):.2f} g  (High-Impact Sensor)
  Peak Gyro Rate       : {metrics.get('peak_gyro_deg_per_s', 0):.1f} °/s
  Max Jerk             : {metrics.get('max_jerk_ms2', 0):.1f} m/s³
  Crash Duration       : {metrics.get('crash_duration_ms', 0)} ms

================================================================================
  SEVERITY CLASSIFICATION
================================================================================
  >> {metrics.get('severity', 'UNKNOWN')} <<

================================================================================
  Seek immediate medical attention and do not remove the helmet until
  medical personnel have assessed potential neck/head injuries.
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[LAYER 5] Crash report saved -> {report_path}")
    print(report)
    return report_path


# ─── Safety Lock File ──────────────────────────────────────────────────────────

def write_safety_lock(timestamp_str, reason, metrics):
    """
    Writes a persistent AIRBAG_DEPLOYED.lock file.
    The Pi will check this file on startup and refuse to arm until
    the lock is manually cleared by a technician.
    """
    lock_data = {
        "deployed": True,
        "timestamp": timestamp_str,
        "reason": reason,
        "severity": metrics.get("severity", "UNKNOWN"),
        "peak_g": metrics.get("peak_accel_hg_g", 0),
    }
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    print(f"[LAYER 5] Safety lock written -> {LOCK_FILE}")
    print("[LAYER 5] *** Helmet LOCKED — Manual reset required by technician. ***")


# ─── Public Entry Point ────────────────────────────────────────────────────────

def run_post_crash_analysis(blackbox_entries, deploy_ts_ms, crash_onset_ms, latency_ms, reason):
    """
    Main entry point called by layer_4/pi_main.py after airbag deployment.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("  LAYER 5 — POST-CRASH BLACK BOX ANALYSIS STARTING...")
    print("=" * 70)

    # 1. Compute crash metrics from Black Box RAM
    metrics = compute_crash_metrics(blackbox_entries)

    # 2. Dump full Black Box telemetry to CSV
    csv_path = dump_blackbox_csv(blackbox_entries, timestamp_str)

    # 3. Generate human-readable crash report
    report_path = generate_crash_report(
        metrics=metrics,
        deploy_ts=deploy_ts_ms,
        crash_onset_ts=crash_onset_ms,
        latency_ms=latency_ms,
        reason=reason,
        timestamp_str=timestamp_str,
    )

    # 4. Write persistent safety lock file
    write_safety_lock(timestamp_str, reason, metrics)

    print(f"\n[LAYER 5] Done.")
    print(f"  Black Box CSV : {csv_path}")
    print(f"  Crash Report  : {report_path}")
    print(f"  Safety Lock   : {LOCK_FILE}")
    print("=" * 70)

    return metrics
