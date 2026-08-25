"""
layer_4/pi_main.py
--------------------
Raspberry Pi Main Loop — Layer 4.

Ties Layer 1 → Layer 3 → Layer 4 together in one Raspberry Pi runtime:
  - Reads telemetry from the pre_decided_sensor_data.csv (simulation)
    OR directly from real MPU6050 / ADXL377 hardware sensors
  - Runs Layer 3 ML Detection in real time at 1000 Hz
  - Triggers Layer 4 Actuation on crash verdict:
      • GPIO Pin 17 → CO2 Solenoid Valve
      • GPIO Pin 27 → Warning LED / Buzzer
      • SIM800L → Emergency SMS
  - Hands off to Layer 5 for post-crash Black Box dump

Usage (on Raspberry Pi):
    python layer_4/pi_main.py --input data/pre_decided_sensor_data.csv
    python layer_4/pi_main.py --input data/pre_decided_sensor_data.csv --hardware
    python layer_4/pi_main.py --input data/pre_decided_sensor_data.csv --phone +91XXXXXXXXXX
"""

import argparse
import os
import sys
import time
import logging
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "layer_3"))
sys.path.insert(0, os.path.join(ROOT_DIR, "layer_4"))
sys.path.insert(0, os.path.join(ROOT_DIR, "layer_5"))

from detector   import Layer3Detector
from actuator   import ActuatorEngine

log = logging.getLogger("layer4.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "CRASH"}


def run_pi_loop(csv_path, hardware=False, phone=None, realtime=False):
    print("=" * 70)
    print("  SMART AIRBAG HELMET — RASPBERRY PI MAIN LOOP")
    print(f"  Mode     : {'HARDWARE GPIO' if hardware else 'SIMULATION'}")
    print(f"  Input    : {csv_path}")
    print(f"  Phone    : {phone or 'Not configured'}")
    print("=" * 70)

    detector = Layer3Detector()
    actuator = ActuatorEngine(hardware=hardware, emergency_number=phone or "+91XXXXXXXXXX")

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} samples from CSV.")

    near_crash_warned    = False
    near_crash_cooldown  = 0      # samples since last warning pulse
    WARN_COOLDOWN_MS     = 500   # minimum gap between warning pulses
    first_crash_ts       = None
    deploy_result        = None

    for _, row in df.iterrows():
        sample = {
            "ax": float(row["ax"]), "ay": float(row["ay"]), "az": float(row["az"]),
            "gx": float(row["gx"]), "gy": float(row["gy"]), "gz": float(row["gz"]),
            "hg_ax": float(row["hg_ax"]), "hg_ay": float(row["hg_ay"]), "hg_az": float(row["hg_az"]),
        }
        true_label = int(row.get("label", 0))
        t_ms       = float(row.get("timestamp_ms", 0))

        # Optional: Real-time 1000Hz pacing
        if realtime:
            time.sleep(0.001)

        result = detector.process_sample(sample, current_time_s=t_ms / 1000.0)

        # Near-Crash Warning (once per event, with 500ms cooldown)
        near_crash_cooldown = max(0, near_crash_cooldown - 1)
        if result["p1_label"] == 1 and not actuator.deployed:
            if not near_crash_warned and near_crash_cooldown == 0:
                near_crash_warned = True
                near_crash_cooldown = WARN_COOLDOWN_MS
                actuator.pulse_warning(n_pulses=2, pulse_ms=80)
        elif result["p1_label"] == 0:
            near_crash_warned = False

        # Track first crash onset
        if true_label == 2 and first_crash_ts is None:
            first_crash_ts = t_ms

        # DEPLOY on Layer 3 verdict
        if result["new_deploy"] and not actuator.deployed:
            fired = actuator.deploy_airbag(reason=result.get("decision_reason", "ML_CRASH"))
            if fired:
                deploy_result = {
                    "deploy_ts_ms": t_ms,
                    "crash_onset_ms": first_crash_ts,
                    "latency_ms": (t_ms - first_crash_ts) if first_crash_ts else 0,
                    "reason": result.get("decision_reason", ""),
                    "blackbox": list(detector.blackbox),
                }
                actuator.send_emergency_sms()
                break

    # ─── Post-Crash: Hand off to Layer 5 ─────────────────────────────────
    if deploy_result:
        try:
            from post_crash_analysis import run_post_crash_analysis
            run_post_crash_analysis(
                deploy_result["blackbox"],
                deploy_result["deploy_ts_ms"],
                deploy_result["crash_onset_ms"],
                deploy_result["latency_ms"],
                deploy_result["reason"],
            )
        except Exception as e:
            log.error(f"[LAYER 5] Post-crash analysis failed: {e}")
    else:
        log.info("[RESULT] No deployment required. Safe ride completed.")

    actuator.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Main Loop — Layer 4")
    parser.add_argument("--input",    default="data/pre_decided_sensor_data.csv")
    parser.add_argument("--hardware", action="store_true", help="Enable real GPIO on Raspberry Pi")
    parser.add_argument("--phone",    type=str, default=None, help="Emergency phone number")
    parser.add_argument("--realtime", action="store_true", help="Pace at real 1000Hz rate")
    args = parser.parse_args()

    run_pi_loop(
        csv_path=args.input,
        hardware=args.hardware,
        phone=args.phone,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
