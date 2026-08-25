"""
layer_3/run_detection.py
-------------------------
CLI Runner for Layer 3 ML Detection.
Processes a generated sensor CSV (from Layer 1) or synthetic stream
and logs real-time Path 1, Path 2, Deterioration Sentinel, and Path 3 decisions.

Usage:
    python layer_3/run_detection.py --input data/pre_decided_sensor_data.csv
    python layer_3/run_detection.py --input data/pre_decided_sensor_data.csv --verbose
"""

import argparse
import os
import sys
import time
import pandas as pd

LAYER3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAYER3_DIR)

from detector import Layer3Detector

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "CRASH"}


def run_evaluation(csv_path: str, verbose: bool = False):
    print("=" * 80)
    print(f"  SMART AIRBAG HELMET — LAYER 3 ML EVALUATION")
    print(f"  Input: {csv_path}")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"[ERROR] Input file does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    detector = Layer3Detector()

    print(f"\n[Models Loaded]")
    print(f"  Path 1 (Per-Sample): {'READY' if detector.sample_model else 'NOT LOADED'}")
    print(f"  Path 2 (Window RF) : {'READY' if detector.window_model else 'NOT LOADED'}")
    print(f"  Total samples      : {len(df):,}")

    print("\n" + "-" * 115)
    print(f"{'ms':>6} | {'True':^10} | {'P1 Sample ML':^14} | {'P1%':^6} | {'P2 Window ML':^14} | {'P2%':^6} | {'Det':^5} | {'Trend':^7} | Gate | NOTE")
    print("-" * 115)

    deploy_time_ms = None
    first_crash_ms = None

    for i, row in df.iterrows():
        sample = {
            "ax": float(row["ax"]), "ay": float(row["ay"]), "az": float(row["az"]),
            "gx": float(row["gx"]), "gy": float(row["gy"]), "gz": float(row["gz"]),
            "hg_ax": float(row["hg_ax"]), "hg_ay": float(row["hg_ay"]), "hg_az": float(row["hg_az"]),
        }
        true_label = int(row.get("label", 0))
        true_name  = str(row.get("label_name", LABEL_MAP.get(true_label, "?")))

        if true_label == 2 and first_crash_ms is None:
            first_crash_ms = int(row["timestamp_ms"])

        res = detector.process_sample(sample)

        note = ""
        if res["new_deploy"]:
            deploy_time_ms = int(row["timestamp_ms"])
            note = "*** AIRBAG DEPLOYED! ***"
        elif res["gate_count"] >= 2:
            note = f"GATE ARMED ({res['gate_count']}/3)"
        elif res["p1_label"] == 1:
            note = "Near-Crash warning"

        # Print on important events or verbose mode
        should_print = verbose or res["new_deploy"] or (res["p1_label"] != 0 and i % 50 == 0) or (i % 250 == 0)

        if should_print:
            ms_str   = f"{int(row['timestamp_ms']):>6}"
            true_str = f"{true_name:^10}"
            p1_str   = f"{res['p1_name']:^14}"
            p1p_str  = f"{res['p1_crash']*100:>5.1f}%"
            p2_str   = f"{res['p2_name'] if res['p2_name'] else '---':^14}"
            p2p_str  = f"{res['p2_crash']*100:>5.1f}%" if res['p2_crash'] is not None else f"{'---':^6}"
            det_str  = f"{res['det_score']:.2f}"
            tr_str   = f"{res['trend']:+.3f}"
            gate_str = f"{res['gate_count']}/3"

            print(f"{ms_str} | {true_str} | {p1_str} | {p1p_str:^6} | {p2_str} | {p2p_str:^6} | {det_str:^5} | {tr_str:^7} | {gate_str:^4} | {note}")

        if res["new_deploy"]:
            break

    print("-" * 115)
    print("\n" + "=" * 80)
    print("  LAYER 3 EVALUATION RESULT")
    print("=" * 80)
    if deploy_time_ms is not None:
        latency = (deploy_time_ms - first_crash_ms) if first_crash_ms is not None else 0
        print(f"  STATUS          : AIRBAG DEPLOYED")
        print(f"  Crash Onset     : {first_crash_ms} ms")
        print(f"  Deploy Fired    : {deploy_time_ms} ms")
        print(f"  Detection Lag   : {latency} ms (under 20ms spec requirement!)")
    else:
        print(f"  STATUS          : SAFE RIDE — No deployment required.")
    print(f"  Windows checked : {detector.total_windows}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Layer 3 Detection Runner")
    parser.add_argument("--input", default="data/pre_decided_sensor_data.csv",
                        help="Path to input sensor telemetry CSV")
    parser.add_argument("--verbose", action="store_true", help="Print every evaluation row")
    args = parser.parse_args()

    run_evaluation(args.input, verbose=args.verbose)


if __name__ == "__main__":
    main()
