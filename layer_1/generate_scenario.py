"""
layer_1/generate_scenario.py
------------------------------
CLI Entry Point — Layer 1: Scenario Generation & Sensor Telemetry Output.

Usage:
    # Option 1 — LLM prompt (uses Gemini API or keyword fallback):
    python layer_1/generate_scenario.py --prompt "highway cruise then hard braking and a lowside crash"

    # Option 2 — Procedural system generation:
    python layer_1/generate_scenario.py --procedural

    # Optional flags:
    --output  path/to/output.csv      (default: data/pre_decided_sensor_data.csv)
    --duration 8000                   (total ms for procedural mode, default: 5000)
    --seed 42                         (random seed for reproducibility)
    --preview                         (print first 10 rows of the sensor data)
"""

import argparse
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── resolve layer_1 imports ───────────────────────────────────────────────────
LAYER1_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(LAYER1_DIR)
sys.path.insert(0, LAYER1_DIR)

from data_generator  import (generate_from_segments, generate_procedural,
                              SAMPLE_RATE_HZ, LABEL_MAP, PHASE_LABELS)
from llm_generator   import generate_segments_from_prompt

# Target: 3 minutes of simulation
# Formula: 3min * 60s * PLAYBACK_FPS(20) * FRAME_STEP(10) = 36,000 rows
TARGET_DURATION_MS = 36_000   # 36 seconds of 1kHz data = 3 min visual at 20fps/step10


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, segments: list[dict]) -> None:
    """Print a human-readable summary of the generated scenario."""
    print("\n" + "=" * 65)
    print("  LAYER 1 — SCENARIO & SENSOR TELEMETRY SUMMARY")
    print("=" * 65)

    print("\n  Scenario Segments:")
    print(f"  {'#':<4} {'Event':<20} {'Duration':>12}   {'Label':<12}")
    print(f"  {'-'*4} {'-'*20} {'-'*12}   {'-'*12}")
    for i, seg in enumerate(segments):
        label_id = df.loc[
            df["timestamp_ms"] >= (sum(s["duration_ms"] for s in segments[:i])),
            "label"
        ].iloc[0] if len(df) > 0 else 0
        label_name = LABEL_MAP.get(label_id, "?")
        print(f"  {i+1:<4} {seg['event']:<20} {seg['duration_ms']:>9} ms   {label_name:<12}")

    print(f"\n  Total samples   : {len(df):,}")
    print(f"  Total duration  : {len(df)} ms  ({len(df)/SAMPLE_RATE_HZ:.2f}s)")
    print(f"  Sample rate     : {SAMPLE_RATE_HZ} Hz")
    print(f"  Sensor channels : ax, ay, az, gx, gy, gz, hg_ax, hg_ay, hg_az")

    counts = df["label_name"].value_counts()
    total  = len(df)
    print(f"\n  Label distribution:")
    for name, cnt in counts.items():
        bar = "|" * int(cnt / total * 30)
        print(f"    {name:<12} {cnt:>6} samples  ({cnt/total*100:5.1f}%)  {bar}")

    print("=" * 65)


def save_output(df: pd.DataFrame, output_path: str) -> None:
    """Save the sensor DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Sensor data saved → {output_path}")
    print(f"  File size         : {os.path.getsize(output_path) / 1024:.1f} KB")


def save_manifest(segments: list, df: pd.DataFrame, output_path: str) -> None:
    """
    Save a scenario_manifest.json alongside the CSV.
    Records exact ms timestamps of each phase transition so the UI can
    draw the timeline and score ML detection latency against ground truth.
    The ML never sees this during runtime — it must detect phases from sensor data alone.
    """
    crash_events   = {"crash", "highside", "lowside", "front_collision", "rear_collision"}
    nc_events      = {"pothole", "swerve", "brake", "gravel", "near_crash"}

    timeline = {}
    t = 0
    for seg in segments:
        ev  = seg["event"]
        dur = seg["duration_ms"]
        ph  = PHASE_LABELS.get(ev, "Normal")
        if ph == "Normal" and "normal_start_ms" not in timeline:
            timeline["normal_start_ms"] = t
        elif ph == "Near-Crash" and "near_crash_ms" not in timeline:
            timeline["near_crash_ms"] = t
        elif ph == "Pre-Crash" and "pre_crash_ms" not in timeline:
            timeline["pre_crash_ms"] = t
        elif ph == "Crash" and "crash_ms" not in timeline:
            timeline["crash_ms"] = t
        t += dur

    timeline["end_ms"] = t

    manifest = {
        "segments": segments,
        "timeline": timeline,
        "crash_known_at_ms": timeline.get("crash_ms"),
        "pre_crash_known_at_ms": timeline.get("pre_crash_ms"),
        "total_samples": len(df),
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }

    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), "scenario_manifest.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved    → {manifest_path}")
    return manifest_path


# ─────────────────────────────────────────────────────────────────────────────
def pad_segments_to_duration(segments: list, target_ms: int = TARGET_DURATION_MS) -> list:
    """
    Adjust segments so they total exactly target_ms milliseconds.
    - If total < target: extend the first normal/highway segment (or prepend one)
    - If total > target: trim the last segment down to fit
    Always keeps the crash at the END so the full story is told.
    """
    # Separate crash/near-crash from normal ones
    crash_events   = {"crash", "highside", "lowside", "front_collision", "rear_collision"}
    nc_events      = {"pothole", "swerve", "brake", "gravel"}
    normal_events  = {"normal", "highway", "city", "accel", "rain"}

    total = sum(s["duration_ms"] for s in segments)

    if total < target_ms:
        gap = target_ms - total
        # Find the first normal segment to extend
        for seg in segments:
            if seg["event"] in normal_events:
                seg["duration_ms"] += gap
                total = target_ms
                break
        else:
            # No normal segment found: prepend one
            segments.insert(0, {"event": "highway", "duration_ms": gap})
            total = target_ms

    elif total > target_ms:
        # Trim from the end to not cut the crash itself
        excess = total - target_ms
        for seg in reversed(segments):
            if seg["event"] in normal_events:
                trim = min(excess, seg["duration_ms"] - 500)  # keep at least 500ms
                seg["duration_ms"] -= trim
                excess -= trim
                if excess <= 0:
                    break

    print(f"[Layer 1] Scenario padded to {sum(s['duration_ms'] for s in segments):,}ms "
          f"= {sum(s['duration_ms'] for s in segments)/1000:.1f}s of sensor data "
          f"(~{sum(s['duration_ms'] for s in segments)/200/60:.1f} min visual)")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 — Scenario Generator & Sensor Telemetry Builder"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prompt", type=str, metavar="TEXT",
        help="Natural language description of the ride scenario (uses Gemini LLM or keyword fallback)."
    )
    mode.add_argument(
        "--procedural", action="store_true",
        help="Automatically generate a random riding scenario using the procedural engine."
    )

    parser.add_argument(
        "--output", type=str,
        default=os.path.join(ROOT_DIR, "data", "pre_decided_sensor_data.csv"),
        help="Output CSV file path (default: data/pre_decided_sensor_data.csv)."
    )
    parser.add_argument(
        "--duration", type=int, default=TARGET_DURATION_MS,
        help=f"Total duration in ms for procedural mode (default: {TARGET_DURATION_MS}ms = 3 min visual)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random each run)."
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print the first 10 rows of the generated sensor data."
    )
    parser.add_argument(
        "--save-segments", type=str, default=None, metavar="PATH",
        help="Optionally save the scenario segment plan as a JSON file."
    )

    args = parser.parse_args()

    # ── Set up RNG ────────────────────────────────────────────────────────────
    import time
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
    rng  = np.random.default_rng(seed)
    print(f"[Layer 1] Random seed: {seed}")

    # ── Generate segments ─────────────────────────────────────────────────────
    if args.prompt:
        print(f"\n[Layer 1] Mode: LLM Prompt")
        print(f"[Layer 1] Prompt: \"{args.prompt}\"")
        segments = generate_segments_from_prompt(args.prompt)
        segments = pad_segments_to_duration(segments, TARGET_DURATION_MS)
        df       = generate_from_segments(segments, rng)
    else:
        print(f"\n[Layer 1] Mode: Procedural System Generation")
        print(f"[Layer 1] Target duration: {args.duration} ms")
        segments, df = generate_procedural(total_duration_ms=args.duration, rng=rng)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(df, segments)

    # ── Preview ───────────────────────────────────────────────────────────────
    if args.preview:
        print("\n  First 10 samples:")
        print(df.head(10).to_string(index=False))
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_output(df, args.output)

    # ── Save Manifest ─────────────────────────────────────────────────────────
    save_manifest(segments, df, args.output)

    # ── Save segments JSON (optional) ─────────────────────────────────────────
    if args.save_segments:
        seg_path = args.save_segments
        os.makedirs(os.path.dirname(os.path.abspath(seg_path)), exist_ok=True)
        with open(seg_path, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"  Segments plan saved → {seg_path}")

    print("\n[Layer 1] Done.\n")


if __name__ == "__main__":
    main()
