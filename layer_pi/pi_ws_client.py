"""
layer_pi/pi_ws_client.py
--------------------------
Raspberry Pi WebSocket Client — Layer 2A (Pi Side).

Connects to the laptop WebSocket server (layer_2/server.py) over WiFi.
Receives 60fps sensor frames, runs full Layer 3 ML detection pipeline,
sends verdicts back to the browser, and triggers Layer 4 actuation + Layer 5
post-crash analysis on confirmed crash.

Usage on Raspberry Pi:
    python layer_pi/pi_ws_client.py --server ws://192.168.1.XX:8765
    python layer_pi/pi_ws_client.py --server ws://192.168.1.XX:8765 --hardware
    python layer_pi/pi_ws_client.py --standalone   # Use local CSV (no server needed)
"""

import asyncio
import json
import os
import sys
import time
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ─── Imports from layered packages ───────────────────────────────────────────
try:
    from l3.detector          import Layer3Detector
    from l4.actuator          import ActuatorEngine
    from l5.post_crash_analysis import run_post_crash_analysis
except ImportError as e:
    print(f"[PI CLIENT] Missing module: {e}")
    print("[PI CLIENT] Make sure l3/, l4/, l5/ folders exist inside layer_pi/")
    sys.exit(1)

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "CRASH"}


# ─── WebSocket Mode ────────────────────────────────────────────────────────────

async def run_ws_client(server_url: str, hardware: bool, phone: str):
    try:
        import websockets
    except ImportError:
        print("[PI CLIENT] websockets not installed. Run: pip install websockets")
        sys.exit(1)

    detector    = Layer3Detector()
    actuator    = ActuatorEngine(hardware=hardware, emergency_number=phone)
    deploy_data = None

    print("=" * 60)
    print("  SMART AIRBAG HELMET — RASPBERRY PI WebSocket CLIENT")
    print(f"  Connecting to : {server_url}")
    print(f"  Hardware GPIO : {'YES' if hardware else 'SIMULATION'}")
    print("=" * 60)

    async with websockets.connect(server_url) as ws:
        print(f"[PI CLIENT] Connected to server.")

        # Register as Pi
        await ws.send(json.dumps({"type": "pi_connect"}))

        async for raw in ws:
            msg = json.loads(raw)

            # ── New scenario starting ────────────────────────────────────────
            if msg.get("type") == "playback_start":
                print(f"[PI CLIENT] Scenario starting: {msg.get('scenario','?')} | Biome: {msg.get('biome','?')}")
                detector  = Layer3Detector()   # Fresh detector per scenario
                actuator  = ActuatorEngine(hardware=hardware, emergency_number=phone)
                deploy_data = None

            # ── Process sensor frame ─────────────────────────────────────────
            elif msg.get("type") == "sensor_frame" and not actuator.deployed:
                sample = {
                    "ax": msg["ax"], "ay": msg["ay"], "az": msg["az"],
                    "gx": msg["gx"], "gy": msg["gy"], "gz": msg["gz"],
                    "hg_ax": msg["hg_ax"], "hg_ay": msg["hg_ay"], "hg_az": msg["hg_az"],
                }
                t_ms   = msg.get("t_ms", 0)
                t_s    = t_ms / 1000.0

                t0     = time.perf_counter()
                result = detector.process_sample(sample, current_time_s=t_s)
                lat_ms = (time.perf_counter() - t0) * 1000

                # Near-crash warning (buzzer)
                if result["p1_label"] == 1:
                    actuator.pulse_warning(n_pulses=1, pulse_ms=50)

                # Send verdict back to browser
                p1_lbl = result.get("p1_label") or 0
                p2_lbl = result.get("p2_label") if result.get("p2_label") is not None else 0
                p1_cr  = result.get("p1_crash", result.get("p1_crash_prob", 0.0)) or 0.0
                p2_cr  = result.get("p2_crash", result.get("p2_crash_prob", 0.0)) or 0.0

                verdict = {
                    "type"       : "pi_verdict",
                    "t_ms"       : t_ms,
                    "p1_label"   : int(p1_lbl),
                    "p1_crash"   : float(p1_cr),
                    "p2_label"   : int(p2_lbl),
                    "p2_crash"   : float(p2_cr),
                    "det_score"  : float(result.get("det_score") or 0.0),
                    "gate_count" : int(result.get("gate_count") or 0),
                    "latency_ms" : round(lat_ms, 2),
                    "deployed"   : False,
                }
                await ws.send(json.dumps(verdict))

                # DEPLOY on confirmed crash
                if result["new_deploy"]:
                    fired = actuator.deploy_airbag(reason=result.get("decision_reason", "ML_CRASH"))
                    if fired:
                        print(f"\n[PI CLIENT] *** AIRBAG DEPLOYED at t={t_ms}ms ***")
                        actuator.send_emergency_sms()

                        # Calculate peak g from blackbox
                        bb = list(detector.blackbox)
                        peak_g = max(
                            (math.sqrt(e.get("hg_ax",0)**2 + e.get("hg_ay",0)**2 + e.get("hg_az",0)**2)/9.81
                             for e in bb), default=0
                        )
                        peak_gyro = max(
                            (math.sqrt(e.get("gx",0)**2 + e.get("gy",0)**2 + e.get("gz",0)**2)
                             for e in bb), default=0
                        )

                        deploy_data = {
                            "t_ms": t_ms,
                            "reason": result.get("decision_reason","ML_CRASH"),
                            "blackbox": bb,
                            "peak_g": round(peak_g,2),
                            "peak_gyro": round(peak_gyro,1),
                        }

                        # Notify browser of deploy
                        verdict["deployed"]   = True
                        verdict["deploy_ts"]  = t_ms
                        verdict["peak_g"]     = deploy_data["peak_g"]
                        verdict["peak_gyro"]  = deploy_data["peak_gyro"]
                        await ws.send(json.dumps(verdict))

                        # Run Layer 5 post-crash analysis
                        try:
                            metrics = run_post_crash_analysis(
                                blackbox_entries= bb,
                                deploy_ts_ms    = t_ms,
                                crash_onset_ms  = t_ms,
                                latency_ms      = lat_ms,
                                reason          = deploy_data["reason"],
                            )
                            # Send full black box summary to browser
                            await ws.send(json.dumps({
                                "type"    : "pi_verdict",
                                "t_ms"    : t_ms,
                                "deployed": True,
                                "severity": metrics.get("severity","SEVERE"),
                                "peak_g"  : metrics.get("peak_accel_hg_g", deploy_data["peak_g"]),
                                "peak_gyro": metrics.get("peak_gyro_deg_per_s", deploy_data["peak_gyro"]),
                                "latency_ms": lat_ms,
                                "deploy_ts" : t_ms,
                                "gate_count": int(result["gate_count"]),
                                "p1_label"  : int(result["p1_label"]),
                                "p2_label"  : int(result["p2_label"]),
                                "det_score" : float(result["det_score"]),
                            }))
                        except Exception as e:
                            print(f"[PI CLIENT] Layer 5 error: {e}")

            elif msg.get("type") == "playback_done":
                print("[PI CLIENT] Scenario complete.")
                if not actuator.deployed:
                    print("[PI CLIENT] No crash detected — safe ride.")

        actuator.cleanup()


# ─── Standalone Mode (no server) ──────────────────────────────────────────────

def run_standalone(csv_path: str, hardware: bool, phone: str):
    import pandas as pd
    import math
    from detector import Layer3Detector

    print("=" * 65)
    print("  🚀 RASPBERRY PI STANDALONE MODE (Offline ML Execution)")
    print(f"  📁 Sensor CSV : {csv_path}")
    print(f"  ⚡ Hardware   : {'REAL GPIO' if hardware else 'SIMULATION'}")
    print("=" * 65)

    if not os.path.exists(csv_path):
        print(f"\n[ERROR] CSV file not found: {csv_path}")
        print("Please copy the CSV file to the Pi first.")
        return

    # Safety: clear stale AIRBAG_DEPLOYED.lock from previous test run
    lock_file = os.path.join(ROOT, "AIRBAG_DEPLOYED.lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print("  [SAFETY] Cleared stale AIRBAG_DEPLOYED.lock from previous run.")

    detector = Layer3Detector()
    actuator = ActuatorEngine(hardware=hardware, emergency_number=phone)
    df       = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} sensor samples (~{total_rows/1000:.1f}s of data)\n")

    first_crash = None
    last_print = 0
    deployed = False

    for idx, row in df.iterrows():
        sample = {
            "ax": float(row["ax"]), "ay": float(row["ay"]), "az": float(row["az"]),
            "gx": float(row["gx"]), "gy": float(row["gy"]), "gz": float(row["gz"]),
            "hg_ax": float(row["hg_ax"]), "hg_ay": float(row["hg_ay"]), "hg_az": float(row["hg_az"]),
        }
        t_ms  = float(row.get("timestamp_ms", idx))
        label = int(row.get("label", 0))

        if label == 2 and first_crash is None:
            first_crash = t_ms

        t0 = time.perf_counter()
        result = detector.process_sample(sample, current_time_s=t_ms/1000.0)
        lat_ms = (time.perf_counter() - t0) * 1000

        # Calculate current g-force and lean
        g_force = math.sqrt(sample["ax"]**2 + sample["ay"]**2 + sample["az"]**2) / 9.81
        lean_deg = abs(sample["ay"] * 4.5)

        # Print periodic progress every 2,000 samples (~2 sec)
        if idx - last_print >= 2000:
            last_print = idx
            progress_pct = (idx / total_rows) * 100
            p1_name = "CRASH" if result["p1_label"] == 2 else "NEAR" if result["p1_label"] == 1 else "NORMAL"
            print(f"  [{progress_pct:5.1f}%] t={t_ms/1000:6.1f}s | G-Force: {g_force:4.1f}g | Lean: {lean_deg:4.1f}° | State: {p1_name:<6} (ML Latency: {lat_ms:.2f}ms)")

        # Near-crash warning (buzzer pulse) — rate-limited in ActuatorEngine to 1 pulse per 600ms
        if result["p1_label"] in (1, 2) and result.get("det_score", 0) >= 0.15:
            actuator.pulse_warning(n_pulses=1, pulse_ms=50)

        # Crash detected -> Deploy Airbag
        if result["new_deploy"] and not actuator.deployed:
            deployed = True
            fired = actuator.deploy_airbag(reason=result.get("decision_reason", "3-Path ML Confirmed Crash"))
            actuator.send_emergency_sms()
            det_lat_ms = (t_ms - first_crash) if first_crash else 0

            # Peak calculation
            bb = list(detector.blackbox)
            peak_g = max(
                (math.sqrt(e.get("hg_ax",0)**2 + e.get("hg_ay",0)**2 + e.get("hg_az",0)**2)/9.81 for e in bb),
                default=g_force
            )
            peak_gyro = max(
                (math.sqrt(e.get("gx",0)**2 + e.get("gy",0)**2 + e.get("gz",0)**2) for e in bb),
                default=0
            )

            print("\n" + "=" * 65)
            print("  💥💥💥 CRASH DETECTED — AIRBAG DEPLOYED! 💥💥💥")
            print("=" * 65)
            print(f"  ⏱️  Deployment Timestamp : {t_ms:.1f} ms ({t_ms/1000:.2f}s into ride)")
            print(f"  ⚡ Detection Latency    : {det_lat_ms:.1f} ms")
            print(f"  🎯 Decision Gate       : Gate {result.get('gate_count', 3)}/3 (Confirmed)")
            print(f"  📊 Peak Impact G       : {peak_g:.1f} g")
            print(f"  🔄 Peak Gyro Spin      : {peak_gyro:.1f} deg/s")
            print(f"  📜 Decision Reason     : {result.get('decision_reason', 'ML_CRASH')}")
            print("=" * 65 + "\n")

            # Layer 5: Post-Crash Black Box
            run_post_crash_analysis(
                list(detector.blackbox), t_ms, first_crash, det_lat_ms,
                result.get("decision_reason", "3-Path ML Confirmed Crash")
            )
            break

    if not deployed:
        print("\n" + "=" * 65)
        print("  ✅ RIDE COMPLETED SAFELY — NO CRASH DETECTED")
        print(f"  Processed {total_rows:,} samples successfully.")
        print("=" * 65 + "\n")

    actuator.cleanup()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Raspberry Pi Layer Pi Client")
    p.add_argument("--server",     default="ws://localhost:5500/ws", help="WebSocket server URL (laptop IP e.g. ws://192.168.1.50:5500/ws)")
    p.add_argument("--hardware",   action="store_true",           help="Enable real GPIO on Raspberry Pi")
    p.add_argument("--phone",      default="+91XXXXXXXXXX",       help="Emergency phone number")
    p.add_argument("--standalone", action="store_true",           help="Run from local CSV without server")
    p.add_argument("--csv",        default="../data/pre_decided_sensor_data.csv", help="CSV path for standalone mode")
    args = p.parse_args()

    if args.standalone:
        run_standalone(args.csv, args.hardware, args.phone)
    else:
        asyncio.run(run_ws_client(args.server, args.hardware, args.phone))


if __name__ == "__main__":
    main()
