"""
send_to_pi.py
-------------
Direct headless trigger: Tells the Layer 2 server to stream sensor telemetry
directly to the Raspberry Pi over WebSocket without needing any browser open!

Usage:
    python send_to_pi.py
    python send_to_pi.py --csv data/pre_decided_sensor_data.csv
    python send_to_pi.py --server http://localhost:5500
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

def trigger_stream(server_url="http://localhost:5500", csv_path=None, scenario="front_collision_demo_01", biome="hill_station"):
    endpoint = f"{server_url.rstrip('/')}/api/start"
    payload = {
        "scenario": scenario,
        "biome": biome
    }
    if csv_path:
        payload["csv_path"] = csv_path

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("=" * 60)
    print("  SMART HELMET AI — DIRECT STREAM TO PI (HEADLESS)")
    print(f"  Target Server : {endpoint}")
    print(f"  Scenario      : {scenario}")
    print("=" * 60)

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_body = response.read().decode("utf-8")
            result = json.loads(res_body)

            if result.get("success"):
                print("\n[SUCCESS] Stream started successfully!")
                print(f"  Message      : {result.get('message')}")
                print(f"  Pi Connected : {'YES (' + str(result.get('pi_type')) + ')' if result.get('pi_connected') else 'NO (Start pi_ws_client.py first)'}")
                print(f"  Browsers     : {result.get('browser_clients_count', 0)} connected (visual simulation bypassed)")
                print("\nTelemetry is now streaming directly to the Raspberry Pi.")
                print("Watch your Pi terminal for live ML detections and Relay actuation.")
            else:
                print(f"\n[ERROR] Server returned failure: {result.get('message')}")

    except urllib.error.URLError as e:
        print(f"\n[CONNECTION ERROR] Could not reach Layer 2 server at {server_url}: {e}")
        print("Make sure layer_2/server.py is running (e.g. python layer_2/server.py).")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Directly stream sensor telemetry from L2 to Pi (no browser needed)")
    parser.add_argument("--server", default="http://localhost:5500", help="Layer 2 server HTTP URL (default: http://localhost:5500)")
    parser.add_argument("--csv", default=None, help="Custom CSV telemetry path (default: data/pre_decided_sensor_data.csv)")
    parser.add_argument("--scenario", default="front_collision_demo_01", help="Scenario ID (default: front_collision_demo_01)")
    args = parser.parse_args()

    trigger_stream(server_url=args.server, csv_path=args.csv, scenario=args.scenario)

if __name__ == "__main__":
    main()
