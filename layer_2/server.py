"""
layer_2/server.py  -  Unified HTTP + WebSocket + Layer-1 API server
----------------------------------------------------------------------
One server on port 5500 handles everything:

  GET  /              -> serves cockpit.html  (Unified AI Cockpit)
  POST /api/generate  -> runs Layer 1 Python, returns CSV path + metadata
  WS   /ws            -> real-time sensor frame streaming to browser & Pi

Usage:
    python layer_2/server.py
    Open http://localhost:5500 in your browser
"""

import asyncio
import json
import os
import sys
import subprocess
import time
from pathlib import Path

from aiohttp import web
import aiohttp

ROOT      = Path(__file__).parent.parent
LAYER2    = Path(__file__).parent
CSV_PATH  = ROOT / "data" / "pre_decided_sensor_data.csv"

# Playback at 20fps. FRAME_STEP=10 means each frame covers 10ms of 1kHz data.
# This gives a 5x richer visual — the bike moves smoothly and events are clearly visible.
# Total time = total_rows / 10 / 20 = total_rows / 200 seconds
PLAYBACK_FPS  = 20
FRAME_STEP    = 10    # 10 rows per frame (not max(1, 1000//fps) which gave 50)

# Shared state
browser_clients: set  = set()
browser_ready: bool   = False   # True once browser signals 3D world is built
pi_client             = None
playback_task         = None
current_df            = None
current_meta: dict    = {}


# ---- HTTP: serve static files -----------------------------------------------

async def handle_index(request):
    """Serve the unified AI cockpit."""
    return web.FileResponse(LAYER2 / "cockpit.html")

async def handle_dashboard(request):
    """Legacy redirect -> cockpit."""
    return web.FileResponse(LAYER2 / "cockpit.html")

async def handle_static(request):
    fname = request.match_info["name"]
    path  = LAYER2 / fname
    if path.exists():
        return web.FileResponse(path)
    return web.Response(status=404, text="Not found")


# ─── HTTP: Layer 1 generate API ───────────────────────────────────────────────

async def handle_generate(request):
    """
    POST /api/generate
    Body: { "prompt": "...", "biome": "hill_station", "mode": "llm"|"proc", "events": [...] }
    Returns: { "success": true, "csv_path": "...", "biome": "...", "events": [...], "duration_ms": ... }
    """
    try:
        data   = await request.json()
        prompt = data.get("prompt", "").strip()
        biome  = data.get("biome", "hill_station")
        mode   = data.get("mode", "llm")
        events = data.get("events", [])

        if mode == "llm" and prompt:
            cmd = [sys.executable, str(ROOT / "layer_1" / "generate_scenario.py"),
                   "--prompt", prompt]
        elif mode == "proc" and events:
            # Build --events arg from event list
            events_arg = ",".join(f"{e['event']}:{e['duration_ms']}" for e in events)
            cmd = [sys.executable, str(ROOT / "layer_1" / "generate_scenario.py"),
                   "--events", events_arg]
        else:
            # Fallback: use existing CSV
            return web.json_response({
                "success": True,
                "csv_path": str(CSV_PATH),
                "biome": biome,
                "scenario": "existing",
                "duration_ms": 5000,
            })

        print(f"[SERVER] Running Layer 1: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode == 0:
            import pandas as pd
            df = pd.read_csv(str(CSV_PATH))
            dur = float(df["timestamp_ms"].max()) if "timestamp_ms" in df.columns else len(df)
            return web.json_response({
                "success"    : True,
                "csv_path"   : str(CSV_PATH),
                "biome"      : biome,
                "scenario"   : prompt or (", ".join(e["event"] for e in events)),
                "duration_ms": dur,
            })
        else:
            errmsg = stderr.decode(errors="replace")
            print(f"[SERVER] Layer 1 error: {errmsg}")
            # Fallback to existing CSV
            return web.json_response({
                "success"    : True,
                "csv_path"   : str(CSV_PATH),
                "biome"      : biome,
                "scenario"   : prompt,
                "duration_ms": 5000,
                "warning"    : "Used existing data (LLM error)",
            })

    except asyncio.TimeoutError:
        return web.json_response({"success": False, "error": "Layer 1 timed out"}, status=504)
    except Exception as ex:
        return web.json_response({"success": False, "error": str(ex)}, status=500)


# ─── WebSocket broadcast ───────────────────────────────────────────────────────

async def broadcast(msg: str):
    dead = set()
    for ws in list(browser_clients):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.add(ws)
    browser_clients.difference_update(dead)


# ─── CSV playback ──────────────────────────────────────────────────────────────

async def playback_csv():
    global current_df, pi_client, browser_ready

    if current_df is None:
        return

    # Wait up to 5 seconds for the browser to finish building the 3D world
    # before we start streaming frames (prevents the rushing/buffering problem)
    for _ in range(50):          # 50 x 100ms = 5 seconds max
        if browser_ready:
            break
        await asyncio.sleep(0.1)

    if not browser_ready:
        print("[SERVER] Warning: browser_ready not received, starting anyway")

    # Extra buffer so camera/scene is stable before first frame arrives
    await asyncio.sleep(0.3)

    interval = 1.0 / PLAYBACK_FPS   # 50ms at 20fps
    rows     = len(current_df)
    has_ts   = "timestamp_ms" in current_df.columns
    total_frames = rows // FRAME_STEP
    est_secs = total_frames / PLAYBACK_FPS

    print(f"[SERVER] Playback: {rows:,} rows -> {total_frames} frames "
          f"@ {PLAYBACK_FPS}fps (~{est_secs:.1f}s real-time)")

    for i in range(0, rows, FRAME_STEP):
        row  = current_df.iloc[i]
        t_ms = float(row["timestamp_ms"]) if has_ts else float(i)

        frame = {
            "type"    : "sensor_frame",
            "t_ms"    : t_ms,
            "label"   : int(row.get("label", 0)),
            "ax"      : float(row["ax"]),
            "ay"      : float(row["ay"]),
            "az"      : float(row["az"]),
            "gx"      : float(row["gx"]),
            "gy"      : float(row["gy"]),
            "gz"      : float(row["gz"]),
            "hg_ax"   : float(row["hg_ax"]),
            "hg_ay"   : float(row["hg_ay"]),
            "hg_az"   : float(row["hg_az"]),
            "progress": round(i / rows, 4),
        }
        raw = json.dumps(frame)
        await broadcast(raw)

        if pi_client and not pi_client.closed:
            try:
                await pi_client.send_str(raw)
            except Exception:
                pi_client = None

        await asyncio.sleep(interval)

    await broadcast(json.dumps({"type": "playback_done"}))
    print("[SERVER] Playback complete.")


# ─── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_ws(request):
    global current_df, playback_task, pi_client, current_meta

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    client_type = "unknown"

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                mtype = data.get("type", "")

                if mtype == "browser_connect":
                    client_type = "browser"
                    browser_clients.add(ws)
                    print(f"[SERVER] Browser connected ({len(browser_clients)} total)")
                    await ws.send_str(json.dumps({
                        "type"        : "server_hello",
                        "pi_connected": pi_client is not None and not pi_client.closed,
                        "has_data"    : current_df is not None,
                    }))

                elif mtype == "pi_connect":
                    client_type = "pi"
                    pi_client   = ws
                    print("[SERVER] Raspberry Pi connected!")
                    await broadcast(json.dumps({"type": "pi_status", "connected": True}))

                elif mtype == "start_playback":
                    import pandas as pd
                    browser_ready = False   # reset: wait for browser_ready before streaming
                    csv_path = data.get("csv_path") or str(CSV_PATH)
                    if not os.path.exists(csv_path):
                        await ws.send_str(json.dumps({"type":"error","msg":f"CSV not found: {csv_path}"}))
                        continue

                    current_df   = pd.read_csv(csv_path)
                    current_meta = {
                        "biome"   : data.get("biome","hill_station"),
                        "scenario": data.get("scenario",""),
                        "csv_path": csv_path,
                    }

                    dur = float(current_df["timestamp_ms"].max()) if "timestamp_ms" in current_df.columns else len(current_df)
                    await broadcast(json.dumps({
                        "type"       : "playback_start",
                        "rows"       : len(current_df),
                        "duration_ms": dur,
                        **current_meta,
                    }))

                    if playback_task and not playback_task.done():
                        playback_task.cancel()
                    playback_task = asyncio.create_task(playback_csv())

                elif mtype == "pi_verdict":
                    await broadcast(msg.data)   # relay Pi results to browsers

                elif mtype == "browser_ready":
                    # Browser has finished building the 3D world - safe to stream now
                    browser_ready = True
                    print("[SERVER] Browser 3D world ready - streaming will begin")

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break

    except Exception as ex:
        print(f"[SERVER] WS error ({client_type}): {ex}")
    finally:
        browser_clients.discard(ws)
        if ws == pi_client:
            pi_client = None
            await broadcast(json.dumps({"type": "pi_status", "connected": False}))
        if client_type == "browser":
            print(f"[SERVER] Browser disconnected ({len(browser_clients)} remain)")

    return ws


# ─── App setup ────────────────────────────────────────────────────────────────

def build_app():
    app = web.Application()
    app.router.add_get("/",              handle_index)
    app.router.add_get("/index.html",    handle_index)
    app.router.add_get("/dashboard",     handle_dashboard)
    app.router.add_get("/dashboard.html",handle_dashboard)
    app.router.add_get("/{name}",        handle_static)
    app.router.add_post("/api/generate", handle_generate)
    app.router.add_get("/ws",            handle_ws)
    return app


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=" * 58)
    print("  SMART AIRBAG HELMET - UNIFIED SERVER")
    print("=" * 58)
    print("  Open in browser ->  http://localhost:5500")
    print("  WebSocket       ->  ws://localhost:5500/ws")
    print("  Layer 1 API     ->  POST /api/generate")
    print("=" * 58)

    app = build_app()
    # Listen on both IPv4 (0.0.0.0) and IPv6 (::) so the Raspberry Pi
    # can connect regardless of whether the local network routes via IPv4 or IPv6
    web.run_app(app, host=["0.0.0.0", "::"], port=5500, access_log=None)
