"""
layer_pi/bundle_for_pi.py
--------------------------
Copies all Pi-required layer files into this layer_pi/ directory,
making it a fully self-contained folder you can SCP to the Raspberry Pi.

Run once on your laptop before transferring:
    python layer_pi/bundle_for_pi.py

Then copy the whole folder to the Pi:
    scp -r layer_pi/ pi@raspberrypi.local:~/smart_helmet/
"""

import os
import shutil

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST     = os.path.dirname(os.path.abspath(__file__))

FILES_TO_COPY = [
    # Layer 3 — ML Detection Engine
    ("layer_3/feature_engineering.py", "feature_engineering.py"),
    ("layer_3/sentinel.py",            "sentinel.py"),
    ("layer_3/arbiter.py",             "arbiter.py"),
    ("layer_3/detector.py",            "detector.py"),
    ("layer_3/compiled_models.js",     "compiled_models.js"),   # kept for reference

    # Layer 4 — Hardware Actuation
    ("layer_4/actuator.py",            "actuator.py"),

    # Layer 5 — Post-Crash Analysis
    ("layer_5/post_crash_analysis.py", "post_crash_analysis.py"),
]

def bundle():
    print("=" * 55)
    print("  SMART AIRBAG HELMET — BUNDLING Pi PACKAGE (L3/L4/L5)")
    print("=" * 55)

    os.makedirs(os.path.join(PI_DIR, "l3"), exist_ok=True)
    os.makedirs(os.path.join(PI_DIR, "l4"), exist_ok=True)
    os.makedirs(os.path.join(PI_DIR, "l5"), exist_ok=True)
    os.makedirs(os.path.join(PI_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(PI_DIR, "logs"), exist_ok=True)

    ok = 0

    # Layer 3 ML files -> layer_pi/l3/
    l3_files = ["feature_engineering.py", "sentinel.py", "arbiter.py", "detector.py"]
    for fname in l3_files:
        src = os.path.join(ROOT, "layer_3", fname)
        dst = os.path.join(PI_DIR, "l3", fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [COPIED]  layer_3/{fname} -> layer_pi/l3/{fname}")
            ok += 1

    # Layer 4 Actuation -> layer_pi/l4/
    src = os.path.join(ROOT, "layer_4", "actuator.py")
    dst = os.path.join(PI_DIR, "l4", "actuator.py")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("  [COPIED]  layer_4/actuator.py -> layer_pi/l4/actuator.py")
        ok += 1

    # Layer 5 Post-Crash -> layer_pi/l5/
    src = os.path.join(ROOT, "layer_5", "post_crash_analysis.py")
    dst = os.path.join(PI_DIR, "l5", "post_crash_analysis.py")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("  [COPIED]  layer_5/post_crash_analysis.py -> layer_pi/l5/post_crash_analysis.py")
        ok += 1

    # ML Models -> layer_pi/models/
    models_src = os.path.join(ROOT, "models")
    models_dst = os.path.join(PI_DIR, "models")
    if os.path.exists(models_src):
        for m in os.listdir(models_src):
            s_file = os.path.join(models_src, m)
            d_file = os.path.join(models_dst, m)
            if os.path.isfile(s_file):
                shutil.copy2(s_file, d_file)
                print(f"  [COPIED]  models/{m} -> layer_pi/models/{m}")
        ok += 1

    print(f"\n  {ok} items bundled into layer_pi/")
    print(f"  Destination: {DEST}")
    print("\n  Next steps for Raspberry Pi:")
    print("  1. Copy layer_pi folder to SD card or SCP to Pi:")
    print("     scp -r layer_pi/ pi@<PI_IP>:~/smart_helmet/")
    print("  2. On Pi: cd ~/smart_helmet && pip install -r requirements.txt")
    print("  3. On Pi: python pi_ws_client.py --server ws://<LAPTOP_IP>:5500/ws")
    print("     (For real GPIO actuation, add --hardware flag)")
    print("=" * 55)

if __name__ == "__main__":
    bundle()
