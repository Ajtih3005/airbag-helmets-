"""
raspberry_pi_interface.py
--------------------------
Real-time hardware interface for Smart Airbag Helmet on Raspberry Pi.

This module:
    1. Reads live IMU data from ESP32 via UART (serial port)
    2. Maintains a rolling 200-sample (200 ms) buffer at 1000 Hz
    3. Feeds each window through the ML pipeline (feature extraction + prediction)
    4. Triggers GPIO pins for airbag deployment and near-crash warning LED
    5. Sends emergency SMS via SIM800L over a second UART

Hardware wiring (default BCM pin numbers):
    GPIO 17  -> MOSFET gate -> solenoid valve / servo (airbag)
    GPIO 27  -> LED / buzzer (near-crash warning)
    /dev/ttyUSB0 or /dev/ttyS0  -> ESP32 UART TX
    /dev/ttyAMA0                 -> SIM800L UART TX

ESP32 sends CSV lines over serial at 115200 baud:
    ax,ay,az,gx,gy,gz,hg_ax,hg_ay,hg_az\n
    e.g.:  0.12,-0.05,9.83,1.2,-0.8,0.3,0.12,-0.05,9.83\n

Usage:
    python src/raspberry_pi_interface.py --port /dev/ttyUSB0 --model-dir models
    python src/raspberry_pi_interface.py --simulate              # no hardware needed
"""

import os
import sys
import time
import signal
import logging
import argparse
from collections import deque

import numpy as np
import pandas as pd
import joblib

# -----------------------------------------------
#  PATH SETUP
# -----------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.feature_engineering import extract_features_window
    from src.data_generator      import WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ
    from src.predict             import (is_deployed, set_deployed_lock,
                                         CRASH_PROB_THRESHOLD,
                                         CONSECUTIVE_WINDOWS_REQ,
                                         NEAR_CRASH_THRESHOLD,
                                         send_sms_alert, log_to_sd)
except ImportError:
    from feature_engineering import extract_features_window
    from data_generator      import WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ
    from predict             import (is_deployed, set_deployed_lock,
                                     CRASH_PROB_THRESHOLD,
                                     CONSECUTIVE_WINDOWS_REQ,
                                     NEAR_CRASH_THRESHOLD,
                                     send_sms_alert, log_to_sd)

# -----------------------------------------------
#  LOGGING
# -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airbag_rpi")

# -----------------------------------------------
#  SENSOR / GPIO CONFIG
# -----------------------------------------------
SENSOR_COLS    = ["ax", "ay", "az", "gx", "gy", "gz"]
HG_COLS        = ["hg_ax", "hg_ay", "hg_az"]
ALL_COLS       = SENSOR_COLS + HG_COLS

AIRBAG_PIN     = 17    # BCM — connect to MOSFET gate (drives solenoid)
WARNING_PIN    = 27    # BCM — LED or buzzer for near-crash warning

BAUD_RATE      = 115200
LOG_PATH       = os.path.join(project_root, "logs", "hardware_log.csv")


# -----------------------------------------------
#  GPIO HELPERS
# -----------------------------------------------

def gpio_setup():
    """Initialize GPIO pins. Returns True if GPIO is available."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(AIRBAG_PIN,  GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(WARNING_PIN, GPIO.OUT, initial=GPIO.LOW)
        log.info(f"GPIO initialized: airbag=BCM{AIRBAG_PIN}, warning=BCM{WARNING_PIN}")
        return True
    except ImportError:
        log.warning("RPi.GPIO not available — running in simulation mode.")
        return False
    except Exception as e:
        log.error(f"GPIO setup error: {e}")
        return False


def gpio_fire_airbag():
    """
    Pull airbag GPIO pin HIGH to trigger MOSFET -> solenoid -> CO2 release.
    This is a one-shot, permanent HIGH (airbag stays deployed).
    """
    try:
        import RPi.GPIO as GPIO
        GPIO.output(AIRBAG_PIN, GPIO.HIGH)
        log.critical(f"*** AIRBAG FIRED: GPIO {AIRBAG_PIN} = HIGH ***")
    except Exception:
        log.critical("*** AIRBAG FIRED (SIMULATION) ***")


def gpio_pulse_warning(n_pulses: int = 3, pulse_ms: int = 50):
    """Pulse the warning LED/buzzer n times."""
    try:
        import RPi.GPIO as GPIO
        for _ in range(n_pulses):
            GPIO.output(WARNING_PIN, GPIO.HIGH)
            time.sleep(pulse_ms / 1000.0)
            GPIO.output(WARNING_PIN, GPIO.LOW)
            time.sleep(pulse_ms / 1000.0)
    except Exception:
        log.info(f"[WARNING PULSE x{n_pulses} — simulation]")


def gpio_cleanup():
    try:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
        log.info("GPIO cleanup done.")
    except Exception:
        pass


# -----------------------------------------------
#  SERIAL READER
# -----------------------------------------------

class SerialIMUReader:
    """
    Reads CSV-formatted IMU lines from ESP32 over UART.

    Expected format (9 floats per line):
        ax,ay,az,gx,gy,gz,hg_ax,hg_ay,hg_az\n
    """

    def __init__(self, port: str, baud: int = BAUD_RATE, timeout: float = 1.0):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        log.info(f"Serial port opened: {port} @ {baud} baud")

    def read_sample(self) -> dict | None:
        """
        Read one line from serial. Returns dict of sensor values or None on error.
        """
        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return None
            parts = [float(x) for x in line.split(",")]
            if len(parts) >= 9:
                return dict(zip(ALL_COLS, parts[:9]))
            elif len(parts) >= 6:
                # Fall back: only MPU6050 data (duplicate to hg channels)
                d = dict(zip(SENSOR_COLS, parts[:6]))
                d["hg_ax"] = d["ax"]
                d["hg_ay"] = d["ay"]
                d["hg_az"] = d["az"]
                return d
            return None
        except (ValueError, UnicodeDecodeError) as e:
            log.debug(f"Serial parse error: {e}")
            return None

    def close(self):
        if self.ser.is_open:
            self.ser.close()


# -----------------------------------------------
#  SYNTHETIC STREAM (for --simulate mode)
# -----------------------------------------------

class SyntheticIMUStream:
    """
    Generates a synthetic Normal -> Near-Crash -> Crash scenario at 1000 Hz.
    Used when no physical ESP32 is connected.
    """

    def __init__(self):
        try:
            from src.data_generator import _normal, _near_crash, _crash
        except ImportError:
            from data_generator import _normal, _near_crash, _crash

        rng = np.random.default_rng(777)
        self._samples = []
        self._labels  = []

        for gen, lbl, n in [
            (_normal,     0, 1000),
            (_near_crash, 1, 500),
            (_crash,      2, 500),
            (_normal,     0, 500),
        ]:
            sig = gen(n, rng)
            for i in range(n):
                self._samples.append({col: float(sig[col][i]) for col in ALL_COLS})
                self._labels.append(lbl)

        self._idx = 0

    def read_sample(self) -> dict | None:
        if self._idx >= len(self._samples):
            return None
        s = self._samples[self._idx]
        self._idx += 1
        return s

    def true_label(self) -> int:
        return self._labels[self._idx - 1] if self._idx > 0 else -1

    def close(self):
        pass


# -----------------------------------------------
#  MAIN INFERENCE LOOP
# -----------------------------------------------

def run_inference_loop(
    model,
    meta: dict,
    stream,
    hardware: bool  = False,
    verbose: bool   = True,
    max_samples: int = 0,    # 0 = run until Ctrl+C or stream ends
):
    """
    Core real-time inference loop.

    Reads samples one by one from `stream`, maintains a rolling buffer of
    WINDOW_SIZE samples, and evaluates an ML window every STRIDE samples.

    Safety gate:
        Airbag fires only after CONSECUTIVE_WINDOWS_REQ consecutive windows
        all predict Crash above CRASH_PROB_THRESHOLD.
    """
    feature_names      = meta.get("feature_names", [])
    buffer             = deque(maxlen=WINDOW_SIZE * 2)  # Circular buffer
    samples_since_pred = 0
    consecutive_crash  = 0
    deploy_fired       = is_deployed()
    n_windows          = 0
    n_samples          = 0

    if deploy_fired:
        log.error("Airbag already deployed (lock file exists). Reset lock to restart.")
        return

    log.info(f"Inference loop started | window={WINDOW_SIZE}smp | stride={STRIDE}smp | hw={hardware}")

    def handle_sigint(sig, frame):
        log.info("\nStopped by user.")
        gpio_cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    t_start = time.perf_counter()

    while True:
        sample = stream.read_sample()
        if sample is None:
            log.info("Stream ended.")
            break

        buffer.append(sample)
        n_samples     += 1
        samples_since_pred += 1

        # Only evaluate when we have enough samples AND at stride boundary
        if len(buffer) >= WINDOW_SIZE and samples_since_pred >= STRIDE:
            samples_since_pred = 0

            window_list = list(buffer)[-WINDOW_SIZE:]
            window_df   = pd.DataFrame(window_list)

            # --- ML INFERENCE ---
            t_inf = time.perf_counter()
            feats    = extract_features_window(window_df)
            feat_df  = pd.DataFrame([[feats[f] for f in feature_names]], columns=feature_names)
            pred_label = int(model.predict(feat_df)[0])
            infer_ms   = (time.perf_counter() - t_inf) * 1000

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feat_df)[0]
            else:
                proba = np.zeros(3); proba[pred_label] = 1.0
            while len(proba) < 3:
                proba = np.append(proba, 0.0)

            crash_prob    = float(proba[2])
            nc_prob       = float(proba[1])
            label_names   = {0: "Normal", 1: "Near-Crash", 2: "Crash"}
            label_name    = label_names.get(pred_label, "?")

            n_windows += 1
            t_elapsed  = time.perf_counter() - t_start

            if verbose:
                sym = {0: "O", 1: "!", 2: "X"}.get(pred_label, "?")
                log.info(
                    f"[{sym}] {label_name:<12} | "
                    f"Crash={crash_prob:.1%} | NC={nc_prob:.1%} | "
                    f"Infer={infer_ms:.2f}ms | "
                    f"n={n_samples}"
                )

            # --- Near-Crash warning ---
            if pred_label == 1 or (nc_prob > NEAR_CRASH_THRESHOLD and pred_label != 2):
                log.warning(f"Near-Crash detected (NC_prob={nc_prob:.1%})")
                if hardware:
                    gpio_pulse_warning()

            # --- Crash multi-check gate ---
            if crash_prob > CRASH_PROB_THRESHOLD:
                consecutive_crash += 1
                log.warning(
                    f"CRASH gate: {consecutive_crash}/{CONSECUTIVE_WINDOWS_REQ} "
                    f"(p={crash_prob:.1%})"
                )
                if consecutive_crash >= CONSECUTIVE_WINDOWS_REQ and not deploy_fired:
                    log.critical(
                        f"AIRBAG DEPLOY! {consecutive_crash} consecutive crash windows. "
                        f"Infer latency={infer_ms:.2f}ms"
                    )
                    set_deployed_lock()
                    deploy_fired = True
                    if hardware:
                        gpio_fire_airbag()
                        send_sms_alert(
                            f"CRASH at t={int(t_elapsed*1000)}ms | "
                            f"p={crash_prob:.1%} | infer={infer_ms:.1f}ms"
                        )
            else:
                consecutive_crash = 0   # Reset on non-crash window

            # --- Log ---
            log_to_sd({
                "t_ms"       : int(t_elapsed * 1000),
                "label"      : pred_label,
                "label_name" : label_name,
                "crash_prob" : round(crash_prob, 4),
                "nc_prob"    : round(nc_prob, 4),
                "infer_ms"   : round(infer_ms, 4),
                "consecutive": consecutive_crash,
                "deployed"   : deploy_fired,
            }, LOG_PATH)

        if max_samples and n_samples >= max_samples:
            log.info(f"Reached max_samples={max_samples}. Stopping.")
            break

    log.info(f"Loop ended. Samples={n_samples}, Windows={n_windows}, Deployed={deploy_fired}")
    gpio_cleanup()


# -----------------------------------------------
#  CLI
# -----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Airbag Helmet -- Raspberry Pi Interface")
    parser.add_argument("--port",       default="/dev/ttyUSB0",  help="Serial port for ESP32 UART")
    parser.add_argument("--baud",       type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--model-dir",  default="models",         help="Path to models directory")
    parser.add_argument("--simulate",   action="store_true",      help="Use synthetic data stream (no hardware)")
    parser.add_argument("--hardware",   action="store_true",      help="Enable GPIO output on Raspberry Pi")
    parser.add_argument("--no-verbose", action="store_true",      help="Suppress per-window logs")
    parser.add_argument("--max",        type=int, default=0,      help="Max samples to process (0=unlimited)")
    args = parser.parse_args()

    # Load model
    model_path = os.path.join(args.model_dir, "best_model.pkl")
    meta_path  = os.path.join(args.model_dir, "model_meta.pkl")

    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        log.error("Run: python src/train_model.py")
        sys.exit(1)

    model = joblib.load(model_path)
    meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    log.info(f"Model loaded: {meta.get('model_name','?')} | "
             f"Acc={meta.get('accuracy',0):.4f} | "
             f"F1={meta.get('f1_macro',0):.4f}")

    # Initialize GPIO (if hardware mode)
    if args.hardware:
        gpio_setup()

    # Build stream
    if args.simulate:
        log.info("Simulation mode: using synthetic IMU stream")
        stream = SyntheticIMUStream()
    else:
        try:
            stream = SerialIMUReader(args.port, args.baud)
        except ImportError:
            log.error("pyserial not installed. Run: pip install pyserial")
            sys.exit(1)
        except Exception as e:
            log.error(f"Cannot open serial port {args.port}: {e}")
            sys.exit(1)

    try:
        run_inference_loop(
            model    = model,
            meta     = meta,
            stream   = stream,
            hardware = args.hardware,
            verbose  = not args.no_verbose,
            max_samples = args.max,
        )
    finally:
        stream.close()


if __name__ == "__main__":
    main()
