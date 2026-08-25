"""
01_hardware_sensors/sensor_reader.py
------------------------------------
Unified sensor reader for the Smart Helmet system.

Supports three modes (auto-negotiated in priority order):
    1. Direct I2C  — MPU6050 on SMBus + ADXL377 via MCP3008 ADC
    2. Serial UART — ESP32 streams CSV lines at 115200 baud
    3. Simulation  — Synthetic baseline noise (automatic fallback)

The reader also handles mid-stream disconnects: if a hardware read
fails at runtime, it logs a warning and returns synthetic data for
that sample so the pipeline never crashes during a demo.
"""

import time
import math
import logging
import random

log = logging.getLogger("sensor_reader")


class SensorReader:
    """
    Unified sensor interface with automatic hardware fallback.

    Args:
        mode:     'auto' | 'i2c' | 'serial' | 'simulate'
        port:     Serial port path (only used in serial mode)
        baud:     Serial baud rate
        simulate: Force simulation mode (overrides mode)
    """

    MODE_I2C      = "i2c"
    MODE_SERIAL   = "serial"
    MODE_SIMULATE = "simulate"

    def __init__(self, mode="auto", port="/dev/ttyUSB0", baud=115200, simulate=False):
        self.active_mode = self.MODE_SIMULATE  # default fallback
        self.ser = None
        self.mpu = None
        self.adxl = None
        self._read_errors = 0

        # Simulation state machine parameters
        self.step_counter = 0
        self.sim_state = "normal"  # "normal" | "near_crash" | "crash"
        self.sim_ticks = 0

        if simulate:
            log.info("[SENSOR] Forced simulation mode.")
            return

        # --- Try modes in priority order ---
        if mode in ("auto", "i2c"):
            if self._try_i2c():
                return

        if mode in ("auto", "serial"):
            if self._try_serial(port, baud):
                return

        log.warning("[SENSOR FALLBACK] No hardware detected. Running synthetic simulation.")

    # -----------------------------------------------------------------
    #  Hardware negotiation helpers
    # -----------------------------------------------------------------

    def _try_i2c(self) -> bool:
        """Attempt direct I2C connection to MPU6050 (+ optional ADXL377)."""
        try:
            from mpu6050_i2c import MPU6050I2C
        except ImportError:
            try:
                from importlib import import_module
                mod = import_module("01_hardware_sensors.mpu6050_i2c")
                MPU6050I2C = mod.MPU6050I2C
            except Exception:
                return False

        try:
            self.mpu = MPU6050I2C()
            if self.mpu.bus is None:
                return False
            self.active_mode = self.MODE_I2C
            log.info("[SENSOR] I2C mode active (MPU6050 direct).")

            # Optional: try ADXL377
            try:
                from adxl377_reader import ADXL377Reader
            except ImportError:
                try:
                    mod = import_module("01_hardware_sensors.adxl377_reader")
                    ADXL377Reader = mod.ADXL377Reader
                except Exception:
                    ADXL377Reader = None

            if ADXL377Reader:
                self.adxl = ADXL377Reader()
                if self.adxl.mcp:
                    log.info("[SENSOR] ADXL377 high-g ADC active.")
                else:
                    log.info("[SENSOR] ADXL377 ADC not found — mirroring MPU6050 accel.")
                    self.adxl = None
            return True
        except Exception as e:
            log.debug(f"I2C init failed: {e}")
            return False

    def _try_serial(self, port, baud) -> bool:
        """Attempt UART serial connection to ESP32."""
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.active_mode = self.MODE_SERIAL
            log.info(f"[SENSOR] Serial mode active on {port} @ {baud} baud.")
            return True
        except Exception as e:
            log.debug(f"Serial init failed: {e}")
            return False

    # -----------------------------------------------------------------
    #  Read sample
    # -----------------------------------------------------------------

    def read_sample(self):
        """
        Returns a dict with keys:
            ax, ay, az, gx, gy, gz, hg_ax, hg_ay, hg_az
        Never returns None — falls back to synthetic on any error.
        """
        if self.active_mode == self.MODE_I2C:
            return self._read_i2c()
        elif self.active_mode == self.MODE_SERIAL:
            return self._read_serial()
        else:
            return self._read_synthetic()

    def _read_i2c(self):
        try:
            data = self.mpu.read_motion6()
            if data is None:
                raise RuntimeError("MPU6050 returned None")
            self._read_errors = 0

            # Add high-g channels
            if self.adxl:
                hg_ax, hg_ay, hg_az = self.adxl.read_high_g(
                    data["ax"], data["ay"], data["az"]
                )
            else:
                hg_ax, hg_ay, hg_az = data["ax"], data["ay"], data["az"]

            data["hg_ax"] = hg_ax
            data["hg_ay"] = hg_ay
            data["hg_az"] = hg_az
            return data
        except Exception as e:
            self._read_errors += 1
            if self._read_errors <= 3:
                log.warning(f"[SENSOR I2C READ ERROR #{self._read_errors}] {e}")
            if self._read_errors >= 10:
                log.error("[SENSOR] Too many I2C errors — switching to simulation.")
                self.active_mode = self.MODE_SIMULATE
            return self._read_synthetic()

    def _read_serial(self):
        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return self._read_synthetic()

            parts = [float(p) for p in line.split(",")]
            if len(parts) >= 9:
                return {
                    "ax": parts[0], "ay": parts[1], "az": parts[2],
                    "gx": parts[3], "gy": parts[4], "gz": parts[5],
                    "hg_ax": parts[6], "hg_ay": parts[7], "hg_az": parts[8],
                }
            elif len(parts) >= 6:
                return {
                    "ax": parts[0], "ay": parts[1], "az": parts[2],
                    "gx": parts[3], "gy": parts[4], "gz": parts[5],
                    "hg_ax": parts[0], "hg_ay": parts[1], "hg_az": parts[2],
                }
        except Exception as e:
            self._read_errors += 1
            if self._read_errors <= 3:
                log.warning(f"[SENSOR SERIAL READ ERROR #{self._read_errors}] {e}")
            if self._read_errors >= 10:
                log.error("[SENSOR] Too many serial errors — switching to simulation.")
                self.active_mode = self.MODE_SIMULATE

        return self._read_synthetic()

    def _read_synthetic(self):
        """
        Dynamically generates realistic, frequency-aware sensor signals at 1000 Hz.
        Cycles through Normal riding -> Near-Crash (pothole) -> Normal -> Severe Crash.
        """
        self.step_counter += 1
        self.sim_ticks += 1
        t = self.step_counter * 0.001  # Time step in seconds (1 ms per step)

        # State transition controller
        if self.sim_state == "normal" and self.step_counter > 1000 and self.step_counter <= 1200:
            self.sim_state = "near_crash"
            self.sim_ticks = 0
            log.info("[SIMULATION STATE] Transitioning to: NEAR_CRASH (Pothole)")
        elif self.sim_state == "near_crash" and self.sim_ticks > 200:
            self.sim_state = "normal"
            self.sim_ticks = 0
            log.info("[SIMULATION STATE] Transitioning to: NORMAL (Recovered)")
        elif self.sim_state == "normal" and self.step_counter > 2000 and self.step_counter <= 2200:
            self.sim_state = "crash"
            self.sim_ticks = 0
            log.info("[SIMULATION STATE] Transitioning to: CRASH (Severe Impact)")

        # State signal generator
        if self.sim_state == "normal":
            # Normal riding engine vibration (35 Hz & 60 Hz harmonics) + road noise
            ax = 0.3 * math.sin(2 * math.pi * 35 * t) + random.gauss(0, 0.1)
            ay = 0.2 * math.sin(2 * math.pi * 60 * t) + random.gauss(0, 0.1)
            az = 9.81 + 0.4 * math.sin(2 * math.pi * 10 * t) + random.gauss(0, 0.15)
            gx = 2.0 * math.sin(2 * math.pi * 5 * t) + random.gauss(0, 0.5)
            gy = 1.5 * math.sin(2 * math.pi * 8 * t) + random.gauss(0, 0.5)
            gz = 3.0 * math.sin(2 * math.pi * 4 * t) + random.gauss(0, 0.5)
        
        elif self.sim_state == "near_crash":
            # Near-Crash (e.g. sharp bump/braking): Decaying shock spike
            decay = math.exp(-self.sim_ticks / 50.0) # decay factor over 50 ms
            ax = 4.0 * decay * math.sin(2 * math.pi * 80 * t) + random.gauss(0, 0.2)
            ay = 2.0 * decay * math.sin(2 * math.pi * 80 * t) + random.gauss(0, 0.2)
            az = 9.81 + 6.0 * decay * math.sin(2 * math.pi * 50 * t) + random.gauss(0, 0.3)
            gx = 40.0 * decay * math.sin(2 * math.pi * 15 * t) + random.gauss(0, 1.0)
            gy = 30.0 * decay * math.sin(2 * math.pi * 15 * t) + random.gauss(0, 1.0)
            gz = 50.0 * decay * math.sin(2 * math.pi * 15 * t) + random.gauss(0, 1.0)
            
        else:  # "crash" state
            # Phase 1: Ramping rotational tumbling and loss of gravity (0-100 ms)
            # Phase 2: Decelerating massive ground impact spike (100 ms+)
            if self.sim_ticks < 100:
                # Tumbling phase
                tumb_factor = self.sim_ticks / 100.0
                ax = 5.0 * tumb_factor * math.sin(2 * math.pi * 15 * t) + random.gauss(0, 0.5)
                ay = 4.0 * tumb_factor * math.sin(2 * math.pi * 20 * t) + random.gauss(0, 0.5)
                # Gravity loss / freefall (az drops down near 0)
                az = (9.81 * (1.0 - tumb_factor)) + random.gauss(0, 0.2)
                gx = 320.0 * tumb_factor * math.sin(2 * math.pi * 10 * t)
                gy = 180.0 * tumb_factor * math.sin(2 * math.pi * 10 * t)
                gz = 250.0 * tumb_factor * math.sin(2 * math.pi * 10 * t)
            else:
                # Sudden high-g ground collision impact
                impact_ticks = self.sim_ticks - 100
                decay = math.exp(-impact_ticks / 30.0)
                ax = 26.0 * decay + random.gauss(0, 0.5)
                ay = 12.0 * decay + random.gauss(0, 0.5)
                az = 18.0 * decay + random.gauss(0, 0.5)
                gx = 350.0 * decay
                gy = 120.0 * decay
                gz = 200.0 * decay

        return {
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "hg_ax": ax, "hg_ay": ay, "hg_az": az,
        }


    # -----------------------------------------------------------------
    #  Cleanup
    # -----------------------------------------------------------------

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        log.info("[SENSOR] Reader closed.")
