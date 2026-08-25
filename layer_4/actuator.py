"""
layer_4/actuator.py
--------------------
Raspberry Pi GPIO Actuation Engine — Layer 4.

Handles:
    1. CO2 Solenoid Valve trigger via MOSFET gate on GPIO Pin 17
    2. Near-Crash Warning LED / Buzzer pulsing on GPIO Pin 27
    3. Emergency SMS Alert via SIM800L GSM module (UART AT commands)
    4. Safe GPIO cleanup on shutdown

All hardware calls have graceful simulation fallback when not on Pi.
"""

import time

# ─── Pin Configuration ─────────────────────────────────────────────────────────
AIRBAG_PIN       = 17          # BCM — MOSFET gate → CO2 solenoid valve
WARNING_PIN      = 27          # BCM — LED / buzzer → near-crash warning
SIM800L_PORT     = "/dev/ttyAMA0"
SIM800L_BAUD     = 9600
EMERGENCY_NUMBER = "+91XXXXXXXXXX"   # Set your emergency contact number here


class ActuatorEngine:
    """
    Raspberry Pi hardware actuation for CO2 airbag solenoid valve and alert systems.
    Runs with graceful simulation fallback on non-Pi environments (Windows dev).
    """

    def __init__(self,
                 airbag_pin=AIRBAG_PIN,
                 warning_pin=WARNING_PIN,
                 hardware=False,
                 sms_port=SIM800L_PORT,
                 sms_baud=SIM800L_BAUD,
                 emergency_number=EMERGENCY_NUMBER):

        self.airbag_pin       = airbag_pin
        self.warning_pin      = warning_pin
        self.hardware         = hardware
        self.gpio_ready       = False
        self.sms_port         = sms_port
        self.sms_baud         = sms_baud
        self.emergency_number = emergency_number

        # Deployment state — one-shot safety lock
        self.deployed         = False
        self.deploy_time      = None
        self.deploy_reason    = None

        if self.hardware:
            self._init_gpio()

    # ─── GPIO Setup ────────────────────────────────────────────────────────────

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.airbag_pin,  GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.warning_pin, GPIO.OUT, initial=GPIO.LOW)
            self.gpio_ready = True
            print(f"[HARDWARE] GPIO ready — Airbag=BCM{self.airbag_pin}, Warning=BCM{self.warning_pin}")
        except ImportError:
            print("[HARDWARE STUB] RPi.GPIO not available — Simulation mode active.")
            self.hardware = False
        except Exception as e:
            print(f"[HARDWARE STUB] GPIO init failed ({e}) — Simulation mode active.")
            self.hardware = False

    # ─── Airbag Deployment ─────────────────────────────────────────────────────

    def deploy_airbag(self, reason="ML_CRASH_DETECTED"):
        """
        Fire the CO2 solenoid valve by pulling GPIO Pin 17 HIGH.
        One-shot only — subsequent calls are safely ignored.
        """
        if self.deployed:
            print("[ACTUATION] Airbag already deployed — duplicate call ignored.")
            return False

        self.deployed      = True
        self.deploy_time   = time.time()
        self.deploy_reason = reason

        print(f"\n*** AIRBAG DEPLOY TRIGGERED — Reason: {reason} ***\n")

        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self.airbag_pin, GPIO.HIGH)
                print(f"[GPIO] BCM{self.airbag_pin} -> HIGH (CO2 valve OPEN)")
            except Exception as e:
                print(f"[GPIO ERROR] Failed to set pin HIGH: {e}")
        else:
            print("[SIMULATION] CO2 solenoid valve opened / Airbag inflating...")

        return True

    # ─── Near-Crash Warning ────────────────────────────────────────────────────

    def pulse_warning(self, n_pulses=1, pulse_ms=50):
        """
        Pulse the warning LED/buzzer for near-crash state detection.
        Rate-limited to at most once per 600ms to avoid flooding.
        """
        if self.deployed:
            return

        now = time.time()
        if hasattr(self, "_last_pulse") and (now - self._last_pulse < 0.6):
            return
        self._last_pulse = now

        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                for _ in range(n_pulses):
                    GPIO.output(self.warning_pin, GPIO.HIGH)
                    time.sleep(pulse_ms / 1000.0)
                    GPIO.output(self.warning_pin, GPIO.LOW)
                    time.sleep(pulse_ms / 1000.0)
            except Exception as e:
                print(f"[WARNING LED ERROR] {e}")
        else:
            print(f"  [⚠️ BUZZER] Near-Crash Warning Pulsed ({pulse_ms}ms)")

    # ─── Emergency SMS ─────────────────────────────────────────────────────────

    def send_emergency_sms(self, lat=None, lon=None, extra_info=""):
        """
        Send emergency SMS via SIM800L GSM module over UART using AT commands.
        Falls back to log-only if pyserial is missing or UART unavailable.
        """
        location = f" Location: {lat},{lon}" if (lat and lon) else ""
        message  = (
            f"EMERGENCY: Helmet airbag deployed! Rider may be injured.{location}"
            f" Reason: {self.deploy_reason or 'CRASH'}."
            f" {extra_info}".strip()
        )

        print(f"[SMS] Sending emergency alert to {self.emergency_number}")
        print(f"[SMS] Message: {message}")

        try:
            import serial
            with serial.Serial(self.sms_port, self.sms_baud, timeout=2) as ser:
                ser.write(b"AT+CMGF=1\r\n")
                time.sleep(0.5)
                _ = ser.read(ser.in_waiting or 1)

                ser.write(f'AT+CMGS="{self.emergency_number}"\r\n'.encode())
                time.sleep(0.5)

                ser.write((message + "\x1A").encode())
                time.sleep(3)

                resp = ser.read(ser.in_waiting or 1).decode(errors="ignore")
                if "+CMGS" in resp:
                    print("[SMS] Emergency SMS delivered successfully.")
                else:
                    print(f"[SMS] Uncertain delivery — Response: {resp.strip()}")

        except ImportError:
            print(f"[SMS STUB] pyserial not installed — Would send: '{message}'")
        except Exception as e:
            print(f"[SMS ERROR] {e} — Message logged locally.")

    # ─── GPIO Cleanup ──────────────────────────────────────────────────────────

    def cleanup(self):
        """Release all GPIO pins safely."""
        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                print("[HARDWARE] GPIO cleanup complete.")
            except Exception:
                pass
