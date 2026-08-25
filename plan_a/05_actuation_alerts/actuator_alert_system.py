"""
05_actuation_alerts/actuator_alert_system.py
---------------------------------------------
Actuator and Emergency SMS Alert Handler.

Handles:
    1. GPIO MOSFET gate drive for CO2 solenoid valve / demo servo
    2. Warning LED / buzzer pulsing for near-crash state
    3. SIM800L GSM module emergency SMS via UART AT commands
    4. GPIO cleanup on exit

All hardware calls have graceful simulation fallback.
"""

import time
import logging
import os

log = logging.getLogger("actuator_alert")

# Default pin & serial config
AIRBAG_PIN      = 17            # BCM — MOSFET gate
WARNING_PIN     = 27            # BCM — LED / buzzer
SIM800L_PORT    = "/dev/ttyAMA0"
SIM800L_BAUD    = 9600
EMERGENCY_NUMBER = "+91XXXXXXXXXX"   # Replace with actual number


class ActuatorAlertSystem:
    """
    Controls physical actuation (airbag solenoid / servo) and
    emergency SMS alert via SIM800L.
    """

    def __init__(self, airbag_pin=AIRBAG_PIN, warning_pin=WARNING_PIN,
                 hardware=False, sms_port=SIM800L_PORT, sms_baud=SIM800L_BAUD,
                 emergency_number=EMERGENCY_NUMBER):
        self.airbag_pin = airbag_pin
        self.warning_pin = warning_pin
        self.hardware = hardware
        self.gpio_ready = False
        self.sms_port = sms_port
        self.sms_baud = sms_baud
        self.emergency_number = emergency_number
        self.deployed = False

        if self.hardware:
            self._init_gpio()

    # -----------------------------------------------------------------
    #  GPIO setup
    # -----------------------------------------------------------------

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.airbag_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.warning_pin, GPIO.OUT, initial=GPIO.LOW)
            self.gpio_ready = True
            log.info(f"[HARDWARE] GPIO ready — airbag=BCM{self.airbag_pin}, warning=BCM{self.warning_pin}")
        except ImportError:
            log.warning("[HARDWARE STUB] RPi.GPIO not installed — simulation mode.")
            self.hardware = False
        except Exception as e:
            log.warning(f"[HARDWARE STUB] GPIO setup failed ({e}) — simulation mode.")
            self.hardware = False

    # -----------------------------------------------------------------
    #  Airbag deployment
    # -----------------------------------------------------------------

    def _trigger_arduino(self, command="DEPLOY"):
        """Attempts to send trigger command over USB Serial to connected Arduino."""
        try:
            import serial
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
            for port in ports:
                try:
                    with serial.Serial(port, 115200, timeout=0.5) as ser:
                        time.sleep(0.1)
                        ser.write(f"{command}\n".encode())
                        log.info(f"[ARDUINO TRIGGER] Sent '{command}' to Arduino on {port}")
                        return True
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"[ARDUINO SERIAL INFO] {e}")
        return False

    def deploy_airbag(self, reason="CRASH_DETECTION"):
        """
        Pull MOSFET gate HIGH / Send Serial DEPLOY command to Arduino.
        On real hardware: opens CO2 solenoid valve.
        In simulation: logs the event.
        """
        if self.deployed:
            log.info("[ACTUATION] Already deployed — ignoring duplicate call.")
            return

        self.deployed = True
        log.critical(f"*** AIRBAG DEPLOY: {reason} ***")

        # 1. Try Arduino USB Serial trigger
        arduino_fired = self._trigger_arduino("DEPLOY")

        # 2. Try Raspberry Pi GPIO trigger
        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self.airbag_pin, GPIO.HIGH)
                log.info(f"[GPIO] BCM{self.airbag_pin} -> HIGH (MOSFET gate open)")
            except Exception as e:
                log.error(f"[GPIO ERROR] {e}")
        elif not arduino_fired:
            log.info("[SIMULATION] Servo rotated 90° / CO2 valve opened.")

    # -----------------------------------------------------------------
    #  Near-crash warning
    # -----------------------------------------------------------------

    def pulse_warning_led(self, n_pulses=3, pulse_ms=50):
        """Pulse warning LED/buzzer for near-crash state."""
        self._trigger_arduino("WARN")

        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                for _ in range(n_pulses):
                    GPIO.output(self.warning_pin, GPIO.HIGH)
                    time.sleep(pulse_ms / 1000.0)
                    GPIO.output(self.warning_pin, GPIO.LOW)
                    time.sleep(pulse_ms / 1000.0)
            except Exception as e:
                log.error(f"[WARNING LED ERROR] {e}")
        else:
            log.info(f"[SIMULATION] Warning LED pulsed {n_pulses}x")


    # -----------------------------------------------------------------
    #  Emergency SMS via SIM800L
    # -----------------------------------------------------------------

    def send_emergency_sms(self, message="EMERGENCY: Crash detected — airbag deployed!"):
        """
        Send SMS via SIM800L GSM module using AT commands over UART.
        Falls back to log-only if pyserial is missing or port unavailable.
        """
        log.info(f"[SMS] Sending emergency SMS to {self.emergency_number}")

        try:
            import serial
            with serial.Serial(self.sms_port, self.sms_baud, timeout=2) as ser:
                # Set SMS to text mode
                ser.write(b"AT+CMGF=1\r\n")
                time.sleep(0.5)
                response = ser.read(ser.in_waiting or 1).decode(errors="ignore")
                log.debug(f"[SMS AT+CMGF] {response.strip()}")

                # Set recipient number
                cmd = f'AT+CMGS="{self.emergency_number}"\r\n'
                ser.write(cmd.encode())
                time.sleep(0.5)

                # Write message body + Ctrl-Z (0x1A) to send
                ser.write((message + "\x1A").encode())
                time.sleep(2)

                response = ser.read(ser.in_waiting or 1).decode(errors="ignore")
                if "+CMGS" in response:
                    log.info(f"[SMS] Successfully sent: '{message}'")
                else:
                    log.warning(f"[SMS] Uncertain delivery. Response: {response.strip()}")

        except ImportError:
            log.info(f"[SMS STUB] pyserial not installed — would send: '{message}'")
        except Exception as e:
            log.warning(f"[SMS ERROR] Could not send SMS ({e}) — logged locally instead.")

    # -----------------------------------------------------------------
    #  Cleanup
    # -----------------------------------------------------------------

    def cleanup(self):
        """Release GPIO pins."""
        if self.gpio_ready:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                log.info("[HARDWARE] GPIO cleanup done.")
            except Exception:
                pass
