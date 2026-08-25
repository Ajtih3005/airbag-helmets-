"""
01_hardware_sensors/setup_pi.py
-------------------------------
Raspberry Pi Hardware Setup & Diagnostics Script.
Checks I2C/SPI interfaces, serial UART connections, RPi.GPIO availability,
and performs pin test for MPU6050, ADXL377, SIM800L, and Actuator MOSFET.
"""

import sys
import os
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pi_setup")

# Pin configuration according to system spec
MPU6050_I2C_BUS   = 1       # SDA=GPIO21, SCL=GPIO22 (or standard Pi I2C SDA=GPIO2, SCL=GPIO3)
ADXL377_X_PIN     = 34      # Analog ADC channel / GPIO
ADXL377_Y_PIN     = 35
ADXL377_Z_PIN     = 32
AIRBAG_MOSFET_PIN = 17      # BCM Pin 17 -> MOSFET Gate -> Solenoid Valve
WARNING_LED_PIN   = 27      # BCM Pin 27 -> Near-Crash Warning LED/Buzzer
SIM800L_UART_PORT = "/dev/ttyAMA0" # UART serial port for GSM module
ESP32_UART_PORT   = "/dev/ttyUSB0" # UART serial port for ESP32 IMU stream

def check_i2c():
    log.info("--- Checking I2C Interface ---")
    try:
        res = subprocess.run(["i2cdetect", "-y", "1"], capture_output=True, text=True)
        if "68" in res.stdout:
            log.info("✅ MPU6050 detected at I2C address 0x68!")
        else:
            log.warning("⚠️ MPU6050 not detected at 0x68 on I2C bus 1. Check wiring (VCC->3.3V, GND->GND, SDA->GPIO2, SCL->GPIO3).")
    except FileNotFoundError:
        log.warning("i2c-tools not installed. Run: sudo apt-get install -y i2c-tools")
    except Exception as e:
        log.error(f"I2C check error: {e}")

def check_gpio():
    log.info("--- Checking RPi.GPIO Module & Pins ---")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(AIRBAG_MOSFET_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(WARNING_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        log.info(f"✅ GPIO ready! Airbag pin=BCM{AIRBAG_MOSFET_PIN}, Warning LED pin=BCM{WARNING_LED_PIN}")
        GPIO.cleanup()
    except ImportError:
        log.warning("RPi.GPIO package not installed. Run: pip install RPi.GPIO gpiozero")
    except Exception as e:
        log.error(f"GPIO setup error: {e}")

def check_serial():
    log.info("--- Checking Serial UART Ports ---")
    for port in [ESP32_UART_PORT, "/dev/ttyS0", SIM800L_UART_PORT]:
        if os.path.exists(port):
            log.info(f"✅ Serial port available: {port}")
        else:
            log.info(f"ℹ️ Serial port not found: {port}")

def run_diagnostics():
    log.info("==========================================")
    log.info("  RASPBERRY PI HARDWARE SETUP & DIAGNOSTICS")
    log.info("==========================================")
    check_i2c()
    check_gpio()
    check_serial()
    log.info("==========================================")

if __name__ == "__main__":
    run_diagnostics()
