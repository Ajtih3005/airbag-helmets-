"""
01_hardware_sensors/README_HARDWARE.md
---------------------------------------
Raspberry Pi Hardware Wiring & Setup Plan
"""

# Hardware Integration Plan (Raspberry Pi MCU)

## 1. Pin Wiring Diagram

### MPU6050 6-Axis IMU (I2C)
- **VCC** -> 3.3V (Pin 1)
- **GND** -> GND (Pin 6)
- **SDA** -> GPIO 2 / SDA (Pin 3)
- **SCL** -> GPIO 3 / SCL (Pin 5)

### ADXL377 High-g Accelerometer (+/- 200g via MCP3008 ADC SPI)
- **VCC** -> 3.3V
- **GND** -> GND
- **X-OUT** -> MCP3008 Channel 0
- **Y-OUT** -> MCP3008 Channel 1
- **Z-OUT** -> MCP3008 Channel 2

### Airbag Actuator (MOSFET Driver)
- **MOSFET Gate** -> BCM GPIO 17 (Pin 11)
- **MOSFET Drain** -> CO2 Solenoid Valve (-) / Servo Signal
- **MOSFET Source** -> Common GND

### SIM800L Emergency GSM Module (UART)
- **TX** -> Raspberry Pi RXD / GPIO 15 (Pin 10)
- **RX** -> Raspberry Pi TXD / GPIO 14 (Pin 8)
- **VCC** -> External 4.2V Power Supply (DO NOT power directly from Pi 3.3V)

---

## 2. Raspberry Pi OS Setup Commands

Run these commands on your Raspberry Pi terminal to enable hardware buses and install required drivers:

```bash
# 1. Enable I2C and SPI Interfaces
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial 0

# 2. Install I2C Tools & System Dependencies
sudo apt-get update
sudo apt-get install -y i2c-tools python3-pip python3-smbus

# 3. Install Python Hardware Libraries
pip3 install RPi.GPIO gpiozero smbus2 spidev pyserial

# 4. Verify MPU6050 on I2C Bus (Should show 0x68)
i2cdetect -y 1

# 5. Run Hardware Setup Diagnostic
python3 01_hardware_sensors/setup_pi.py
```
