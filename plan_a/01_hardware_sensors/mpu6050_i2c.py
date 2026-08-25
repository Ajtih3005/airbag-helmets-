"""
01_hardware_sensors/mpu6050_i2c.py
----------------------------------
Direct I2C driver for MPU6050 6-axis IMU on Raspberry Pi.
Reads raw accelerometer (g) and gyroscope (deg/s) data directly over SMBus I2C.
"""

import time
import logging

log = logging.getLogger("mpu6050_i2c")

# MPU6050 I2C registers
MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43

class MPU6050I2C:
    def __init__(self, bus_id=1, address=MPU6050_ADDR):
        self.address = address
        self.bus = None
        try:
            import smbus2
            self.bus = smbus2.SMBus(bus_id)
            # Wake up MPU6050 from sleep mode
            self.bus.write_byte_data(self.address, PWR_MGMT_1, 0)
            log.info(f"[MPU6050] Connected on I2C bus {bus_id} at address 0x{address:02X}")
        except Exception as e:
            log.warning(f"[MPU6050 I2C ERROR] Could not initialize smbus2 ({e}).")

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        return val

    def read_motion6(self):
        """Returns dict of ax, ay, az (in m/s^2 or g) and gx, gy, gz (in deg/s)."""
        if not self.bus:
            return None

        try:
            # Scale factors for +/- 2g and +/- 250 deg/s default
            raw_ax = self._read_word_2c(ACCEL_XOUT_H)
            raw_ay = self._read_word_2c(ACCEL_XOUT_H + 2)
            raw_az = self._read_word_2c(ACCEL_XOUT_H + 4)

            raw_gx = self._read_word_2c(GYRO_XOUT_H)
            raw_gy = self._read_word_2c(GYRO_XOUT_H + 2)
            raw_gz = self._read_word_2c(GYRO_XOUT_H + 4)

            ax = (raw_ax / 16384.0) * 9.81
            ay = (raw_ay / 16384.0) * 9.81
            az = (raw_az / 16384.0) * 9.81

            gx = raw_gx / 131.0
            gy = raw_gy / 131.0
            gz = raw_gz / 131.0

            return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}
        except Exception as e:
            log.error(f"Error reading MPU6050: {e}")
            return None
