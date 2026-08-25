import logging

log = logging.getLogger("adxl377")

class ADXL377Reader:
    def __init__(self, spi_bus=0, spi_device=0):
        self.mcp = None
        self.chan_x = None
        self.chan_y = None
        self.chan_z = None

        try:
            # Modern Adafruit CircuitPython MCP3008 imports
            import busio
            import digitalio
            import board
            import adafruit_mcp3008.mcp3008 as MCP
            from adafruit_mcp3008.analog_in import AnalogIn

            # SPI setup (using default CE0 Pin 24 / GPIO 8 chip select pin on Raspberry Pi)
            # Standard Pi SPI Pins: SCLK (GPIO 11), MOSI (GPIO 10), MISO (GPIO 9)
            spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
            cs = digitalio.DigitalInOut(board.D8)  # CE0
            self.mcp = MCP.MCP3008(spi, cs)
            
            # MCP3008 channels connected to ADXL377 outputs
            self.chan_x = AnalogIn(self.mcp, MCP.P0)
            self.chan_y = AnalogIn(self.mcp, MCP.P1)
            self.chan_z = AnalogIn(self.mcp, MCP.P2)

            log.info("[ADXL377] Connected via modern CircuitPython MCP3008 ADC on SPI CE0")
        except Exception as e:
            log.warning(f"[ADXL377 STUB] Could not initialize SPI/ADC ({e}). ADXL377 running in pass-through mode.")

    def read_high_g(self, base_ax=0.0, base_ay=0.0, base_az=9.81):
        """
        Returns (hg_ax, hg_ay, hg_az) in g.
        If ADC is missing, mirrors MPU6050 low-g acceleration.
        """
        if self.mcp is None or self.chan_x is None:
            return base_ax, base_ay, base_az

        try:
            # Read voltage directly (already converted by library from 10-bit raw)
            volts_x = self.chan_x.voltage
            volts_y = self.chan_y.voltage
            volts_z = self.chan_z.voltage

            # Convert ADC voltage (0-3.3V) to g (+/- 200g scale)
            # ADXL377 zero-g voltage ~ 1.65V, sensitivity ~ 6.5 mV/g
            hg_ax = (volts_x - 1.65) / 0.0065
            hg_ay = (volts_y - 1.65) / 0.0065
            hg_az = (volts_z - 1.65) / 0.0065

            return hg_ax, hg_ay, hg_az
        except Exception as e:
            log.error(f"ADXL377 read error: {e}")
            return base_ax, base_ay, base_az

