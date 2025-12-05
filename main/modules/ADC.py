#!/usr/bin/env python

"""
Simplified ADC Interface optimized for excavator ADC sensors.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import the original ADCPi library
try:
    from ADCPi import ADCPi
except ImportError:
    raise ImportError("ADCPi library not found. Please install the original AB Electronics ADCPi library.")


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


@dataclass
class ADCConfig:
    """
    Configuration class for ADC settings with built-in excavator sensor configuration.

    This class contains all the configuration needed for the excavator pressure monitoring system.
    No external YAML file is required - all settings are defined here with explanations.
    """

    def __init__(self):
        """Initialize with default excavator configuration."""

        # =============================================================================
        # ADC HARDWARE CONFIGURATION
        # =============================================================================

        # I2C addresses for ADC boards
        # Each ADC Pi board uses 2 I2C addresses (for the dual MCP3424 chips)
        # Format: board_name: [address1, address2]
        self.i2c_addresses = {
            "b1": [0x6E, 0x6F],  # Board 1 - Main hydraulic sensors
            # "b2": [0x68, 0x69], # Board 2 - Additional sensors (if needed)
        }

        # PGA (Programmable Gain Amplifier) gain setting
        # Higher gain = more sensitive to small voltage changes
        # Valid values: 1, 2, 4, 8
        # Recommendation: Start with 1 for 0-5V sensors, increase if more resolution needed
        self.pga_gain = 1

        # ADC resolution and sampling rate
        # Higher bit rate = more precision but slower sampling

        # Note: When used in single-ended mode (as on the ADC Pi), the effective resolution
        # is actually 11, 13, 15, and 17 bits respectively, as one bit is used for sign
        # representation in the MCP3424's output.

        # Valid values and max sampling rates:
        # 12 = 12-1 bit (240 SPS max) - Fast sampling, lower precision
        # 14 = 14-1 bit (60 SPS max)  - Good balance
        # 16 = 16-1 bit (15 SPS max)  - High precision, slower
        # 18 = 18-1 bit (3.75 SPS max) - Highest precision, very slow
        # Note: more channels - less SPS! With 240sps and 8 channels the (realistic) read speed is about 25 SPS
        self.bit_rate = 12

        # Conversion mode
        # 0 = One-shot conversion (power saving, manual trigger)
        # 1 = Continuous conversion (real-time monitoring)
        # For excavator monitoring, continuous mode is recommended
        self.conversion_mode = 1

        # =============================================================================
        # EXCAVATOR ADC SENSOR CONFIGURATION
        # =============================================================================

        # sensor mappings
        # Channels 1-8 available on each board
        self.sensors = {
            "LiftBoom retract ps": {
                "input": ["b1", 1],
            },

            "LiftBoom extend ps": {
                "input": ["b1", 2],
            },

            "TiltBoom retract ps": {
                "input": ["b1", 3],
            },

            "TiltBoom extend ps": {
                "input": ["b1", 4],
            },

            "Scoop extend ps": {
                "input": ["b1", 5],
            },

            "Scoop retract ps": {
                "input": ["b1", 6],
            },

            "Pump ps": {
                "input": ["b1", 7],
            },

            "Slew encoder rot": {
                "input": ["b1", 8],
            },

        }

        # Validate configuration on initialization
        self.validate()

    def validate(self):
        """Validate all configuration parameters."""

        # Validate PGA gain
        valid_gains = [1, 2, 4, 8]
        if self.pga_gain not in valid_gains:
            raise ValueError(f"Invalid PGA gain {self.pga_gain}. Must be one of {valid_gains}")

        # Validate bit rate
        valid_rates = [12, 14, 16, 18]
        if self.bit_rate not in valid_rates:
            raise ValueError(f"Invalid bit rate {self.bit_rate}. Must be one of {valid_rates}")

        # Validate conversion mode
        if self.conversion_mode not in [0, 1]:
            raise ValueError(f"Invalid conversion mode {self.conversion_mode}. Must be 0 or 1")

        # Validate I2C addresses
        for board_name, addresses in self.i2c_addresses.items():
            if len(addresses) != 2:
                raise ValueError(f"Board {board_name} must have exactly 2 I2C addresses")
            for addr in addresses:
                if not (0x08 <= addr <= 0x77):
                    raise ValueError(f"Invalid I2C address {hex(addr)} for board {board_name}")

        # Validate sensor configurations
        for sensor_name, sensor_config in self.sensors.items():
            if 'input' not in sensor_config:
                raise ValueError(f"Sensor '{sensor_name}' missing 'input' configuration")

            input_config = sensor_config['input']
            if len(input_config) != 2:
                raise ValueError(f"Sensor '{sensor_name}' input must be [board, channel]")

            board_name, channel = input_config
            if board_name not in self.i2c_addresses:
                raise ValueError(f"Sensor '{sensor_name}' references unknown board '{board_name}'")

            if not isinstance(channel, int) or not (1 <= channel <= 8):
                raise ValueError(f"Sensor '{sensor_name}' channel must be integer 1-8")

    def get_needed_boards(self) -> List[str]:
        """Get list of boards needed by configured sensors."""
        needed_boards = set()
        for sensor_config in self.sensors.values():
            board_name = sensor_config['input'][0]
            needed_boards.add(board_name)
        return list(needed_boards)

    def get_sensor_by_input(self, board_name: str, channel: int) -> str:
        """Get sensor name by board and channel."""
        for sensor_name, sensor_config in self.sensors.items():
            if sensor_config['input'] == [board_name, channel]:
                return sensor_name
        return None

    def get_channel_mapping(self, board_name: str) -> Dict[int, str]:
        """
        Get channel-to-sensor-name mapping for a specific board.

        Args:
            board_name: Name of the ADC board (e.g., "b1")

        Returns:
            Dictionary mapping channel numbers (1-8) to sensor names
        """
        mapping = {}
        for sensor_name, sensor_config in self.sensors.items():
            if sensor_config['input'][0] == board_name:
                channel = sensor_config['input'][1]
                mapping[channel] = sensor_name
        return mapping

    def get_board_addresses(self, board_name: str) -> Tuple[int, int]:
        """Get I2C addresses for a specific board."""
        if board_name not in self.i2c_addresses:
            raise ValueError(f"Unknown board '{board_name}'")
        addresses = self.i2c_addresses[board_name]
        return addresses[0], addresses[1]

    def get_sampling_info(self) -> Dict:
        """Get information about current sampling configuration."""
        rate_info = {
            12: {"max_sps": 240},
            14: {"max_sps": 60},
            16: {"max_sps": 15},
            18: {"max_sps": 3.75}
        }

        return {
            "bit_rate": self.bit_rate,
            "max_samples_per_second": rate_info[self.bit_rate]["max_sps"],
            "pga_gain": self.pga_gain,
            "conversion_mode": "Continuous" if self.conversion_mode == 1 else "One-shot"
        }


class SimpleADC:
    def __init__(self, custom_config: ADCConfig = None, filter_alpha: float = 1.0, log_level: str = "INFO"):
        """
        Initialize ADC with built-in or custom configuration.

        :param custom_config: Optional custom ADCConfig instance. If None, uses default excavator config.
        :param filter_alpha: EMA filter alpha (0-1). Higher = more responsive, lower = smoother. Default 0.8.
        :param log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
        """
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.SimpleADC")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config = custom_config if custom_config else ADCConfig()
        self.adcs = {}
        self.initialized = False
        self.filter_alpha = filter_alpha
        self.filtered_values = {}  # Store filtered values per board/channel: {(board, channel): value}
        self.initialize_adc()

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _board_needed(self, board_name: str) -> bool:
        """Check if a board is needed by any configured sensor."""
        return board_name in self.config.get_needed_boards()

    def initialize_adc(self):
        """Initialize ADC boards based on sensor configurations."""
        needed_boards = self.config.get_needed_boards()
        if not needed_boards:
            raise RuntimeError("No sensors configured - nothing to initialize!")

        # Only initialize boards that are actually needed
        for board_name in needed_boards:
            try:
                addr1, addr2 = self.config.get_board_addresses(board_name)

                # Create ADCPi instance with the original library
                adc = ADCPi(address=addr1, address2=addr2, bus=3) # use virtual bus 3

                # Configure the ADC with our settings
                adc.set_pga(self.config.pga_gain)
                adc.set_conversion_mode(self.config.conversion_mode)
                adc.set_bit_rate(self.config.bit_rate)

                # Store the configured ADC
                self.adcs[board_name] = adc

            except Exception as e:
                raise Exception(f"Failed to initialize {board_name}: {e}")

        if not self.adcs:
            raise RuntimeError("No ADC boards were initialized!")

        self.initialized = True

    def read_sensors(self) -> Dict[str, float]:
        """Read filtered voltage from all sensors.

        Raises:
            RuntimeError: If ADC not initialized
            Exception: If any sensor read fails (no fallbacks - fail fast for debugging)
        """
        if not self.initialized:
            raise RuntimeError("ADC not initialized!")

        readings = {}

        for sensor_name, sensor_config in self.config.sensors.items():
            board_name = sensor_config['input'][0]
            channel = sensor_config['input'][1]

            # Use read_channel which applies EMA filtering
            # Don't catch exceptions - let them propagate for better debugging
            readings[sensor_name] = self.read_channel(board_name, channel)

        return readings

    def read_channel(self,
                     board_name: str, channel: int) -> float:
        """
        Read filtered voltage from a specific board and channel with EMA filtering.

        :param board_name: Name of the ADC board
        :param channel: Channel number (1-8)
        :return: Filtered voltage reading
        """
        if not self.initialized:
            raise RuntimeError("ADC not initialized!")

        adc = self.adcs.get(board_name)
        if not adc:
            raise ValueError(f"ADC board {board_name} not found or not initialized")

        if not 1 <= channel <= 8:
            raise ValueError("Channel must be between 1 and 8")

        try:
            # Read raw voltage
            raw_voltage = adc.read_voltage(channel)

            # Apply EMA filter
            key = (board_name, channel)
            if key not in self.filtered_values:
                # First reading - initialize filter
                self.filtered_values[key] = raw_voltage
            else:
                # EMA: filtered = alpha * new + (1 - alpha) * old
                self.filtered_values[key] = (self.filter_alpha * raw_voltage +
                                            (1 - self.filter_alpha) * self.filtered_values[key])

            return round(self.filtered_values[key], 2)
        except Exception as e:
            raise Exception(f"Error reading {board_name} channel {channel}: {e}")

    def get_board_info(self) -> Dict[str, Dict]:
        """Get information about initialized ADC boards."""
        if not self.initialized:
            return {}

        board_info = {}
        for board_name, adc in self.adcs.items():
            addr1, addr2 = self.config.get_board_addresses(board_name)
            board_info[board_name] = {
                'address1': hex(addr1),
                'address2': hex(addr2),
                'pga_gain': self.config.pga_gain,
                'bit_rate': self.config.bit_rate,
                'conversion_mode': 'continuous' if self.config.conversion_mode == 1 else 'one-shot'
            }
        return board_info

    def get_config_summary(self) -> Dict:
        """Get a summary of the current configuration."""
        sampling_info = self.config.get_sampling_info()
        return {
            'pga_gain': self.config.pga_gain,
            'bit_rate': self.config.bit_rate,
            'max_sampling_rate': sampling_info['max_samples_per_second'],
            'conversion_mode': sampling_info['conversion_mode'],
            'filter_alpha': self.filter_alpha,
            'total_sensors': len(self.config.sensors),
            'total_boards': len(self.config.i2c_addresses),
            'initialized_boards': len(self.adcs),
            'needed_boards': self.config.get_needed_boards()
        }

    def list_sensors(self):
        """List all configured pressure sensors"""
        print("\nConfigured Excavator Pressure Sensors:")
        print("=" * 70)
        for sensor_name, config in self.config.sensors.items():
            board, channel = config['input']
            status = "✓ Ready" if board in self.adcs else "✗ Board not initialized"
            print(f"  {sensor_name:18} - {board}:{channel}  [{status}]")

    def test_all_channels(self):
        """Test all channels on all initialized boards."""
        if not self.initialized:
            raise RuntimeError("ADC not initialized!")

        print("\nTesting all ADC channels:")
        print("=" * 60)

        for board_name, adc in self.adcs.items():
            print(f"\n{board_name.upper()}:")
            for channel in range(1, 9):
                try:
                    voltage = adc.read_voltage(channel)
                    sensor_name = self.config.get_sensor_by_input(board_name, channel)
                    if sensor_name:
                        sensor_info = f" -> {sensor_name}"
                    else:
                        sensor_info = " (unused)"
                    print(f"  Channel {channel}{sensor_info}: {voltage:.3f}V")
                except Exception as e:
                    print(f"  Channel {channel}: ERROR - {e}")

    def print_config(self):
        """Print detailed configuration information."""
        print("\nExcavator ADC Configuration:")
        print("=" * 60)

        # Sampling configuration
        sampling_info = self.config.get_sampling_info()
        print(f"Sampling Configuration:")
        print(f"  PGA Gain: {sampling_info['pga_gain']}x")
        print(f"  Bit Rate: {sampling_info['bit_rate']} bit")
        print(f"  Max Sampling Rate: {sampling_info['max_samples_per_second']} SPS")
        print(f"  Conversion Mode: {sampling_info['conversion_mode']}")
        print(f"  EMA Filter Alpha: {self.filter_alpha} (higher = more responsive)")

        # Board configuration
        print(f"\nBoard Configuration:")
        for board_name, addresses in self.config.i2c_addresses.items():
            status = "✓ Active" if board_name in self.adcs else "○ Standby"
            print(f"  {status} {board_name}: I2C addresses {hex(addresses[0])}, {hex(addresses[1])}")

        # Sensor summary
        summary = self.get_config_summary()
        print(f"\nSensor Summary:")
        print(f"  Total Sensors Configured: {summary['total_sensors']}")
        print(f"  Boards Needed: {', '.join(summary['needed_boards'])}")
        print(f"  Boards Initialized: {summary['initialized_boards']}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        # The original ADCPi library handles cleanup automatically
        pass

# example
if __name__ == "__main__":
    try:
        with SimpleADC() as adc:
            adc.print_config()
            adc.list_sensors()

            print("\nReading all sensors:")
            readings = adc.read_sensors()
            for sensor, voltage in readings.items():
                print(f"  {sensor}: {voltage:.2f}V")
            time.sleep(5)
            while True:
                readings = adc.read_sensors()
                print(readings)
                time.sleep(1)


    except Exception as e:
        print(f"Error: {e}")