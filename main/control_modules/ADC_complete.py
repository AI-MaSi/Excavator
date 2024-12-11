#!/usr/bin/env python

"""
Simplified ADC Interface optimized for excavator pressure sensors.
Provides basic setup and raw reading capabilities for AB Electronics UK ADC Pi 8-Channel ADC.

Features:
- YAML configuration support for excavator sensors
- Basic ADC setup with specific bit rate and PGA settings
- Raw voltage readings for pressure sensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import yaml
import platform
import time
import re
from typing import Dict, Optional

try:
    from smbus2 import SMBus
except ImportError:
    try:
        from smbus import SMBus
    except ImportError:
        raise ImportError("python-smbus or smbus2 not found")


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class TimeoutError(Error):
    """The operation exceeded the given deadline."""
    pass


class SimplifiedADC:
    def __init__(self, config_file: str):
        """
        Initialize ADC with configuration from YAML file.

        :param config_file: Path to YAML configuration file
        """
        self._load_config(config_file)
        self.adcs = {}
        self.initialized = False
        self.initialize_adc()

    def _load_config(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as file:
                configs = yaml.safe_load(file)
                self.pressure_sensors = configs.get('PRESSURE_SENSORS', {})
                adc_config = configs['ADC_CONFIG']
                self.i2c_addresses = adc_config['i2c_addresses']
                self.pga_gain = adc_config['pga_gain']
                self.bit_rate = adc_config['bit_rate']
                self.conversion_mode = adc_config['conversion_mode']
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _get_smbus(self, bus=None):
        """Get SMBus instance for the target device."""
        i2c_bus = 1
        if bus is not None:
            i2c_bus = bus
        else:
            device = platform.uname()[1]

            # Device-specific bus mapping
            device_bus_map = {
                "orangepione": 0,
                "orangepizero2": 3,
                "orangepiplus": 0,
                "orangepipcplus": 0,
                "linaro-alip": 1,
                "bpi-m2z": 0,
                "bpi-iot-ros-ai": 0
            }

            if device in device_bus_map:
                i2c_bus = device_bus_map[device]
            elif device == "raspberrypi":
                for line in open('/proc/cpuinfo').readlines():
                    model = re.match('(.*?)\\s*:\\s*(.*)', line)
                    if model:
                        name, value = model.group(1), model.group(2)
                        if name == "Revision":
                            i2c_bus = 0 if value[-4:] in ('0002', '0003') else 1
                            break

        try:
            return SMBus(i2c_bus)
        except FileNotFoundError:
            raise FileNotFoundError("I2C bus not found. Check selected bus number.")
        except IOError as err:
            raise IOError(f"I2C communication error: {err}")

    def initialize_adc(self):
        """Initialize ADC boards based on sensor configurations."""
        board_needs_init = {board: False for board in self.i2c_addresses.keys()}

        # Check which boards need initialization
        for sensor_config in self.pressure_sensors.values():
            board_name = sensor_config['input'][0]
            if board_name in board_needs_init:
                board_needs_init[board_name] = True

        # Initialize required boards
        for board_name, needs_init in board_needs_init.items():
            if needs_init:
                addr1, addr2 = self.i2c_addresses[board_name]
                try:
                    adc = ADCBoard(addr1, addr2, self.bit_rate)
                    adc.set_pga(self.pga_gain)
                    adc.set_conversion_mode(self.conversion_mode)
                    self.adcs[board_name] = adc
                    print(f"Initialized {board_name} with addresses {hex(addr1)}, {hex(addr2)}")
                except Exception as e:
                    raise Exception(f"Failed to initialize {board_name}: {e}")

        self.initialized = True

    def read_raw(self) -> Dict[str, float]:
        """Read raw voltage from all pressure sensors."""
        if not self.initialized:
            raise RuntimeError("ADC not initialized!")

        readings = {}
        for sensor_name, sensor_config in self.pressure_sensors.items():
            board_name = sensor_config['input'][0]
            channel = sensor_config['input'][1]

            adc = self.adcs.get(board_name)
            if adc:
                try:
                    voltage = adc.read_voltage(channel)
                    readings[sensor_name] = round(voltage, 2)
                except Exception as e:
                    print(f"Error reading {sensor_name}: {e}")
            else:
                print(f"ADC board {board_name} not found for sensor {sensor_name}")

        return readings

    def list_sensors(self):
        """List all configured pressure sensors."""
        print("\nConfigured Pressure Sensors:")
        for sensor_name, config in self.pressure_sensors.items():
            board, channel = config['input']
            print(f"  {sensor_name:20} - Board: {board}, Channel: {channel}")


class ADCBoard:
    """
    Handles low-level ADC operations for a single board
    """

    def __init__(self, address1, address2, bit_rate):
        """Initialize ADC board with proper sequence."""
        self.__adc1_address = address1
        self.__adc2_address = address2
        self.__bus = self._get_smbus()

        # Initialize configuration registers first
        self.__adc1_conf = 0x9C
        self.__adc2_conf = 0x9C
        self.__conversion_mode = 1
        self.__pga = float(0.5)

        # Set initial bit rate (this will also set self.__lsb)
        self.set_bit_rate(bit_rate)

        # Write initial configuration to both ADCs
        self.__bus.write_byte(self.__adc1_address, self.__adc1_conf)
        self.__bus.write_byte(self.__adc2_address, self.__adc2_conf)

    def _get_smbus(self, bus=None):
        """Get SMBus instance."""
        return SMBus(1 if bus is None else bus)

    def set_bit_rate(self, rate):
        """Set bit rate and update LSB value."""
        bit_rate_settings = {
            12: (0x00, 0.0005),
            14: (0x04, 0.000125),
            16: (0x08, 0.00003125),
            18: (0x0C, 0.0000078125)
        }

        if rate not in bit_rate_settings:
            raise ValueError('Bit rate must be 12, 14, 16, or 18')

        conf_bits, self.__lsb = bit_rate_settings[rate]
        self.__bitrate = rate
        self.__adc1_conf = (self.__adc1_conf & 0xF3) | conf_bits
        self.__adc2_conf = (self.__adc2_conf & 0xF3) | conf_bits

        if hasattr(self, '_ADCBoard__bus'):  # Only write if bus is initialized
            self.__bus.write_byte(self.__adc1_address, self.__adc1_conf)
            self.__bus.write_byte(self.__adc2_address, self.__adc2_conf)

    def set_pga(self, gain):
        """Set PGA gain."""
        gain_settings = {
            1: (0x00, 0.5),
            2: (0x01, 1.0),
            4: (0x02, 2.0),
            8: (0x03, 4.0)
        }

        if gain not in gain_settings:
            raise ValueError('PGA gain must be 1, 2, 4, or 8')

        conf_bits, self.__pga = gain_settings[gain]
        self.__adc1_conf = (self.__adc1_conf & 0xFC) | conf_bits
        self.__adc2_conf = (self.__adc2_conf & 0xFC) | conf_bits

        self.__bus.write_byte(self.__adc1_address, self.__adc1_conf)
        self.__bus.write_byte(self.__adc2_address, self.__adc2_conf)

    def set_conversion_mode(self, mode):
        """Set conversion mode (0: one-shot, 1: continuous)."""
        if mode not in (0, 1):
            raise ValueError('Conversion mode must be 0 or 1')

        self.__conversion_mode = mode
        conf_bit = mode << 4
        self.__adc1_conf = (self.__adc1_conf & 0xEF) | conf_bit
        self.__adc2_conf = (self.__adc2_conf & 0xEF) | conf_bit

        self.__bus.write_byte(self.__adc1_address, self.__adc1_conf)
        self.__bus.write_byte(self.__adc2_address, self.__adc2_conf)

    def read_voltage(self, channel):
        """Read voltage from specified channel."""
        if not 1 <= channel <= 8:
            raise ValueError('Channel must be between 1 and 8')

        raw = self._read_raw(channel)
        if raw is not None:
            return float(raw * (self.__lsb / self.__pga) * 2.471)
        return 0.0

    def _read_raw(self, channel):
        """Read raw value from ADC."""
        address = self.__adc1_address if channel <= 4 else self.__adc2_address
        conf = self.__adc1_conf if channel <= 4 else self.__adc2_conf

        # Set channel bits
        channel_bits = ((channel - 1) % 4) << 5
        conf = (conf & 0x9F) | channel_bits

        if self.__conversion_mode == 0:
            conf |= (1 << 7)
            self.__bus.write_byte(address, conf)
            conf &= ~(1 << 7)

        # Wait for conversion
        timeout = time.monotonic() + 1.0
        while True:
            data = self.__bus.read_i2c_block_data(address, conf, 4)
            if not (data[3] & (1 << 7)):
                break
            if time.monotonic() > timeout:
                raise TimeoutError(f"Channel {channel} conversion timed out")
            time.sleep(0.001)

        # Process reading based on bit rate
        if self.__bitrate == 18:
            raw = ((data[0] & 0x03) << 16) | (data[1] << 8) | data[2]
            raw &= ~(1 << 17)  # Clear sign bit
        else:
            raw = (data[0] << 8) | data[1]
            raw &= ~(1 << 15)  # Clear sign bit for 16-bit and lower

        return raw