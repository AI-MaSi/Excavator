"""
GPIO_sensors.py

This module implements interfaces for sensors connected to GPIO (General Purpose Input/Output) pins,
specifically designed for Raspberry Pi or similar single-board computers. It provides classes for
RPM (Revolutions Per Minute) sensors and center position sensors.

Key features:
1. Configurable GPIO sensors using a YAML configuration file
2. Support for RPM sensors with configurable number of magnets
3. Real-time RPM calculation using threaded measurements
4. Center position sensing for positional feedback
5. Clean GPIO setup and cleanup procedures

The module includes two main classes:
1. RPMSensor:
   - Initializes GPIO for RPM sensing
   - Calculates RPM in real-time using a separate thread
   - Provides methods to read current RPM

2. CenterPositionSensor:
   - Initializes GPIO for position sensing
   - Provides method to check if sensor is in center position

Usage:
1. Create a YAML configuration file defining your GPIO sensor setup (for RPM sensors)
2. Initialize the appropriate sensor class with the configuration
3. Use the provided methods to read sensor data
4. Ensure to call the cleanup method when done to release GPIO resources
"""

from time import time, sleep
import threading
import yaml
from typing import Dict

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO module not found. Make sure you're running on a Raspberry Pi.")
    # TODO: simlation mode, prob not needed here...


class RPMSensor:
    def __init__(self, config_file: str, sensor_name: str, decimals: int = 2):
        """
        Initialize RPM sensor with configuration from a YAML file.

        :param config_file: Path to the YAML configuration file.
        :param sensor_name: Name of the sensor in the configuration.
        :param decimals: Number of decimal places for RPM value.
        """
        self.sensor_configs = self._load_config(config_file, sensor_name)
        self.sensor_pin = self.sensor_configs['GPIO pin']
        self.magnets = self.sensor_configs['magnets']
        self.pulse_count = 0
        self.rpm = 0
        self.decimals = decimals
        self._setup_gpio()
        self._start_measurement()

    def _load_config(self, config_file: str, sensor_name: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as file:
                configs = yaml.safe_load(file)
                return configs['RPM_SENSORS'][sensor_name]
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _setup_gpio(self):
        """Set up GPIO for the RPM sensor."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.sensor_pin, GPIO.FALLING, callback=self._sensor_callback)

    def _sensor_callback(self, channel):
        """Callback function for GPIO event detection."""
        self.pulse_count += 1

    def _calculate_rpm(self):
        """Calculate RPM based on pulse count."""
        last_checked_time = time()
        while True:
            current_time = time()
            elapsed_time = current_time - last_checked_time

            if elapsed_time >= 0.1:  # Update every 100ms
                self.rpm = (self.pulse_count / self.magnets) * 60 / elapsed_time
                self.pulse_count = 0
                last_checked_time = current_time

            sleep(0.001)

    def _start_measurement(self):
        """Start the RPM measurement thread."""
        self.thread = threading.Thread(target=self._calculate_rpm)
        self.thread.daemon = True
        self.thread.start()

    def read_rpm(self) -> float:
        """Read the current RPM value."""
        return round(self.rpm, self.decimals)

    def cleanup(self):
        """Clean up GPIO resources."""
        GPIO.cleanup(self.sensor_pin)


class GPIOSensor:
    """
    Base class for GPIO sensors.
    Returns the current state of the GPIO pin as a boolean value.
    """
    def __init__(self, sensor_pin: int, pull_up_down: int = GPIO.PUD_UP):
        """
        Initialize a universal GPIO Sensor.

        :param sensor_pin: GPIO pin number for the sensor.
        :param pull_up_down: Pull-up/down resistor setting (GPIO.PUD_UP or GPIO.PUD_DOWN).
        """
        self.sensor_pin = sensor_pin
        self.pull_up_down = pull_up_down
        self._setup_gpio()

    def _setup_gpio(self):
        """Set up GPIO for the sensor."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=self.pull_up_down)

    def read_value(self) -> bool:
        """Read the current state of the GPIO pin."""
        return GPIO.input(self.sensor_pin)

    def cleanup(self):
        """Clean up GPIO resources."""
        GPIO.cleanup(self.sensor_pin)