# new PWM controller without ServoKit. Using lower-level adafruit_pca9685 library instead.

import threading
import yaml
import time
from collections import deque

try:
    from adafruit_pca9685 import PCA9685
    import board
    import busio

    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False
    print("PWM module not found. Running in simulation mode.\n")
    time.sleep(3)


class PWMcontroller:
    NUM_CHANNELS = 16  # PCA9685 hardware constant

    def __init__(self, config_file: str, pump_variable: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0) -> None:
        self.pump_variable = pump_variable
        self.toggle_channels = toggle_channels
        self.time_window = 10

        # Load and validate config
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.channel_configs = configs['CHANNEL_CONFIGS']

        self._validate_configuration(self.channel_configs)

        self.values = [0.0 for _ in range(self.NUM_CHANNELS)]

        self.num_inputs = None  # calculated and updated inside _build_channel_data()
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking

        self.input_event = threading.Event()
        self.monitor_thread = None
        self.running = False

        self.input_count = 0
        self.last_input_time = time.time()
        max_inputs = int(self.time_window * (self.input_rate_threshold or 20))
        self.input_timestamps = deque(maxlen=max_inputs)

        self.pump_enabled = True
        self.pump_variable_sum = 0.0
        self.manual_pump_load = 0.0

        # Build optimized data structures
        self._channel_data, self._pump_data = self._build_channel_data()

        if SERVOKIT_AVAILABLE:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50  # Set to 50Hz for standard servos
        else:
            self.pca = PCA9685Stub()

        self.reset()

        if not self.skip_rate_checking:
            self._start_monitoring()
        return

    def _build_channel_data(self):
        """
        Build optimized channel data structures from configuration.
        Updates self.num_inputs and returns tuple of (channel_data, pump_data)
        """
        channel_data = {}
        input_channels = set()
        pump_data = None

        for channel_name, config in self.channel_configs.items():
            input_channel = config.get('input_channel')

            # Fix: Normalize None values
            if input_channel in [None, "None", "none", "null"]:
                input_channel = None

            if isinstance(input_channel, int):
                input_channels.add(input_channel)

            # Handle pump config separately
            if channel_name == 'pump':
                pump_data = {
                    'channel': config['output_channel'],
                    'multiplier': config['multiplier'],
                    'idle': config['idle'],
                    'input_channel': input_channel,
                    'pulse_min': config['pulse_min'],
                    'pulse_max': config['pulse_max']
                }
                continue

            channel_data[channel_name] = {
                'input_channel': input_channel,
                'output_channel': config['output_channel'],
                'deadzone_threshold': config['deadzone'] / 100.0 * 2,
                'affects_pump': config.get('affects_pump', False),
                'toggleable': config.get('toggleable', False),
                'gamma_pos': config['gamma_positive'],
                'gamma_neg': config['gamma_negative'],
                'pulse_max': config['pulse_max'],
                'pulse_min': config['pulse_min'],
                'direction': config['direction'],
                'center': config.get('center')
            }

            channel_data[channel_name]['pulse_range'] = (
                    channel_data[channel_name]['pulse_max'] -
                    channel_data[channel_name]['pulse_min']
            )

            # Fix: Properly handle center: None by checking for both None and string "None"
            if channel_data[channel_name]['center'] is None or channel_data[channel_name]['center'] == "None" or \
                    channel_data[channel_name]['center'] == "none" or channel_data[channel_name]['center'] == "null":
                channel_data[channel_name]['center'] = (
                        channel_data[channel_name]['pulse_min'] +
                        (channel_data[channel_name]['pulse_range'] / 2)
                )

        self.num_inputs = len(input_channels)
        return channel_data, pump_data

    @staticmethod
    def _validate_configuration(config):
        # Define validation constants once
        VALIDATION_LIMITS = {
            'channels': 16,
            'gamma': {'min': 0.1, 'max': 3.0},
            'pulse': {'min': 0, 'max': 4095},
            'pump': {
                'idle': {'min': -1.0, 'max': 0.6},
                'multiplier': {'max': 1.0}
            }
        }

        # required keys
        REQUIRED_CHANNEL_KEYS = {
            'input_channel',
            'output_channel',
            'pulse_min',
            'pulse_max',
            'direction'
        }
        REQUIRED_PUMP_KEYS = {'idle', 'multiplier'}

        # Track used channels for duplicate detection
        used_channels = {
            'input': {},
            'output': {}
        }

        # Collection of all validation errors
        errors = []

        for channel_name, config_data in config.items():
            # Check required keys exist
            missing_keys = REQUIRED_CHANNEL_KEYS - set(config_data.keys())
            if missing_keys:
                errors.append(f"Missing required keys {missing_keys} in configuration for channel '{channel_name}'")

            # Special handling for pump channel
            if channel_name == 'pump':
                missing_pump_keys = REQUIRED_PUMP_KEYS - set(config_data.keys())
                if missing_pump_keys:
                    errors.append(f"Missing required pump keys {missing_pump_keys}")

                # Validate pump specific values
                if 'idle' in config_data and not (VALIDATION_LIMITS['pump']['idle']['min'] <= config_data['idle'] <=
                                                  VALIDATION_LIMITS['pump']['idle']['max']):
                    errors.append(
                        f"Pump idle must be between {VALIDATION_LIMITS['pump']['idle']['min']} and {VALIDATION_LIMITS['pump']['idle']['max']}")

                if 'multiplier' in config_data and not (
                        0 < config_data['multiplier'] <= VALIDATION_LIMITS['pump']['multiplier']['max']):
                    errors.append(
                        f"Pump multiplier must be between 0 and {VALIDATION_LIMITS['pump']['multiplier']['max']}")
                continue

            # Validate direction (now required)
            if 'direction' in config_data and config_data['direction'] not in [-1, 1]:
                errors.append(f"direction must be either -1 or 1 for channel '{channel_name}'")

            # Validate input channel
            if 'input_channel' in config_data:
                input_channel = config_data['input_channel']
                # Fix: Check against all variations of None properly
                is_none_value = (input_channel is None or
                                 input_channel == "None" or
                                 input_channel == "none" or
                                 input_channel == "null")

                if not is_none_value and not isinstance(input_channel, int):
                    errors.append(
                        f"Input channel must be an integer or None for channel '{channel_name}', got {type(input_channel).__name__}")
                elif not is_none_value:  # Now we know it's an int
                    if input_channel in used_channels['input']:
                        errors.append(
                            f"Input channel {input_channel} is used by both '{channel_name}' and '{used_channels['input'][input_channel]}'")
                    if not (0 <= input_channel < VALIDATION_LIMITS['channels']):
                        errors.append(f"Input channel must be between 0 and {VALIDATION_LIMITS['channels'] - 1}")
                    used_channels['input'][input_channel] = channel_name

            # Validate output channel
            if 'output_channel' in config_data:
                output_channel = config_data['output_channel']
                if not isinstance(output_channel, int):
                    errors.append(
                        f"Output channel must be an integer for channel '{channel_name}', got {type(output_channel).__name__}")
                else:
                    if output_channel in used_channels['output']:
                        errors.append(
                            f"Output channel {output_channel} is used by both '{channel_name}' and '{used_channels['output'][output_channel]}'")
                    if not (0 <= output_channel < VALIDATION_LIMITS['channels']):
                        errors.append(f"Output channel must be between 0 and {VALIDATION_LIMITS['channels'] - 1}")
                    used_channels['output'][output_channel] = channel_name

            # Validate pulse ranges
            if 'pulse_min' in config_data and 'pulse_max' in config_data:
                if config_data['pulse_min'] >= config_data['pulse_max']:
                    errors.append(f"pulse_min must be less than pulse_max for channel '{channel_name}'")

                for pulse_key in ['pulse_min', 'pulse_max']:
                    if not (VALIDATION_LIMITS['pulse']['min'] <= config_data[pulse_key] <= VALIDATION_LIMITS['pulse'][
                        'max']):
                        errors.append(
                            f"{pulse_key} must be between {VALIDATION_LIMITS['pulse']['min']} and {VALIDATION_LIMITS['pulse']['max']}")

            # Validate center if provided and not None/none/null
            if 'center' in config_data and config_data['center'] is not None and config_data['center'] != "None" and \
                    config_data['center'] != "none" and config_data['center'] != "null":
                # First ensure center is numeric
                if not isinstance(config_data['center'], (int, float)):
                    errors.append(
                        f"center must be a number or None for channel '{channel_name}', got {type(config_data['center']).__name__}")
                # Then validate the value
                elif not (config_data['pulse_min'] <= config_data['center'] <= config_data['pulse_max']):
                    errors.append(f"center must be between pulse_min and pulse_max for channel '{channel_name}'")

            # Validate gamma values if provided
            for gamma_key in ['gamma_positive', 'gamma_negative']:
                if gamma_key in config_data:
                    gamma_value = config_data[gamma_key]
                    if not (VALIDATION_LIMITS['gamma']['min'] <= gamma_value <= VALIDATION_LIMITS['gamma']['max']):
                        errors.append(
                            f"{gamma_key} must be between {VALIDATION_LIMITS['gamma']['min']} and {VALIDATION_LIMITS['gamma']['max']}")

            # Validate boolean fields
            for bool_field in ['affects_pump', 'toggleable']:
                if bool_field in config_data and not isinstance(config_data[bool_field], bool):
                    errors.append(f"{bool_field} must be a boolean for channel '{channel_name}'")

        # If any errors were found, raise them all together
        if errors:
            raise ValueError("\n".join(errors))

        return True

    def _start_monitoring(self) -> None:
        """Start the input rate monitoring thread."""
        if self.skip_rate_checking:
            print("Input rate checking is disabled.")
            return

        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_input_rate, daemon=True)
            self.monitor_thread.start()
        else:
            print("Monitoring is already running.")
        return

    def _stop_monitoring(self) -> None:
        """Stop the input rate monitoring thread."""
        self.running = False
        if self.monitor_thread is not None:
            self.input_event.set()  # Wake up the thread if it's waiting
            self.monitor_thread.join()
            self.monitor_thread = None
            print("Stopped input rate monitoring...")
        return

    def monitor_input_rate(self, limit_percentage=0.25) -> None:
        """
        Monitor the input rate and ensure it stays above the threshold.

        Args:
            limit_percentage: Percentage of threshold rate required for consecutive inputs
        """
        print("Monitoring input rate...")
        while self.running:
            if self.input_event.wait(timeout=1.0 / self.input_rate_threshold):
                self.input_event.clear()
                current_time = time.time()
                time_diff = current_time - self.last_input_time
                self.last_input_time = current_time

                if time_diff > 0:
                    current_rate = 1 / time_diff
                    if current_rate >= self.input_rate_threshold:
                        self.input_count += 1
                        # Require consecutive good inputs. 25% of threshold rate, rounded down
                        if self.input_count >= int(self.input_rate_threshold * limit_percentage):
                            self.is_safe_state = True
                            self.input_count = 0
                    else:
                        self.input_count = 0

                    # Add timestamp to deque - old timestamps are automatically removed
                    self.input_timestamps.append(current_time)

            else:
                if self.is_safe_state:
                    print("Input rate too low. Entering safe state...")
                    self.reset(reset_pump=False)
                    self.is_safe_state = False
                    self.input_count = 0

    def update_values(self, raw_value_list):
        """
        Update channel values based on input and handle both channel and pump outputs.

        Args:
            raw_value_list: List of raw input values to process
        """
        # Safety checks
        if not self.skip_rate_checking:
            self.input_event.set()

            if not self.is_safe_state:
                print(f"System in safe state. Ignoring input. Average rate: {self.get_average_input_rate():.2f}Hz")
                return

        if raw_value_list is None:
            print("Warning: Input values are None, resetting to defaults")
            self.reset()
            return

        # Input validation and normalization
        if isinstance(raw_value_list, (float, int)):
            raw_value_list = [raw_value_list]

        if len(raw_value_list) != self.num_inputs:
            print(
                f"Warning: Expected {self.num_inputs} inputs, but received {len(raw_value_list)}. Ignoring this update.")
            return

        # Reset pump variable sum before processing channels
        self.pump_variable_sum = 0.0

        # Process all channels using pre-computed data
        for channel_name, data in self._channel_data.items():
            input_channel = data['input_channel']

            # Skip if channel has no valid input
            if input_channel is None or not isinstance(input_channel, int) or input_channel >= len(raw_value_list):
                continue

            # Get and clamp input value
            input_value = max(-1, min(raw_value_list[input_channel], 1))

            # Apply pre-computed deadzone threshold
            if abs(input_value) < data['deadzone_threshold']:
                input_value = 0.0

            # Store processed value
            self.values[data['output_channel']] = input_value

            # Update pump variable sum if channel affects pump
            if data['affects_pump']:
                self.pump_variable_sum += abs(input_value)

        # Process outputs for all channels and pump
        self.handle_channels(self.values)
        self.handle_pump(self.values)

    def handle_channels(self, values):
        """
        Process each channel's output using pre-computed configuration data.

        Args:
            values: List of input values for each channel
        """
        for channel_name, data in self._channel_data.items():
            output_channel = data['output_channel']

            if output_channel >= len(values):
                print(f"Channel '{channel_name}': No data available.")
                continue

            if not self.toggle_channels and data['toggleable']:
                continue

            input_value = values[output_channel]

            # Use pre-computed gamma values based on input sign
            if input_value >= 0:
                gamma = data['gamma_pos']
                normalized_input = input_value
            else:
                gamma = data['gamma_neg']
                normalized_input = -input_value

            # Apply gamma correction
            adjusted_input = normalized_input ** gamma
            gamma_corrected_value = adjusted_input if input_value >= 0 else -adjusted_input

            # Use pre-computed pulse range and center values
            pulse_width = data['center'] + (gamma_corrected_value * data['pulse_range'] / 2 * data['direction'])
            pulse_width = max(data['pulse_min'], min(data['pulse_max'], pulse_width))

            # Convert to duty cycle
            duty_cycle = int((pulse_width / 20000) * 65535)

            if output_channel < len(self.pca.channels):
                self.pca.channels[output_channel].duty_cycle = duty_cycle
            else:
                print(f"Warning: Channel {output_channel} for '{channel_name}' is not available.")

    def handle_pump(self, values):
        """
        Process the pump output based on configuration.

        Args:
            values: List of input values for channels
        """
        if not hasattr(self, '_pump_data') or self._pump_data is None:
            return

        if not self.pump_enabled:
            throttle_value = -1.0
        elif self._pump_data['input_channel'] is None or self._pump_data['input_channel'] == "None" or self._pump_data[
            'input_channel'] == "none" or self._pump_data['input_channel'] == "null":
            if self.pump_variable:
                throttle_value = self._pump_data['idle'] + (self._pump_data['multiplier'] * self.pump_variable_sum / 10)
            else:
                throttle_value = self._pump_data['idle'] + (self._pump_data['multiplier'] / 10)
            throttle_value += self.manual_pump_load
        elif isinstance(self._pump_data['input_channel'], int) and 0 <= self._pump_data['input_channel'] < len(values):
            throttle_value = values[self._pump_data['input_channel']]
        else:
            throttle_value = self._pump_data['idle']

        throttle_value = max(-1.0, min(1.0, throttle_value))
        pulse_width = self._pump_data['pulse_min'] + (self._pump_data['pulse_max'] - self._pump_data['pulse_min']) * (
                (throttle_value + 1) / 2)

        duty_cycle = int((pulse_width / 20000) * 65535)

        if self._pump_data['channel'] < len(self.pca.channels):
            self.pca.channels[self._pump_data['channel']].duty_cycle = duty_cycle

    def reset(self, reset_pump=True):
        for channel_name, config in self.channel_configs.items():
            if channel_name != 'pump':
                # Fix: Handle case where center is None or string "None" by calculating it on the fly
                center = config.get('center')
                if center is None or center == "None" or center == "none" or center == "null":
                    center = config['pulse_min'] + (config['pulse_max'] - config['pulse_min']) / 2

                duty_cycle = int((center / 20000) * 65535)  # Convert microseconds to duty cycle
                if config['output_channel'] < len(self.pca.channels):
                    self.pca.channels[config['output_channel']].duty_cycle = duty_cycle
                else:
                    print(f"Warning: Channel {config['output_channel']} for '{channel_name}' is not available.")

        if reset_pump and 'pump' in self.channel_configs:
            pump_config = self.channel_configs['pump']
            pump_channel = pump_config['output_channel']
            pulse_width = pump_config['pulse_min']
            duty_cycle = int((pulse_width / 20000) * 65535)  # Convert microseconds to duty cycle
            if pump_channel < len(self.pca.channels):
                self.pca.channels[pump_channel].duty_cycle = duty_cycle
            else:
                print(f"Warning: Pump channel {pump_channel} is not available.")

        self.is_safe_state = False
        self.input_count = 0
        return

    def set_threshold(self, number_value):
        """Update the input rate threshold value."""
        if not isinstance(number_value, (int, float)) or number_value <= 0:
            print("Threshold value must be a positive number.")
            return
        self.input_rate_threshold = number_value
        print(f"Threshold rate set to: {self.input_rate_threshold}Hz")
        return

    def disable_channels(self, bool_value):
        """Enable/Disable channels with toggleable state """
        if not isinstance(bool_value, bool):
            print("State value value must be boolean.")
            return
        self.toggle_channels = bool_value
        print(f"State boolean set to: {self.toggle_channels}!")
        return

    def set_pump(self, bool_value):
        """Enable/Disable pump"""
        if not isinstance(bool_value, bool):
            print("Pump value must be boolean.")
            return
        self.pump_enabled = bool_value
        print(f"Pump enabled set to: {self.pump_enabled}!")
        return

    def toggle_pump_variable(self, bool_value):
        """Enable/Disable pump variable sum (vs static speed)"""
        if not isinstance(bool_value, bool):
            print("Pump variable value must be boolean.")
            return
        self.pump_variable = bool_value
        print(f"Pump variable set to: {self.pump_variable}!")
        return

    def reload_config(self, config_file: str):
        """
        Reload the configuration from the specified file.
        """
        self.reset(reset_pump=True)  # stop everything first
        print(f"Reloading configuration from {config_file}")

        try:
            # Load and validate new configuration
            with open(config_file, 'r') as file:
                new_configs = yaml.safe_load(file)
                new_channel_configs = new_configs['CHANNEL_CONFIGS']

            try:
                self._validate_configuration(new_channel_configs)
            except ValueError as e:
                print(f"Error in new configuration: {e}")
                print("Keeping the current configuration")
                return False

            # Stop monitoring temporarily
            was_monitoring = self.running
            if was_monitoring:
                self._stop_monitoring()

            # Update configuration and rebuild data structures
            self.channel_configs = new_channel_configs
            self._channel_data, self._pump_data = self._build_channel_data()

            # Reset all channels to their new center positions
            self.reset(reset_pump=True)

            # Restart monitoring if it was running
            if was_monitoring:
                self._start_monitoring()

            print("Configuration reloaded successfully")
            return True

        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found")
            return False
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def print_input_mappings(self):
        """Print the input and output mappings for each channel."""
        print("Input mappings:")
        input_to_name_and_output = {}

        for channel_name, config in self.channel_configs.items():
            input_channel = config['input_channel']
            output_channel = config.get('output_channel', 'N/A')  # Get output channel or default to 'N/A'

            # Fix: Check against None, "None", "none", and "null"
            if input_channel is not None and input_channel != "None" and input_channel != "none" and input_channel != "null" and isinstance(
                    input_channel, int):
                if input_channel not in input_to_name_and_output:
                    input_to_name_and_output[input_channel] = []
                input_to_name_and_output[input_channel].append((channel_name, output_channel))

        for input_num in range(self.num_inputs):
            if input_num in input_to_name_and_output:
                names_and_outputs = ', '.join(
                    f"{name} (PWM output {output})" for name, output in input_to_name_and_output[input_num]
                )
                print(f"Input {input_num}: {names_and_outputs}")
            else:
                print(f"Input {input_num}: Not assigned")
        return

    def get_average_input_rate(self) -> float:
        """
        Calculate the average input rate over the last time_window seconds.

        :return: Average input rate in Hz, or 0 if no inputs in the last time_window seconds.
        """
        current_time = time.time()

        # Filter timestamps to last time_window seconds
        recent_timestamps = [t for t in self.input_timestamps if current_time - t <= self.time_window]

        if len(recent_timestamps) < 2:
            return 0.0  # Not enough data to calculate rate

        # Calculate rate based on number of inputs and time span
        time_span = recent_timestamps[-1] - recent_timestamps[0]
        if time_span > 0:
            return (len(recent_timestamps) - 1) / time_span
        else:
            return 0.0  # Avoid division by zero

    def update_pump(self, adjustment):
        """
        Manually update the pump load.

        :param adjustment: The adjustment to add to the pump load (float between -1.0 and 1.0)
        """

        if not isinstance(adjustment, (int, float)):
            print("Pump adjustment value must be a number.")
            return

        # Absolute limits for pump load. ESC dependant
        pump_min = -1.0
        pump_max = 0.3

        self.manual_pump_load = max(pump_min, min(pump_max, self.manual_pump_load + adjustment / 10))
        return

    def reset_pump_load(self):
        """
        Reset the manual pump load to zero.
        """
        self.manual_pump_load = 0.0

        # Re-calculate pump throttle without manual load
        self.handle_pump(self.values)
        return


class PCA9685Stub:
    def __init__(self):
        self.channels = [PCA9685ChannelStub() for _ in range(16)]
        self.frequency = 50


class PCA9685ChannelStub:
    def __init__(self):
        self._duty_cycle = 0

    @property
    def duty_cycle(self):
        return self._duty_cycle

    @duty_cycle.setter
    def duty_cycle(self, value):
        self._duty_cycle = value
        print(f"[SIMULATION] Channel duty cycle set to: {self._duty_cycle}")