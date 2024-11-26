# new PWM controller without ServoKit. Using lower-level adafruit_pca9685 library instead.

import threading
import yaml
import time

try:
    from adafruit_pca9685 import PCA9685
    import board
    import busio
    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False
    print("PWM module not found. Running in simulation mode.\n")
    time.sleep(3)


class PWM_controller:
    # simulation mode check removed as it comes on automatically
    def __init__(self, config_file: str, pump_variable: bool = False,
                 tracks_disabled: bool = False, input_rate_threshold: float = 0, deadzone: float = 0) -> None:

        self.pump_variable = pump_variable
        self.tracks_disabled = tracks_disabled
        self.deadzone = deadzone

        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.channel_configs = configs['CHANNEL_CONFIGS']

        self.num_channels = 16
        self.values = [0.0 for _ in range(self.num_channels)]
        self.num_inputs = self.calculate_num_inputs()

        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking

        self.input_event = threading.Event()
        self.monitor_thread = None
        self.running = False

        self.input_count = 0
        self.last_input_time = time.time()
        self.input_timestamps = []

        self.pump_enabled = True
        self.pump_variable_sum = 0.0
        self.manual_pump_load = 0.0

        if SERVOKIT_AVAILABLE:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50  # Set to 50Hz for standard servos
        else:
            self.pca = PCA9685Stub()

        self._validate_configuration(self.channel_configs)

        self.reset()

        if not self.skip_rate_checking:
            self._start_monitoring()
        return

    def calculate_num_inputs(self) -> int:
        input_channels = set()
        for config in self.channel_configs.values():
            input_channel = config.get('input_channel')
            if isinstance(input_channel, int):
                input_channels.add(input_channel)
        return len(input_channels)

    @staticmethod
    def _validate_configuration(config):
        """
        Validate the given configuration.

        :param config: Configuration dictionary to validate
        :raises ValueError: If the configuration is invalid
        """
        channels = 16
        gamma_min = 0.1
        gamma_max = 3.0
        pulse_min = 0  # find good safe values!
        pulse_max = 4095  # find good safe values!
        pump_idle_min = -1.0
        pump_idle_max = 0.3
        pump_multiplier_max = 10.0

        required_keys = ['input_channel', 'output_channel', 'pulse_min', 'pulse_max', 'center']
        pump_required_keys = ['idle', 'multiplier']

        used_input_channels = {}
        used_output_channels = {}

        for channel_name, channel_config in config.items():
            # Check for required keys
            for key in required_keys:
                if key not in channel_config:
                    raise ValueError(f"Missing '{key}' in configuration for channel '{channel_name}'")

            # Check for duplicate input channels
            input_channel = channel_config['input_channel']
            if input_channel != 'None' and isinstance(input_channel, int):
                if input_channel in used_input_channels:
                    raise ValueError(
                        f"Input channel {input_channel} is used by both '{channel_name}' and '{used_input_channels[input_channel]}'")
                used_input_channels[input_channel] = channel_name

            # Check for duplicate output channels
            output_channel = channel_config['output_channel']
            if output_channel in used_output_channels:
                raise ValueError(
                    f"Output channel {output_channel} is used by both '{channel_name}' and '{used_output_channels[output_channel]}'")
            used_output_channels[output_channel] = channel_name

            # Validate servo range
            if channel_config['pulse_min'] >= channel_config['pulse_max']:
                raise ValueError(f"pulse_min must be less than pulse_max for channel '{channel_name}'")

            if channel_config['center'] < channel_config['pulse_min'] or channel_config['center'] > channel_config[
                'pulse_max']:
                raise ValueError(f"center must be between pulse_min and pulse_max for channel '{channel_name}'")

            # Validate pulse width ranges
            if not (pulse_min <= channel_config['pulse_min'] <= pulse_max):
                raise ValueError(f"pulse_min must be between {pulse_min} and {pulse_max} for channel '{channel_name}'")
            if not (pulse_min <= channel_config['pulse_max'] <= pulse_max):
                raise ValueError(f"pulse_max must be between {pulse_min} and {pulse_max} for channel '{channel_name}'")
            if not (pulse_min <= channel_config['center'] <= pulse_max):
                raise ValueError(f"center must be between {pulse_min} and {pulse_max} for channel '{channel_name}'")

            # Validate input and output channels
            if not (0 <= channel_config['output_channel'] < channels):
                raise ValueError(f"output_channel must be between 0 and {channels - 1} for channel '{channel_name}'")

            if channel_config['input_channel'] != 'None' and not isinstance(channel_config['input_channel'], int):
                raise ValueError(f"input_channel must be an integer or 'None' for channel '{channel_name}'")

            if isinstance(channel_config['input_channel'], int) and not (
                    0 <= channel_config['input_channel'] < channels):
                raise ValueError(f"input_channel must be between 0 and {channels - 1} for channel '{channel_name}'")

            # Validate direction
            if 'direction' in channel_config and channel_config['direction'] not in [-1, 1]:
                raise ValueError(f"direction must be either -1 or 1 for channel '{channel_name}'")

            # Validate gamma values
            for gamma_key in ['gamma_positive', 'gamma_negative']:
                if gamma_key in channel_config:
                    gamma_value = channel_config[gamma_key]
                    if not (gamma_min <= gamma_value <= gamma_max):
                        raise ValueError(
                            f"{gamma_key} must be between {gamma_min} and {gamma_max} for channel '{channel_name}'")

            # Validate affects_pump
            if 'affects_pump' in channel_config and not isinstance(channel_config['affects_pump'], bool):
                raise ValueError(f"affects_pump must be a boolean for channel '{channel_name}'")

            # Pump-specific validations
            if channel_name == 'pump':
                for key in pump_required_keys:
                    if key not in channel_config:
                        raise ValueError(f"Missing '{key}' in configuration for pump channel")

                if not (pump_idle_min <= channel_config['idle'] <= pump_idle_max):
                    raise ValueError(f"idle must be between {pump_idle_min} and {pump_idle_max} for pump channel")

                if not (0 < channel_config['multiplier'] <= pump_multiplier_max):
                    raise ValueError(f"multiplier must be between 0 and {pump_multiplier_max} for pump channel")

        return

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

    def monitor_input_rate(self) -> None:
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
                        if self.input_count >= int(self.input_rate_threshold * 0.25):
                            self.is_safe_state = True
                            self.input_count = 0
                    else:
                        self.input_count = 0

                    # Save the timestamp for monitoring
                    self.input_timestamps.append(current_time)

                    # Remove timestamps older than 30 seconds
                    self.input_timestamps = [t for t in self.input_timestamps if current_time - t <= 30]

            else:
                if self.is_safe_state:
                    print("Input rate too low. Entering safe state...")
                    self.reset(reset_pump=False)
                    self.is_safe_state = False
                    self.input_count = 0

    def update_values(self, raw_value_list):
        # input limits
        min_cap = -1
        max_cap = 1

        if not self.skip_rate_checking:
            # Wake up the monitoring thread
            self.input_event.set()

        if not self.skip_rate_checking and not self.is_safe_state:
            print(f"System in safe state. Ignoring input. Average rate: {self.get_average_input_rate():.2f}Hz")
            return

        if raw_value_list is None:
            self.reset()
            raise ValueError("Input values are None")

        if isinstance(raw_value_list, (float, int)):
            # convert single value to list
            raw_value_list = [raw_value_list]

        if len(raw_value_list) != self.num_inputs:
            self.reset()
            raise ValueError(f"Expected {self.num_inputs} inputs, but received {len(raw_value_list)}.")

        deadzone_threshold = self.deadzone / 100.0 * (max_cap - min_cap)

        self.pump_variable_sum = 0.0
        for channel_name, config in self.channel_configs.items():
            input_channel = config['input_channel']
            if input_channel is None or not isinstance(input_channel, int) or input_channel >= len(raw_value_list):
                continue

            capped_value = max(min_cap, min(raw_value_list[input_channel], max_cap))

            if abs(capped_value) < deadzone_threshold:
                capped_value = 0.0

            self.values[config['output_channel']] = capped_value

            if config.get('affects_pump', False):
                self.pump_variable_sum += abs(capped_value)

        self.handle_channels(self.values)
        self.handle_pump(self.values)
        return

    def handle_channels(self, values):
        for channel_name, config in self.channel_configs.items():
            if channel_name == 'pump':
                # Pump has a special handling, so we skip it here
                continue

            if self.tracks_disabled and channel_name in ['trackL', 'trackR']:
                # Tracks have been disabled, so we skip them here
                continue

            output_channel = config['output_channel']

            if output_channel >= len(values):
                print(f"Channel '{channel_name}': No data available.")
                continue

            input_value = values[output_channel]
            center = config['center']

            if input_value >= 0:
                gamma = config.get('gamma_positive', 1)
                normalized_input = input_value
            else:
                gamma = config.get('gamma_negative', 1)
                normalized_input = -input_value

            adjusted_input = normalized_input ** gamma
            gamma_corrected_value = adjusted_input if input_value >= 0 else -adjusted_input

            # Map the input (-1 to 1) to the servo range
            pulse_range = config['pulse_max'] - config['pulse_min']
            pulse_width = center + (gamma_corrected_value * pulse_range / 2 * config['direction'])
            pulse_width = max(config['pulse_min'], min(config['pulse_max'], pulse_width))

            duty_cycle = int((pulse_width / 20000) * 65535)  # Convert microseconds to duty cycle

            # Check if the channel is available before setting it
            if output_channel < len(self.pca.channels):
                self.pca.channels[output_channel].duty_cycle = duty_cycle
            else:
                print(f"Warning: Channel {output_channel} for '{channel_name}' is not available.")

    def handle_pump(self, values):
        if 'pump' not in self.channel_configs:
            return  # Silently return if pump is not configured

        pump_config = self.channel_configs['pump']
        pump_channel = pump_config['output_channel']
        pump_multiplier = pump_config['multiplier']
        pump_idle = pump_config['idle']
        input_channel = pump_config.get('input_channel')

        if not self.pump_enabled:
            throttle_value = -1.0
        elif input_channel is None or input_channel == 'None':
            if self.pump_variable:
                throttle_value = pump_idle + (pump_multiplier * self.pump_variable_sum)
            else:
                throttle_value = pump_idle + (pump_multiplier / 10)
            throttle_value += self.manual_pump_load
        elif isinstance(input_channel, int) and 0 <= input_channel < len(values):
            throttle_value = values[input_channel]
        else:
            print(f"Warning: Invalid input channel {input_channel} for pump. Using pump_idle.")
            throttle_value = pump_idle

        throttle_value = max(-1.0, min(1.0, throttle_value))
        pulse_width = pump_config['pulse_min'] + (pump_config['pulse_max'] - pump_config['pulse_min']) * (
                (throttle_value + 1) / 2)
        duty_cycle = int((pulse_width / 20000) * 65535)  # Convert microseconds to duty cycle

        if pump_channel < len(self.pca.channels):
            self.pca.channels[pump_channel].duty_cycle = duty_cycle
        else:
            print(f"Warning: Pump channel {pump_channel} is not available.")

    def reset(self, reset_pump=True):
        for channel_name, config in self.channel_configs.items():
            if channel_name != 'pump':
                center = config['center']
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

    def set_deadzone(self, int_value):
        """Update the Deadzone value"""
        if not isinstance(int_value, int):
            print("Deadzone value must be an integer.")
            return
        self.deadzone = int_value
        print(f"Deadzone set to: {self.deadzone}%")
        return

    def set_tracks(self, bool_value):
        """Enable/Disable tracks"""
        if not isinstance(bool_value, bool):
            print("Tracks value value must be boolean.")
            return
        self.tracks_disabled = bool_value
        print(f"Tracks boolean set to: {self.tracks_disabled}!")
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

        :param config_file: Path to the new configuration file
        """
        print(f"Reloading configuration from {config_file}")

        # Load the new configuration
        with open(config_file, 'r') as file:
            new_configs = yaml.safe_load(file)
            new_channel_configs = new_configs['CHANNEL_CONFIGS']

        # Validate the new configuration
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

        # Update the configuration
        self.channel_configs = new_channel_configs
        self.num_inputs = self.calculate_num_inputs()

        # Reset all channels to their new center positions
        self.reset(reset_pump=True)

        # Restart monitoring if it was running before
        if was_monitoring:
            self._start_monitoring()

        print("Configuration reloaded successfully")
        return True

    def print_input_mappings(self):
        """Print the input and output mappings for each channel."""
        print("Input mappings:")
        input_to_name_and_output = {}

        for channel_name, config in self.channel_configs.items():
            input_channel = config['input_channel']
            output_channel = config.get('output_channel', 'N/A')  # Get output channel or default to 'N/A'

            if input_channel != 'none' and isinstance(input_channel, int):
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
        Calculate the average input rate over the last 30 seconds.

        :return: Average input rate in Hz, or 0 if no inputs in the last 30 seconds.
        """
        current_time = time.time()

        # Filter timestamps to last 30 seconds
        recent_timestamps = [t for t in self.input_timestamps if current_time - t <= 30]

        if len(recent_timestamps) < 2:
            return 0.0  # Not enough data to calculate rate

        # Calculate rate based on number of inputs and time span
        time_span = recent_timestamps[-1] - recent_timestamps[0]
        if time_span > 0:
            return (len(recent_timestamps) - 1) / time_span
        else:
            return 0.0  # Avoid division by zero

    def update_pump(self,adjustment):
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

        self.manual_pump_load = max(pump_min, min(pump_max, self.manual_pump_load + adjustment/10))
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