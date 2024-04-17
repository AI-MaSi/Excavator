from time import sleep
import threading
import yaml

try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False


class ExcavatorController:
    def __init__(self, inputs, config_file, simulation_mode=False, pump_variable=True,
                 tracks_disabled=False, input_rate_threshold=5, deadzone=6):

        #full_name = f"{config_file}.yaml"

        pwm_channels = 16
        print(f"PWM channels in use: {pwm_channels}")

        # Load configs from .yaml file
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.channel_configs = configs['CHANNEL_CONFIGS']
            # general_settings = configs['GENERAL_SETTINGS']
            # exception if not available

        self.simulation_mode = simulation_mode
        #self.toggle_pump = toggle_pump
        self.pump_variable = pump_variable
        self.tracks_disabled = tracks_disabled

        self.values = [0.0 for _ in range(pwm_channels)]
        self.num_inputs = inputs
        self.num_outputs = pwm_channels

        input_channels = [config['input_channel'] for config in self.channel_configs.values() if config['type'] != 'none' and config['input_channel'] != 'none']
        unique_input_channels = set(input_channels)

        if SERVOKIT_AVAILABLE and not self.simulation_mode:
            self.kit = ServoKit(channels=pwm_channels)
        elif not SERVOKIT_AVAILABLE and not self.simulation_mode:
            raise ServoKitNotAvailableError("ServoKit is not available but required for non-simulation mode.")
        else:
            print("Simulation mode activated! Simulated drive prints will be used.")

        if self.num_inputs < len(unique_input_channels):
            print(f"Warning: The number of inputs specified ({self.num_inputs}) is less than the number of unique input channels used in channel_configs ({len(unique_input_channels)}). This may result in some inputs not being correctly mapped.")
            sleep(3)
        elif self.num_inputs > len(unique_input_channels):
            print(f"Warning: The number of inputs specified ({self.num_inputs}) is more than the number of unique input channels used in channel_configs ({len(unique_input_channels)}). This will result in some inputs being  left out.")
            sleep(3)


        self.input_counter = 0
        self.running = None
        self.monitor_thread = None
        self.input_rate_threshold = input_rate_threshold
        # self.current_hz = None

        self.center_val_servo = 90
        self.deadzone = deadzone

        self.reset()
        self.validate_configuration()
        self.start_monitoring()

    def validate_configuration(self):
        required_keys = ['type', 'input_channel', 'output_channel', 'direction', 'offset']
        angle_specific_keys = ['multiplier_positive', 'multiplier_negative', 'gamma_positive', 'gamma_negative']

        for channel_name, config in self.channel_configs.items():
            # Validate existence of required keys
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing '{key}' in configuration for channel '{channel_name}'")

            # Validate types of operation
            if config['type'] not in ['angle', 'throttle', 'switch', 'none', 'pump']:
                raise ValueError(f"Invalid type '{config['type']}' for channel '{channel_name}'")

            # Validate input_channel
            if isinstance(config['input_channel'], int):
                if not (0 <= config['input_channel'] < self.num_inputs):
                    raise ValueError(
                        f"Input channel {config['input_channel']} out of range for channel '{channel_name}'")
            elif config['input_channel'] != 'none':
                raise ValueError(
                    f"Input channel must be an integer or 'none', got {type(config['input_channel'])} for channel '{channel_name}'")

            # Validate output_channel
            if isinstance(config['output_channel'], int):
                if not (0 <= config['output_channel'] < self.num_outputs):
                    raise ValueError(
                        f"Output channel {config['output_channel']} out of range for channel '{channel_name}'")
            else:
                raise ValueError(
                    f"Output channel must be an integer, got {type(config['output_channel'])} for channel '{channel_name}'")

            # Validate gamma values for angles
            if config['type'] == 'angle':
                for key in angle_specific_keys:
                    if key not in config:
                        raise ValueError(f"Missing '{key}' in configuration for angle type channel '{channel_name}'")
                    if 'gamma' in key:
                        if not (0.1 <= config[key] <= 3.0):
                            raise ValueError(
                                f"{key} {config[key]} out of range (0.1 to 3.0) for channel '{channel_name}'")
                    elif 'multiplier' in key:
                        if not (1 <= abs(config[key]) <= 50):
                            raise ValueError(f"{key} {config[key]} out of range (1 to 50) for channel '{channel_name}'")

            # Validate offset for reasonable adjustments
            if not (-30 <= config['offset'] <= 30):
                raise ValueError(f"Offset {config['offset']} out of range (-30 to 30) for channel '{channel_name}'")

        print("Validating configs done. Jee!")
        print("------------------------------------------\n")

    def start_monitoring(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_input_rate, daemon=True)
            self.monitor_thread.start()
        else:
            print("Monitoring is already running.")

    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
            self.monitor_thread = None
            print("Stopped input rate monitoring...")

    def monitor_input_rate(self, check_interval=0.2):
        print("Monitoring input rate...")
        expected_inputs_per_check = (self.input_rate_threshold * check_interval)
        while self.running:
            current_count = self.input_counter
            sleep(check_interval)
            if (self.input_counter - current_count) < expected_inputs_per_check:
                self.reset(reset_pump=False)
            self.input_counter = 0

    def get_current_rate(self):
        #WIP
        # monitor_input_rate could calculate incoming Hz rate and user could ask it
        # Return the most recently calculated Hz, or None if not yet calculated
        #return getattr(self, 'current_hz', None)
        pass

    def update_values(self, raw_values, min_cap=-1, max_cap=1):
        if raw_values is None:
            self.reset()
            raise ValueError("Input values are None")

        if len(raw_values) != self.num_inputs:
            self.reset()
            raise ValueError(f"Expected {self.num_inputs} inputs, but received {len(raw_values)}.")

        deadzone_threshold = self.deadzone / 100.0 * (max_cap - min_cap)

        for channel_name, config in self.channel_configs.items():
            input_channel = config['input_channel']
            if input_channel == 'none' or input_channel >= len(raw_values):
                continue

            capped_value = max(min_cap, min(raw_values[input_channel], max_cap))
            if abs(capped_value) < deadzone_threshold:
                capped_value = 0.0

            # Apply gamma correction if gamma is specified and value is not zero
            if capped_value != 0:
                if capped_value > 0:
                    gamma = config.get('gamma_positive', 1)  # Default to 1 if not specified
                else:
                    gamma = config.get('gamma_negative', 1)  # Default to 1 if not specified

                normalized_input = (capped_value - min_cap) / (max_cap - min_cap)  # Normalize to 0-1 range
                adjusted_input = normalized_input ** gamma  # Apply gamma power
                gamma_corrected_value = adjusted_input * (max_cap - min_cap) + min_cap  # Scale back to original range
                self.values[config['output_channel']] = gamma_corrected_value
            else:
                self.values[config['output_channel']] = capped_value

            self.input_counter += 1

        self.__use_values(self.values)

    def handle_pump(self, values):
        pump_config = self.channel_configs['pump']
        pump_channel = pump_config['output_channel']
        pump_multiplier = pump_config['multiplier']
        pump_idle = pump_config['idle']

        if self.pump_variable:
            # Calculate the number of active channels that affect the pump
            active_channels_count = sum(1 for channel, config in self.channel_configs.items()
                                        if
                                        config.get('affects_pump', False) and abs(values[config['output_channel']]) > 0)
        else:
            active_channels_count = 2  # Static value if not variable

        # Calculate the pump's throttle setting
        throttle_value = pump_idle + ((pump_multiplier / 100) * active_channels_count)
        throttle_value = max(-1.0, min(1.0, throttle_value))  # Ensure throttle is within valid range

        if self.simulation_mode:
            print(f"Pump level: {active_channels_count} with throttle setting {throttle_value}")
        else:
            self.kit.continuous_servo[pump_channel].throttle = throttle_value

    def handle_angles(self, values):
        for channel_name, config in self.channel_configs.items():
            if config['type'] == 'angle':
                if self.tracks_disabled and channel_name in ['trackL', 'trackR']:
                    continue  # Skip if tracks are disabled

                output_channel = config['output_channel']
                if output_channel >= len(values):  # Validate index
                    if self.simulation_mode:
                        print(f"Simulating channel '{channel_name}': No data available.")
                    continue

                input_value = values[output_channel]
                center = self.center_val_servo + config['offset']

                # Handling positive and negative inputs with respective gamma and multipliers
                if input_value >= 0:
                    gamma = config.get('gamma_positive', 1)
                    multiplier = config.get('multiplier_positive', 1)
                    normalized_input = input_value  # input_value is already [0, 1]
                else:
                    gamma = config.get('gamma_negative', 1)
                    multiplier = config.get('multiplier_negative', 1)
                    normalized_input = -input_value  # Convert to positive

                # Apply gamma correction safely
                adjusted_input = normalized_input ** gamma
                gamma_corrected_value = adjusted_input if input_value >= 0 else -adjusted_input  # Reapply sign if negative

                # Calculate final angle using the directionally adjusted multiplier
                angle = center + (gamma_corrected_value * multiplier * config['direction'])
                angle = max(0, min(180, angle))  # Clamp to valid servo range

                if self.simulation_mode:
                    print(f"Simulating channel '{channel_name}': angle value '{angle}' ({input_value}).")
                else:
                    try:
                        self.kit.servo[config['output_channel']].angle = angle
                    except Exception as e:
                        print(f"Failed to set angle for {channel_name}: {e}")

    def __use_values(self, values):

        self.handle_pump(values)

        self.handle_angles(values)

        self.input_counter += 1

        # more different

    def reset(self, reset_pump=True):
        if self.simulation_mode:
            #print("Simulated reset")
            return

        for config in self.channel_configs.values():
            if config['type'] == 'angle':
                self.kit.servo[config['output_channel']].angle = self.center_val_servo + config.get('offset', 0)

        if reset_pump and 'pump' in self.channel_configs:
            self.kit.continuous_servo[self.channel_configs['pump']['output_channel']].throttle = -1.0


    # Update values during driving
    def set_threshold(self, number_value):
        if not isinstance(number_value, (int, float)):
            #raise TypeError("Threshold value must be an integer or float.")
            print("Threshold value must be an integer.")
            return
        self.input_rate_threshold = number_value
        print(f"Threshold rate set to: {self.input_rate_threshold}Hz")

    def set_deadzone(self, int_value):
        if not isinstance(int_value, (int)):
            #raise TypeError("Deadzone value must be an integer.")
            print("Deadzone value must be an integer.")
            return
        self.deadzone = int_value
        print(f"Deadzone set to: {self.deadzone}%")

    def set_tracks(self, bool_value):
        if not isinstance(bool_value, bool):
            #raise TypeError("Tracks value must be a boolean (True or False).")
            print("Tracks value value must be boolean.")
            return
        self.tracks_disabled = bool_value
        print(f"Tracks boolean set to: {self.tracks_disabled}!")

    def set_pump(self, bool_value):
        if not isinstance(bool_value, bool):
            #raise TypeError("Pump value must be a boolean (True or False).")
            print("Pump value value must be boolean.")
            return
        self.toggle_pump = bool_value
        print(f"Pump boolean set to: {self.toggle_pump}!")


class ServoKitNotAvailableError(Exception):
    pass

class ServoKitWriteError(Exception):
    pass
