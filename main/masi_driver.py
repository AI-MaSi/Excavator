# this script drives the excavator

# user changeable values
from config import *
from time import sleep
import threading

try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False


class ExcavatorController:
    def __init__(self, simulation_mode=False, toggle_pump=True, pump_variable=True, tracks_disabled=False, num_inputs=20):
        # simulated values instead of real control inputs
        self.simulation_mode = simulation_mode
        # pump on/off, for testing
        self.toggle_pump = toggle_pump
        # variable or stable pump speed
        self.pump_variable = pump_variable
        # enable driving with tracks
        self.tracks_disabled = tracks_disabled

        self.num_inputs = num_inputs
        self.input_counter = 0
        self.values = [0.0 for _ in range(num_inputs)]

        self.input_counter = 0
        self.running = None
        self.monitor_thread = None

        if not self.simulation_mode:
            if SERVOKIT_AVAILABLE:
                # 16 channels in the PWM hat
                self.kit = ServoKit(channels=16)
            else:
                raise ServoKitNotAvailableError("ServoKit is not available but required for non-simulation mode.")

        elif self.simulation_mode:
            print("Simulation mode activated! Simulated drive prints will be used.")

        self.reset()
        self.start_monitoring()

    def start_monitoring(self):
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.stop_monitoring()

        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_input_rate)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()

    def monitor_input_rate(self):
        # this method check how much data we are getting
        print("monitoring input rate...")
        check_interval = 0.2
        input_rate_threshold = 5  # this many needed per second
        expected_inputs_per_check = (input_rate_threshold * check_interval)
        while True:
            current_count = self.input_counter
            sleep(check_interval)
            if (self.input_counter - current_count) < expected_inputs_per_check:
                self.reset(reset_pump=False)

            self.input_counter = 0

    def update_values(self, raw_values, min_cap=-1, max_cap=1):
        # check values and make them go in the right place here
        if len(raw_values) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but received {len(raw_values)}.")

        # Calculate threshold value from deadzone % and min/max value
        deadzone_threshold = deadzone / 100.0 * (max_cap - min_cap)

        # Iterate through channel configurations
        for channel_name, config in CHANNEL_CONFIGS.items():
            if config['type'] != 'none':
                input_channel = config['input_channel']
                output_channel = config['output_channel']

                # Skip if input_channel is 'none' or if index is out of range
                if input_channel == 'none' or input_channel >= len(raw_values):
                    continue

                # Apply limits to raw value
                capped_value = max(min_cap, min(raw_values[input_channel], max_cap))

                # Apply deadzone. Same for all channels now...
                if abs(capped_value) < deadzone_threshold:
                    capped_value = 0

                # Assign the processed value to the correct output channel
                self.values[output_channel] = capped_value

                self.input_counter += 1

        # Use these values
        self.use_values(self.values)

    def use_values(self, values):
        if self.pump_variable:
            # Deadzone already handled, increase multiplier if value is not 0
            active_channels_count = sum(1 for channel, config in CHANNEL_CONFIGS.items() if config['affects_pump']
                                        and abs(values[config['output_channel']]) > 0)
        else:
            # spin pump a bit faster (than idle) with fixed speed
            active_channels_count = 3

        if self.simulation_mode:
            print(f"values: {values}")

        if 'pump' in CHANNEL_CONFIGS and CHANNEL_CONFIGS['pump']['type'] == 'throttle':
            pump_channel = CHANNEL_CONFIGS['pump']['output_channel']
            pump_multiplier = CHANNEL_CONFIGS['pump']['multiplier']
            pump_idle = CHANNEL_CONFIGS['pump']['idle']

            if self.toggle_pump:
                self.kit.continuous_servo[pump_channel].throttle = pump_idle + (
                            (pump_multiplier / 100) * active_channels_count)
            else:
                self.kit.continuous_servo[pump_channel].throttle = -1.0

        try:
            # Adjust the servo angles
            for channel_name, config in CHANNEL_CONFIGS.items():
                if config['type'] == 'angle':
                    # Skip setting if tracks are disabled
                    if self.tracks_disabled and channel_name in ['trackR', 'trackL']:
                        continue

                    value = config['offset'] + center_val_servo + (config['direction'] * values[config['output_channel']]
                                                                   * config['multiplier'])

                    # set angle values
                    self.kit.servo[config['output_channel']].angle = value

                # set throttle value
                # add not pump!!
                #elif config['type'] == 'throttle':
                    #self.kit.continuous_servo[config['output_channel']].throttle = value


                # finish this. check absolute etc.!!
                elif config['type'] == 'switch':
                    value = values[config['output_channel']]
                    if value > config['threshold']:
                        # set true
                        pass

                    else:
                        # set false
                        pass

        except (ValueError, IndexError, KeyError) as e:
            self.reset()
            raise ServoKitWriteError(f"Failed to set values! Error: {e}")

    def reset(self, reset_pump=True):
        if self.simulation_mode:
            print("Simulated reset")
            return

        if reset_pump:
            # Set the pump to -1 throttle
            if 'pump' in CHANNEL_CONFIGS and CHANNEL_CONFIGS['pump']['type'] == 'throttle':
                self.kit.continuous_servo[CHANNEL_CONFIGS['pump']['output_channel']].throttle = -1.0

        # Reset all servos that have a type of 'angle' to their center value
        for channel_name, config in CHANNEL_CONFIGS.items():
            if config['type'] == 'angle':
                self.kit.servo[config['output_channel']].angle = center_val_servo + config.get('offset', 0)


class ServoKitNotAvailableError(Exception):
    pass


class ServoKitWriteError(Exception):
    pass
