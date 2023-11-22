# this script drives the excavator

# user changeable values
from config import *
from time import sleep

try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except ImportError:
    SERVOKIT_AVAILABLE = False

toggle_pump = True


class ExcavatorController:
    def __init__(self, simulation_mode=False, num_inputs=20):
        self.simulation_mode = simulation_mode
        self.num_inputs = num_inputs
        self.values = [0.0 for _ in range(num_inputs)]

        if not self.simulation_mode:
            if SERVOKIT_AVAILABLE:
                # 16 channels in the PWM hat
                self.kit = ServoKit(channels=16)
            else:
                raise ServoKitNotAvailableError("ServoKit is not available but required for non-simulation mode.")

        elif self.simulation_mode:
            print("Simulation mode activated! Simulated drive prints will be used.")

        self.reset()

    def update_values(self, raw_values):
        if len(raw_values) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but received {len(raw_values)}.")

        # Iterate through channel configurations
        for channel_name, config in CHANNEL_CONFIGS.items():
            if config['type'] != 'none':
                input_channel = config['input_channel']
                output_channel = config['output_channel']

                """
                # Check if the input_channel is within the valid range
                if 0 <= input_channel < len(raw_values):
                    # Check if the input value is out of bounds (-1 to 1) and cap it
                    if raw_values[input_channel] > 1:
                        raw_values[input_channel] = 1
                    elif raw_values[input_channel] < -1:
                        raw_values[input_channel] = -1
                """

                # Skip if input_channel is 'none' or if index is out of range
                if input_channel == 'none' or input_channel >= len(raw_values):
                    continue

                # Assign the value from the input channel to the correct output channel
                self.values[output_channel] = raw_values[input_channel]

        # Use these values
        self.use_values(self.values)

    def use_values(self, values):
        # Increase pump speed when set channels go over a certain threshold
        pump_variable = CHANNEL_CONFIGS['pump']['multiplier']
        if pump_variable:
            active_channels_count = sum([1 for i in hydraulic_multiplier_channels if abs(values[i]) > (deadzone / 100)])
        else:
            active_channels_count = 0

        if self.simulation_mode:
            print(f"values: {values}")
            for i in hydraulic_multiplier_channels:
                print(f"Hydraulic channel {i}, Value: {values[i]}, Abs Value: {abs(values[i])}, Deadzone: {deadzone}%")
                if abs(values[i]) > deadzone / 100:
                    print(f"Channel {i} is active!")
            return

        if 'pump' in CHANNEL_CONFIGS and CHANNEL_CONFIGS['pump']['type'] == 'throttle':
            pump_channel = CHANNEL_CONFIGS['pump']['output_channel']
            pump_multiplier = CHANNEL_CONFIGS['pump']['multiplier']
            pump_idle = CHANNEL_CONFIGS['pump']['idle']

            if toggle_pump:
                # and pump_variable
                self.kit.continuous_servo[pump_channel].throttle = pump_idle + (
                            (pump_multiplier / 100) * active_channels_count)
            else:
                self.kit.continuous_servo[pump_channel].throttle = -0.9

        # Adjust the servo angles
        for channel_name, config in CHANNEL_CONFIGS.items():
            if config['type'] == 'angle':
                value = config['offset'] + center_val_servo + (config['direction'] * values[config['output_channel']]
                                                            * config['multiplier'])

                # give the value to the servo
                self.kit.servo[config['output_channel']].angle = value


                # TEST THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #elif config['type'] == 'throttle':
                    #self.kit.continuous_servo[config['output_channel']].throttle = value

    def reset(self):
        if self.simulation_mode:
            print("Simulated reset")
            sleep(2)
            return

        # Set the pump to -1 throttle
        if 'pump' in CHANNEL_CONFIGS and CHANNEL_CONFIGS['pump']['type'] == 'throttle':
            self.kit.continuous_servo[CHANNEL_CONFIGS['pump']['output_channel']].throttle = -0.9

        # Reset all servos that have a type of 'angle' to their center value
        for channel_name, config in CHANNEL_CONFIGS.items():
            if config['type'] == 'angle':
                self.kit.servo[config['output_channel']].angle = center_val_servo + config.get('offset', 0)

        print("Reseted servos!")
        sleep(2)

class ServoKitNotAvailableError(Exception):
    pass

class ServoKitWriteError(Exception):
    pass