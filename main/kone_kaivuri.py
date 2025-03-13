# simple code for bluetooth driving. All channels available!

# import PWM (servo controller)
import control_modules.PCA9685_controller as PWM_controller
# import joystick module (evdev-version)
import control_modules.joystick_evdev as joystick_module
# import time for sleep
from time import sleep


def main(pwm, controller):

    # Initialize layer_2_values dictionary with all values set to 0
    layer_2_values = {
        'RightJoystickX': 0,
        'RightJoystickY': 0,
        'RightTrigger': 0,
        'LeftTrigger': 0,
        'LeftJoystickX': 0,
        'LeftJoystickY': 0
    }

    print(pwm.get_input_mapping())
    sleep(5)

    while True:
        if not controller.is_connected():
            # controller not connected, do nothing
            # input threshold check does the safety stuff, no need to call reset here!
            sleep(1)

        else:
            # read all the joystick values
            joy_values = controller.read()
            print(f"joystick values: {joy_values}")


            # flip the LeftTrigger if the LeftBumper is pressed
            if joy_values['LeftBumper']:
                joy_values['LeftTrigger'] = -joy_values['LeftTrigger']

            # flip the RightTrigger if the RightBumper is pressed
            if joy_values['RightBumper']:
                joy_values['RightTrigger'] = -joy_values['RightTrigger']

            # Handle pump control with X and Y buttons
            if joy_values['X']:
                pwm.set_pump(True) # pump on
            elif joy_values['Y']:
                pwm.set_pump(False) # pump off

            # Layer control with A button (not enough buttons on the controller)
            if joy_values['A']:
                # When A is held, update layer_2_values and zero out main layer controls
                layer_2_values.update({
                    'RightJoystickX': joy_values['RightJoystickX'],
                    'RightJoystickY': joy_values['RightJoystickY'],
                    'RightTrigger': joy_values['RightTrigger'],
                    'LeftTrigger': joy_values['LeftTrigger'],
                    'LeftJoystickX': joy_values['LeftJoystickX'],
                    'LeftJoystickY': joy_values['LeftJoystickY']
                })
                # Zero out main layer controls
                joy_values.update({
                    'RightJoystickX': 0,
                    'RightJoystickY': 0,
                    'RightTrigger': 0,
                    'LeftTrigger': 0,
                    'LeftJoystickX': 0,
                    'LeftJoystickY': 0
                })
            else:
                # When A is not held, zero out layer_2_values
                layer_2_values.update({
                    'RightJoystickX': 0,
                    'RightJoystickY': 0,
                    'RightTrigger': 0,
                    'LeftTrigger': 0,
                    'LeftJoystickX': 0,
                    'LeftJoystickY': 0
                })

            # map the joystick values to the servo controller
            # ! pump takes one channel
            controller_list = [
                joy_values['RightJoystickX'],       # scoop, index 0
                joy_values['LeftJoystickY'],        # lift boom, index 1
                joy_values['LeftJoystickX'],        # rotate cabin, index 2
                joy_values['RightJoystickY'],       # tilt boom, index 3
                joy_values['RightTrigger'],         # track R, index 4
                joy_values['LeftTrigger'],          # track L, index 5
                joy_values['DPadY'],                # extra0, index 6
                joy_values['DPadX'],                # extra1, index 7
                layer_2_values['RightJoystickX'],   # extra2, index 8
                layer_2_values['RightJoystickY'],   # extra3, index 9
                layer_2_values['RightTrigger'],     # extra4, index 10
                layer_2_values['LeftTrigger'],      # extra5, index 11
                layer_2_values['LeftJoystickX'],    # extra6, index 12
                layer_2_values['LeftJoystickY'],    # extra7, index 13
                # pump 15 automatically!
            ]

            # update the servo controller with the new values
            pwm.update_values(controller_list)
            #print(f"controller_list: {controller_list}")

            #print(f"rate: {pwm.get_average_input_rate()} Hz")


        sleep(0.02)  # rough 50Hz, good enough here


if __name__ == '__main__':
    # initialize the servo controller
    pwm = PWM_controller.PWMcontroller(
        config_file='configuration_files/kone_config.yaml',
        input_rate_threshold=10,
        pump_variable=True
    )

    # initialize the joystick controller, thats all we need
    controller = joystick_module.XboxController()

    try:
        # run the mainloop
        main(pwm, controller)
    except (KeyboardInterrupt, SystemExit):
        print("Exiting...")
        pwm.reset()