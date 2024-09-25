from control_modules import PWM_controller, socket_manager



addr = '192.168.0.136'
port = 5111


identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
# My (num) inputs received from the other end
inputs = 20
# My (num) outputs im going to send
outputs = 0



# init pwm driver
pwm = PWM_controller.PWM_hat(config_file='configuration_files/excavator_channel_configs.yaml',
                             simulation_mode=False,
                             pump_variable=True,
                             tracks_disabled=True,
                             deadzone=0,
                             input_rate_threshold=5
                             )

# init socket
socket = socket_manager.MasiSocketManager()


# set up Excavator as server.
if not socket.setup_socket(addr, port, identification_number, inputs, outputs, socket_type='server'):
    raise Exception("could not setup socket!")
    # setup done

# receive handshake and extra arguments (loop_frequency, int_scale). Also send example arg.
handshake_result, extra_args = socket.handshake(example_arg_you_could_send_here=69)

loop_frequency, int_scale = extra_args[0], extra_args[1]

print(f"Received extra arguments: Loop_freq ({loop_frequency}), int_scale ({int_scale})")

# Update the PWM-controller safety threshold to be 50% of the loop_frequency
pwm.set_threshold(loop_frequency*0.50)

if not handshake_result:
    raise Exception("could not make handshake!")

# switcheroo (to save bandwidth)
socket.tcp_to_udp()
# -------------------------------------------------------------------





def int_to_float(int_data, decimals=2, scale=int_scale):
    """
    To save bandwidth, control signals are converted to 1-byte integers (ranging from -128 to 127) before transmission.
    This process involves scaling the float control signals to fit the 8-bit integer range.
    Values are converted back to floats here.

    The precision of these values is limited by the 256 discrete levels available (due to the 8-bit conversion).
    We are using hobby-grade servos so the lost value resolution doesn't affect basically anything.
    """

    float_data = []  # List to store converted float values
    for value in int_data:
        float_value = round((value / scale), decimals)
        float_data.append(float_value)
    return float_data


# and finally the main loop
def run():
    # start threads for receiving and saving data
    socket.start_data_recv_thread()


    while True:
        # Get latest received joystick values
        value_list = socket.get_latest_received()

        if value_list is not None:
            # Convert received integer values to float
            float_values = int_to_float(value_list)

            # Update the PWM controller with the received values
            # we only use the first 8 values to direct control.
            # Rest of the values (buttons) could be used to e.g. change safety threshold or disable tracks during drive

            # example. Pump load manual adjustement
            #pump_increase = float_values[11]
            #pump_decrease = float_values[12]
            #if pump_increase:
            #    pwm.update_pump(0.3)

            #if pump_decrease:
            #    pwm.update_pump(0.15)

            # example, reload configs
            #if float_values[13]:
            #    pwm.reload_config()

            control_values = float_values[:8]
            pwm.update_values(control_values)


if __name__ == "__main__":
    try:
        run()
    finally:
        # Cleanup
        pwm.reset()              # reset servos and stop the pump
        socket.stop_all()         # close socket connections and kill all used (masi_manager) threads
        pwm.stop_monitoring()    # stop servo safety monitoring thread. You don't have to stop this, and it's probably better to leave running haha

