import asyncio
import time
from control_modules import PWM_controller, socket_manager, IMU_sensors

addr = '192.168.0.136'
port = 5111

identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
inputs = 20  # Number of inputs received from the other end
outputs = 4  # Number of outputs I'm going to send. three values for testing

# Initialize PWM controller
# For this demo, we're not fetching the full configuration from the excavator_config.yaml file,
pwm = PWM_controller.PWM_hat(
    config_file='configuration_files/excavator_channel_configs.yaml',
    simulation_mode=False,
    pump_variable=True,
    tracks_disabled=True,
    deadzone=0,
    input_rate_threshold=5
)

# Initialize IMU sensor
imu = IMU_sensors.ISM330DHCX(
    config_file='configuration_files/excavator_sensor_configs.yaml',
)

# Initialize socket
socket = socket_manager.MasiSocketManager()

# Set up Excavator as server
if not socket.setup_socket(addr, port, identification_number, inputs, outputs, socket_type='server'):
    raise Exception("Could not set up socket!")

# Receive handshake and extra arguments (loop_frequency, int_scale). Also, send example argument.
# Note: We're not specifying local_datatype here, so it defaults to 'double' (float)
# This means we'll be sending floats but can still receive integers if the client sends them
handshake_result, extra_args = socket.handshake(example_arg_you_could_send_here=69)

if not handshake_result:
    raise Exception("Could not make handshake!")

loop_frequency, int_scale = extra_args[0], extra_args[1]
print(f"Received extra arguments: Loop_freq ({loop_frequency}), int_scale ({int_scale})")

# Update the PWM-controller safety threshold to be 10% of the loop_frequency
pwm.set_threshold(loop_frequency * 0.10)

# Switch socket communication to UDP (to save bandwidth)
socket.tcp_to_udp()


# Function to convert received integer values to float
def int_to_float(int_data, decimals=2, scale=int_scale):
    """
    To save bandwidth, control signals are converted to 1-byte integers (ranging from -128 to 127) before transmission.
    This process involves scaling the float control signals to fit the 8-bit integer range.
    Values are converted back to floats here.

    Note: This function is necessary only if the client is sending integers.
    If the client is sending floats, this conversion is not needed.
    """
    float_data = [round((value / scale), decimals) for value in int_data]
    return float_data


# Main async loop
async def async_main():
    # Define control signal loop (20Hz for receiving)
    control_task = asyncio.create_task(control_signal_loop(20))

    # Define sensor feedback loop (5Hz for sending)
    sensor_task = asyncio.create_task(sensor_signal_loop(5))

    # Run both loops concurrently
    await asyncio.gather(control_task, sensor_task)


# Control signal loop (20Hz)
async def control_signal_loop(frequency):
    interval = 1.0 / frequency  # 20Hz = 1/20 seconds per loop
    button_pressed = False
    while True:
        start_time = time.time()

        # Get the latest received joystick values
        value_list = socket.get_latest_received()

        if value_list is not None:
            # Convert received integer values to float if necessary
            # Note: If the client is sending floats, this conversion is not needed
            # and you can directly use value_list instead of float_values
            float_values = int_to_float(value_list)

            # Update the PWM controller with the received values (values 0-7)
            control_values = float_values[:8]
            pwm.update_values(control_values)

            # Check if the reload config button is pressed (index 12)
            # (# 12. right stick button top) Look up the mapping from the Git Docs, or form the NiDAQ_controller.py
            button_state = float_values[12] > 0.8
            # 0 False 1 True but we're using 0.8 as a threshold as we mangled the data with int_to_float

            # Detect button press (transition from not pressed to pressed)
            if button_state and not button_pressed:
                print("Reloading PWM controller configuration...")
                pwm.reload_config(
                    config_file='configuration_files/excavator_channel_configs.yaml')

                button_pressed = True
            # Detect button release, so we only reload once per press
            elif not button_state and button_pressed:
                button_pressed = False


        # Wait to maintain 20Hz frequency
        elapsed_time = time.time() - start_time
        await asyncio.sleep(max(0, interval - elapsed_time))


# Sensor feedback loop (5Hz)
async def sensor_signal_loop(frequency):
    interval = 1.0 / frequency  # 5Hz = 1/5 seconds per loop

    while True:
        start_time = time.time()

        # Send sensor data or feedback back to the client
        # Note: We're sending floats here. If you need to send integers,
        # you would need to use float_to_int() function from the sending end
        imu_scoop_angles = imu.read_ism330(channel=3, read_mode='angles')
        imu_values = list(imu_scoop_angles.values())
        print(f"IMU scoop kalman: {imu_values}")

        # Placeholder, I need to first check the IMU data (and remove names etc., only values for UDP!)
        #feedback = [6.9] * outputs
        socket.send_data(imu_values)

        # Wait to maintain 5Hz frequency
        elapsed_time = time.time() - start_time
        await asyncio.sleep(max(0, interval - elapsed_time))


# Run the main event loop
def run():
    # Start the thread for receiving data
    socket.start_data_recv_thread()

    # Create and run asyncio event loop
    asyncio.run(async_main())


# Entry point
if __name__ == "__main__":
    try:
        run()
    finally:
        # Cleanup
        pwm.reset()  # Reset servos and stop the pump
        socket.stop_all()  # Close socket connections and kill all threads
        pwm.stop_monitoring()  # Stop servo safety monitoring thread