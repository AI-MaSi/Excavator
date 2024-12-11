import asyncio
import time
from control_modules import PCA9685_controller, socket_manager, IMU_simple

# ADC board broken (thanks to the one that broke it)

addr = '192.168.0.132'
port = 5111

identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
inputs = 20  # Number of inputs received from the other end
outputs = 18  # 6*3 imu values

# Initialize PWM controller
pwm = PCA9685_controller.PWM_controller(
    config_file='configuration_files/channel_configs.yaml',
    pump_variable=True,
    tracks_disabled=False,
    deadzone=0.8,
    input_rate_threshold=5
    # simulation_move arg removed for now, automatic usage
)

# Initialize IMU reading
imu = IMU_simple.IMUReader()

# Initialize socket
socket = socket_manager.MasiSocketManager()

# Set up Excavator as server
if not socket.setup_socket(addr, port, identification_number, inputs, outputs, socket_type='server'):
    raise Exception("Could not set up socket!")

# Receive handshake and extra arguments
handshake_result, extra_args = socket.handshake(example_arg_you_could_send_here=69, local_datatype='double')

if not handshake_result:
    raise Exception("Could not make handshake!")

loop_frequency, int_scale = extra_args[0], extra_args[1]
print(f"Received extra arguments: Loop_freq ({loop_frequency}), int_scale ({int_scale})")

# Update the PWM-controller safety threshold to be 10% of the loop_frequency
pwm.set_threshold(loop_frequency * 0.10)

# Switch socket communication to UDP (to save bandwidth)
socket.tcp_to_udp()


def int_to_float(int_data, decimals=2, scale=int_scale):
    return [round((value / scale), decimals) for value in int_data]


class ButtonState:
    # Button state tracking
    def __init__(self):
        self.states = {}

    def check_button(self, button_index, value, threshold=0.8):
        # check rising edge. Return True if button is pressed and was not pressed before
        # threshold 0.8 because we mangled the data with int_scale
        current_state = value > threshold
        previous_state = self.states.get(button_index, False)
        self.states[button_index] = current_state
        return current_state and not previous_state


async def imu_data_loop(frequency):
    interval = 1.0 / frequency

    while True:
        start_time = time.time()

        # Read IMU data
        imu_data = imu.read(decimals=5)
        #print(f"Raw IMU data: {imu_data}")

        # Initialize output list with zeros
        output_data = [0] * 18  # 3 IMUs * 6 values each

        # Map the IMU data to the output list
        for i, imu_name in enumerate(['IMU_0', 'IMU_1', 'IMU_2']):
            if imu_name in imu_data:
                base_idx = i * 6  # Each IMU takes 6 positions
                imu_values = imu_data[imu_name]

                # Fill in the values in order: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
                output_data[base_idx] = imu_values['accel_x']
                output_data[base_idx + 1] = imu_values['accel_y']
                output_data[base_idx + 2] = imu_values['accel_z']
                output_data[base_idx + 3] = imu_values['gyro_x']
                output_data[base_idx + 4] = imu_values['gyro_y']
                output_data[base_idx + 5] = imu_values['gyro_z']

        # Send data through socket
        socket.send_data(output_data)
        #print(f"Sent: {output_data}!")

        # Sleep for remaining time to maintain frequency
        elapsed_time = time.time() - start_time
        await asyncio.sleep(max(0, interval - elapsed_time))


async def control_signal_loop(frequency):
    interval = 1.0 / frequency
    step = 0
    button_state = ButtonState()

    while True:
        start_time = time.time()

        value_list = socket.get_latest_received()

        if value_list is not None:
            float_values = int_to_float(value_list)

            control_values = float_values[:8]  # forward the first 8 values (joystick values) to the PWM controller
            pwm.update_values(control_values)

            # print average input rate roughly every second
            if step % loop_frequency == 0:
                print(f"Avg rate: {pwm.get_average_input_rate():.2f} Hz")
            step += 1

            # Button checks (mostly placeholders)
            if button_state.check_button(8, float_values[8]):
                print("Right stick rocker up pressed")

            if button_state.check_button(9, float_values[9]):
                print("Right stick rocker down pressed")

            if button_state.check_button(10, float_values[10]):
                print("Right stick button rear pressed")

            if button_state.check_button(11, float_values[11]):
                print("Right stick button bottom pressed")

            if button_state.check_button(12, float_values[12]):
                print("Reloading PWM controller configuration...")
                pwm.reload_config(config_file='configuration_files/channel_configs.yaml')

            if button_state.check_button(13, float_values[13]):
                print("Right stick button mid pressed")

            if button_state.check_button(14, float_values[14]):
                print("Left stick rocker up pressed")

            if button_state.check_button(15, float_values[15]):
                print("Left stick rocker down pressed")

            if button_state.check_button(16, float_values[16]):
                print("Left stick button rear pressed")

            if button_state.check_button(17, float_values[17]):
                print("Left stick button top pressed")

            if button_state.check_button(18, float_values[18]):
                print("Left stick button bottom pressed")

            if button_state.check_button(19, float_values[19]):
                pwm.reset_pump_load()
                print("Left stick button mid pressed")

        elapsed_time = time.time() - start_time
        await asyncio.sleep(max(0, interval - elapsed_time))


async def async_main():
    control_task = asyncio.create_task(control_signal_loop(20))
    imu_task = asyncio.create_task(imu_data_loop(10))  # 10 Hz frequency
    await asyncio.gather(control_task, imu_task)


if __name__ == "__main__":
    try:
        socket.start_data_recv_thread()
        asyncio.run(async_main())
    finally:
        pwm.reset()
        socket.stop_all()
        pwm._stop_monitoring()