# UPD / TCP hybrid connection demo
# handshake with TCP, data transmission with UDP

import universal_connection_manager
import masi_driver
import sensor_manager
from time import sleep


addr = '192.168.0.136'
#addr = '127.0.0.1'
port = 5111

# Who am I. Check config for names
identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
# My (num) inputs received from the other end
inputs = 20
# My (num) outputs im going to send
outputs = 0


# init pwm driver
controller = masi_driver.ExcavatorController(inputs=inputs,
                                             simulation_mode=False,
                                             config_file='driver_config_Motionplatform.yaml',
                                             pump_variable=False,)

# init socket
masi_manager = universal_connection_manager.MasiSocketManager()


# set up Excavator as server.
if not masi_manager.setup_socket(addr, port, identification_number, inputs, outputs, socket_type='server'):
    raise Exception("could not setup socket!")
    # setup done

# receive handshake and extra arguments (loop_frequency, int_scale). Also send example arg.
handshake_result, extra_args = masi_manager.handshake(example_arg_you_could_send_here=69)

loop_frequency, int_scale = extra_args[0], extra_args[1]

print(f"Received extra arguments: Loop_freq ({loop_frequency}), int_scale ({int_scale})")

# Update the PWM-controller safety threshold to be 50% of the loop_frequency
controller.set_threshold(loop_frequency*0.50)

if not handshake_result:
    raise Exception("could not make handshake!")

# switcheroo
masi_manager.tcp_to_udp()
# -------------------------------------------------------------------


# init pressure sensors
pressures = sensor_manager.PressureSensor()

# init IMUs
imus = sensor_manager.IMUSensorManager(simulation_mode=False)

# init RPM check
rpm = sensor_manager.RPMSensor()


def collect_data(received_float_values):        # control signal. 20 values
    imu_data = imus.read_all()                  # get values from the sensors. 4x6 values
    #print(f"imu data: {imu_data}")
    received_float_values.extend(imu_data)
    pressure_data = pressures.read_pressure()   # get pressure values. 7 values
    #print(f"pressure data: {pressure_data}")
    received_float_values.extend(pressure_data)
    rpm_data = rpm.read_rpm()                  # get the pump rpm. 1 value converted to a list.
    #print(f"rpm data: {rpm_data}")
    received_float_values.append(rpm_data)
    #print(f"full data: {received_float_values}")
    #sleep(10)

    #print(f"len data to buffer: {len(received_float_values)}")
    masi_manager.add_data_to_buffer(received_float_values)  # 52 values added

def get_datalen():
    #WIP
    # get the length of the full data list
    return 52

def int_to_float(int_data, decimals=2, scale=int_scale):
    """
    To save bandwidth, control signals are converted to 1-byte integers (ranging from -128 to 127) before transmission.
    This process involves scaling the float control signals to fit the 8-bit integer range.
    Values are converted back to floats here.

    The precision of these values is limited by the 256 discrete levels available (due to the 8-bit conversion).
    We are using hobby-grade servos so the lost value resolution doesn't affect anything.
    """

    float_data = []  # List to store converted float values
    for value in int_data:
        float_value = round((value / scale), decimals)
        float_data.append(float_value)
    return float_data


def run():
    # clear the file
    masi_manager.clear_file()
    # start threads for receiving and saving data
    masi_manager.start_data_recv_thread()
    masi_manager.start_saving_thread()

    while True:
        # Get latest received joystick values
        value_list = masi_manager.get_latest_received()

        if value_list is not None:
            float_values = int_to_float(value_list)

            controller.update_values(float_values)

            # save all data to buffer. Add received control data
            collect_data(float_values)

        #sleep(1/loop_frequency)  # relax time, could be made way better especially for higher Hz usage

if __name__ == "__main__":
    try:
        run()
    finally:
        # Cleanup

        # reset servos and stop the pump
        controller.reset()

        # close socket connections and kill all used (masi_manager) threads
        masi_manager.stop_all()
        # clean up used GPIO pins
        rpm.cleanup()

        # stop servo safety monitoring thread. You don't have to stop this, and it's probably better to leave running haha
        controller.stop_monitoring()

        # Misc. Print the saved values.
        masi_manager.print_bin_file(num_values=get_datalen()) # num_values is the length of the data added to the buffer!
