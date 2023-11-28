"""
This is a messy test script to drive around with the excavator.
The script will save ISM330, pressure sensor and pump rpm data. Saves to .bin file
Also the script tests bno08x-IMU, and saves the test results to .txt -file
"""

import universal_socket_manager
import masi_driver
import sensor_manager

import threading
from time import sleep

# Shared variables and lock for thread communication
message_count = 0
message_count_lock = threading.Lock()
stop_thread = False


simulation_mode = False
threshold = 8  # at least this many messages required every interval time

# init socket
manager = universal_socket_manager.MasiSocketManager()
# exceptions soon...

# init servo controller
controller = masi_driver.ExcavatorController(simulation_mode=simulation_mode)

# init IMU's
imu_manager = sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)

# init pressure checking
pressure_manager = sensor_manager.PressureSensor()

# init RPM check
rpm_manager = sensor_manager.RPMSensor()

def setup():
    manager.clear_file()
    setup_result = manager.setup_socket(socket_type='client')
    if not setup_result:
        print("could not set up socket!")
        return False

    handshake_result = manager.handshake()
    if not handshake_result:
        print("could not make handshake!")
    return handshake_result


def message_monitor(threshold, interval=1.0):
    global message_count, stop_thread
    while not stop_thread:
        sleep(interval)
        with message_count_lock:
            if message_count < threshold:
                print("Message count below threshold.")
                controller.reset()
            # Reset message count for the next interval
            message_count = 0

def collect_data():
    # get values from the sensors
    data_i_want_to_save = imu_manager.read_all()  # BNO has problems!

    # get pressure values
    pressure_data = pressure_manager.read_pressure()
    data_i_want_to_save += pressure_data

    # get the pump rpm
    rpm_data = rpm_manager.get_rpm()
    data_i_want_to_save.append(rpm_data)

    print(f"datalen: {len(data_i_want_to_save)}")

    packed_data = manager.pack_data(data_i_want_to_save)

    # Add an empty checksum byte to the packed_data
    # Now data will be in the same format as the packed MotionPlatform joystick data
    packed_data += b'\x00'

    manager.add_data_to_buffer(packed_data)


def run():
    global message_count

    while True:

        # I only want to send handshake
        # outputs are set to 0, handshake is sent automatically
        manager.send_data(data=None)

        # receive joystick values
        data_i_want_to_receive = manager.receive_data()

        # get all the sensor data. should be 25 doubles
        collect_data()
        # this sends them to buffer straight away

        # use joystick values to control the excavator
        controller.update_values(data_i_want_to_receive)

        with message_count_lock:
            message_count += 1


if __name__ == "__main__":
    if setup():
        # Start the message monitoring thread
        monitor_thread = threading.Thread(target=message_monitor, args=(threshold,))
        monitor_thread.start()
        try:
            run()
        finally:
            # Signal the monitoring thread to stop and wait for it to finish
            stop_thread = True
            monitor_thread.join()
            # Cleanup
            manager.save_remaining_data(num_doubles=32)
            manager.close_socket()
            controller.reset()
            rpm_manager.cleanup()
            # Misc
            sleep(2)
            manager.print_bin_file(num_doubles=32)
