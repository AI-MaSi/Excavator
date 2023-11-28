"""
This is a messy test script to drive around with the excavator.
The script will save ISM330, pressure sensor and pump rpm data. Saves to .bin file
Also the script tests bno08x-IMU, and saves the test results to .txt -file
"""

import universal_socket_manager
import masi_driver
import sensor_manager
# bno08x is messed up
import bno_tester


import threading
from time import time, sleep

# num_passes = int(input("Enter the (thousand) number of measurements to perform: "))
# etc. etc. etc...

simulation_mode = False
threshold = 2  # at least this many messages required every second

# init socket
manager = universal_socket_manager.MasiSocketManager()
# exceptions soon...

# init servo controller
controller = masi_driver.ExcavatorController(simulation_mode=simulation_mode)
# exceptions soon...

# init IMU's
try:
    imu_manager = sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)
    # imu_manager = sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)
# except (imu_sensor_manager.BNO08xInitializationError, imu_sensor_manager.ISM330InitializationError) as e:
except (sensor_manager.BNO08xInitializationError, sensor_manager.ISM330InitializationError) as e:
    imu_manager = None
    raise e

# init pressure checking
pressure_manager = sensor_manager.PressureSensor()
# exceptions soon...

# init RPM check
rpm_manager = sensor_manager.RPMSensor()
# exceptions soon...


def bno08x_test_thread(num_passes, frequency):
    test_results = []
    test_results.append((f'Test with {frequency} Hz', bno_tester.test_bno08x(num_passes, frequency)))
    bno_tester.save_test_results_to_file(test_results, num_passes, "bno08x_test_results.txt")
    print("BNO08x test results saved to 'bno08x_test_results.txt'")

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

def collect_data():
    # get values from the sensors
    data_i_want_to_save = imu_manager.read_all()  # BNO has problems!

    # get pressure values
    pressure_data = pressure_manager.read_pressure()
    if pressure_data is not None:
        data_i_want_to_save += pressure_data
    else:
        print("pressure data is None!!!!!!!!!")

    # get the pump rpm
    rpm_data = rpm_manager.get_rpm()
    if rpm_data is not None:
        #data_i_want_to_save += rpm_data
        data_i_want_to_save.append(rpm_data)
    else:
        print("RPM data is None!!!!!!!!!")

    packed_data = manager.pack_data(data_i_want_to_save)

    # Add an empty checksum byte to the packed_data
    # Now data will be in the same format as the packed MotionPlatform joystick data
    packed_data += b'\x00'

    manager.add_data_to_buffer(packed_data)


def run():
    check_interval = 1.0
    data_received_count = 0
    start_time = time()

    log_data = True

    while True:

        try:
            # I only want to send handshake
            # outputs are set to 0, handshake is sent automatically
            manager.send_data(data=None)

            # receive joystick values
            data_i_want_to_receive = manager.receive_data()

            if data_i_want_to_receive[11] and not log_data:
                log_data=True
                print("Started logging data!")

            if data_i_want_to_receive[13] and log_data:
                log_data=False
                print("Stopped logging data!")


            if log_data:
                try:
                    collect_data()
                except Exception as e:
                    print(e)
                    continue
            # use joystick values to control the excavator
            controller.update_values(data_i_want_to_receive)

            # check if the loop rate exceeds the threshold
            # safety feature
            elapsed_time = time() - start_time
            if elapsed_time < check_interval:
                data_received_count += 1
            else:
                if data_received_count < threshold:
                    controller.reset()
                    print("Threshold reset!")
                    sleep(2)
                data_received_count = 0
                start_time = time()
            # relax for a while
            # sleep(0.01)

        except (Exception, KeyboardInterrupt) as e:
            print(e)
            # this is important!
            controller.reset()
            manager.close_socket()
            rpm_manager.cleanup()
            return

if __name__ == "__main__":
    if setup():
        # Start the BNO08x test thread
        num_passes = 5  # *1000 passes
        frequency = 30  # Hz
        bno08x_thread = threading.Thread(target=bno08x_test_thread, args=(num_passes, frequency))
        #bno08x_thread.start()

        # Run the main function
        try:
            run()
        finally:
            # Wait for the BNO08x test to complete before exiting
            #bno08x_thread.join()
            manager.save_remaining_data()
            manager.close_socket()
            controller.reset()
            rpm_manager.cleanup()
