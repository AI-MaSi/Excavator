"""
This is a messy test script to drive around with the excavator.
The script will save sensor data to .bin -file
"""

import universal_socket_manager
import masi_driver
import sensor_manager
from time import sleep

# init socket
manager = universal_socket_manager.MasiSocketManager()
# exceptions soon...

# init servo controller
controller = masi_driver.ExcavatorController(simulation_mode=False, tracks_disabled=True, pump_variable=False)

# init IMU's
imu_manager = sensor_manager.IMUSensorManager(simulation_mode=False)

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


def collect_data():
    # get values from the sensors
    data_i_want_to_save = imu_manager.read_all()  # BNO has problems!

    # get pressure values
    pressure_data = pressure_manager.read_pressure()
    data_i_want_to_save += pressure_data

    # get the pump rpm
    rpm_data = rpm_manager.get_rpm()
    data_i_want_to_save.append(rpm_data)
    packed_data = manager.pack_data(data_i_want_to_save)

    # Add an empty checksum byte to the packed_data
    # Now data will be in the same format as the packed MotionPlatform joystick data
    packed_data += b'\x00'
    manager.add_data_to_buffer(packed_data)


def run():
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


if __name__ == "__main__":
    if setup():
        try:
            run()
        finally:
            # Cleanup
            # save data left over
            manager.save_remaining_data(num_doubles=31)
            # close socket connections
            manager.close_socket()
            # reset servos and stop the pump
            controller.reset()
            # clean up rpm-GPIO pins
            rpm_manager.cleanup()
            sleep(2)
            # Misc. Print the saved values.
            manager.print_bin_file(num_doubles=31)
