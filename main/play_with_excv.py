import universal_socket_manager
import masi_driver
#import imu_sensor_manager


import sensor_manager

from time import time, sleep

simulation_mode = False
threshold = 5  # atleast this many messages required every second

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

    # get the pump rpm
    rpm_data = rpm_manager.get_rpm()
    if rpm_data is not None:
        # data_i_want_to_save += rpm_data
        data_i_want_to_save.append(rpm_data)

    packed_data = manager.pack_data(data_i_want_to_save)

    # Add an empty checksum byte to the packed_data
    # Now data will be in the same format as the packed MotionPlatform joystick data
    packed_data += b'\x00'

    manager.add_data_to_buffer(packed_data)


def run():
    check_interval = 1.0
    data_received_count = 0
    start_time = time()

    log_data = False

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
                collect_data()

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

        except Exception as e:
            print(e)
            # this is important!
            controller.reset()
            manager.close_socket()
            rpm_manager.cleanup()
            return

if __name__ == "__main__":
    if setup():
        run()
    else:
        manager.close_socket()
        controller.reset()
        rpm_manager.cleanup()
