import universal_socket_manager
import masi_driver
import imu_sensor_manager
from time import time, sleep

simulation_mode = False
log_data = True
threshold = 5  # atleast this many messages required every second

# init socket
manager = universal_socket_manager.MasiSocketManager()
# exceptions soon...

# init servo controller
controller = masi_driver.ExcavatorController(simulation_mode=simulation_mode)

# init IMU's
if log_data:
    try:
        imu_manager = imu_sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)
    except (imu_sensor_manager.BNO08xInitializationError, imu_sensor_manager.ISM330InitializationError) as e:
        imu_manager = None
        raise e

def setup():
    if log_data:
        manager.clear_file()

    setup_result = manager.setup_socket(socket_type='client')
    if not setup_result:
        print("could not set up socket!")
        return False

    handshake_result = manager.handshake()
    if not handshake_result:
        print("could not make handshake!")
    return handshake_result

def run():
    check_interval = 1.0
    data_received_count = 0
    start_time = time()

    while True:
        try:
            # I only want to send handshake
            # outputs are set to 0, handshake is sent automatically
            manager.send_data(data=None)

            # receive joystick values
            data_i_want_to_receive = manager.receive_data()

            # use joystick values to control the excavator
            controller.update_values(data_i_want_to_receive)

            if log_data:
                # get values from the sensors
                data_i_want_to_save = imu_manager.read_all()  # BNO has problems!

                packed_data = manager.pack_data(data_i_want_to_save)

                # Add an empty checksum byte to the packed_data
                # Now data will be in the same format as the packed MotionPlatform joystick data
                packed_data += b'\x00'

                manager.add_data_to_buffer(packed_data)

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
            return

if __name__ == "__main__":
    if setup():
        run()
    else:
        manager.close_socket()
        controller.reset()
