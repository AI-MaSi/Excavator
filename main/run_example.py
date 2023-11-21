import imu_sensor_manager
import universal_socket_manager
import masi_driver
from time import sleep

simulation_mode = True

# init joysticks
try:
    imu_manager = imu_sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)
except (imu_sensor_manager.BNO08xInitializationError, imu_sensor_manager.ISM330InitializationError) as e:
    imu_manager = None
    raise e

# init socket
manager = universal_socket_manager.MasiSocketManager()
# exceptions soon...

# init servo controller
controller = masi_driver.ExcavatorController(simulation_mode=simulation_mode)



def setup_example():
    # these could be set in eg. GUI
    # manager.setup_socket(addr, port, client_type='server')

    setup_result = manager.setup_socket(socket_type='client')
    if not setup_result:
        print("could not setup socket!")
        sleep(3)
        return

    handshake_result = manager.handshake()
    sleep(5)
    if not handshake_result:
        print("could not make handshake!")
        sleep(3)
        return


def run_example():
    while True:
        try:
            # I only want to send handshake
            # outputs are set to 0, handshake is sent automatically
            manager.send_data(data=None)

            # receive joystick values
            data_i_want_to_receive = manager.receive_data()

            # use joystick values to control the excavator
            controller.update_values(data_i_want_to_receive)

            # get values from the sensors
            data_i_want_to_save = imu_manager.read_all()
            packed_data = manager.pack_data(data_i_want_to_save)
            manager.add_data_to_buffer(packed_data)



        except KeyboardInterrupt:
            manager.close_socket()
            return

        except Exception:
            # this is important!
            controller.reset()
            return

if __name__ == "__main__":
    setup_example()
    run_example()
