import imu_sensor_manager
import universal_socket_manager
from time import sleep

simulation_mode = True

# init joysticks
try:
    imu_manager = imu_sensor_manager.IMUSensorManager(simulation_mode=simulation_mode)
except (imu_sensor_manager.BNO08xInitializationError, imu_sensor_manager.ISM330InitializationError) as e:
    imu_manager = None
    raise e

try:
    # init socket
    manager = universal_socket_manager.MasiSocketManager()
except Exception as e:
    print("make exceptions!")


def setup_example():
    # these could be set in GUI
    # manager.setup_socket(addr, port, client_type='server')

    setup_result = manager.setup_socket(client_type='client')
    if not setup_result:
        print("could not setup socket!")
        sleep(3)
        return
    print("starting handshake...")
    sleep(2)
    handshake_result = manager.handshake()
    sleep(5)
    if not handshake_result:
        print("could not setup socket!")
        sleep(3)
        return


def run_example():
    # just set flags to True
    manager.set_start_flag(True)
    manager.set_record_flag(True)

    while True:
        data_i_want_to_send = ''
        # print(imu_manager.read_bno08())
        manager.send_data(data_i_want_to_send)

        data_i_want_to_receive = manager.receive_data()
        print(f"Received: {data_i_want_to_receive}")

        sleep(1)


if __name__ == "__main__":
    setup_example()
    run_example()