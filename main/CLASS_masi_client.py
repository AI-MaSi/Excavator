import socket
import struct
import datetime
from time import sleep
import threading
from config import host, port, inputs, outputs, multiplexer_channels
import masi_driver
import imu_sensor_manager

class ExcavatorClient:
    def __init__(self, server_address, inputs, outputs):
        self.server_address = server_address
        self.inputs = inputs
        self.outputs = outputs
        self.client_socket = None
        self.data_save_lock = threading.Lock()
        self.data_buffer = []
        self.expected_sequence_number = 0
        self.file_name = "../logging/masi_data.bin"
        self.delay = 0.00
        self.BUFFER_SIZE = 10
        """
        MAKE THIS!!!!!!
        try:
            imu_manager = IMUSensorManager()
            # ... other operations ...
        except (ISM330InitializationError, BNO08xInitializationError) as e:
            print(f"Initialization error: {e}")
            # Handle initialization error
        except (ISM330ReadError, BNO08xReadError) as e:
            print(f"Read error: {e}")
            # Handle read error
        """

        try:
            self.controller = masi_driver.ExcavatorController(simulation_mode=True)
        except Exception as e:
            pass

        self.imu_manager = None
        # self.other_sensors = None
        # none not init, false error true ok etc....
        self.init_sensors()


    def init_sensors(self):
        # add try except and returning
        # add other sensors
        self.imu_manager = imu_sensor_manager.IMUSensorManager(multiplexer_channels, simulation_mode=True)

    @staticmethod
    def compute_checksum(data):
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum

    # ... other methods like receive_data, send_keep_alive, request_sensor_data ...

    def handshake(self):
        try:
            handshake_data = struct.pack('<3i', 0, self.outputs, self.inputs)
            self.client_socket.send(handshake_data)
            # ... rest of the handshake logic ...
            return True
        except Exception as e:
            print(f"\nHandshake Error: {e}")
            return False

    def client_function(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect(self.server_address)
            # ... rest of the client_function logic ...
        except Exception as e:
            print(f"Error in client function: {e}")
        finally:
            # ... cleanup ...
            self.client_socket.close()

    def run(self):
        self.controller.reset()
        print(f"\nConnecting to: {self.server_address}...")
        while True:
            self.client_function()
            print(f"\nReconnecting to: {self.server_address}")
            sleep(5)

if __name__ == "__main__":
    client = ExcavatorClient((host, port), inputs, outputs)
    client.run()
