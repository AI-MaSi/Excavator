# this script handles the communication on the excavator

import socket
import struct
from time import sleep
import datetime
import threading
# import os.path


from config import host, port, inputs, outputs, multiplexer_channels
import masi_driver
import imu_sensor_manager

controller = masi_driver.ExcavatorController(simulation_mode=False)
imu_manager = imu_sensor_manager.IMUSensorManager(multiplexer_channels, simulation_mode=True)

delay = 0.00
data_save_lock = threading.Lock()
data_buffer = []
BUFFER_SIZE = 10
file_name = "../logging/masi_data.bin"


def compute_checksum(data):
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def receive_data(client_socket, num_inputs, data_type='<d'):
    sequence_bytes = struct.calcsize('<I')  # bytes for sequence number
    checksum_bytes = struct.calcsize('<B')  # bytes for checksum

    recv_bytes = sequence_bytes + struct.calcsize(data_type) * num_inputs + checksum_bytes
    full_data = client_socket.recv(recv_bytes)

    if len(full_data) != recv_bytes:
        print(f"Data received is shorter than expected: {len(full_data)} instead of {recv_bytes}.")
        return None

    # Extract and validate sequence number
    sequence_received, = struct.unpack('<I', full_data[:sequence_bytes])

    # sequence check here

    # print(f"\nReceived sequence number: {sequence_received}")

    # Extract and validate checksum
    received_checksum, = struct.unpack('<B', full_data[-checksum_bytes:])
    computed_checksum = compute_checksum(full_data[:-checksum_bytes])

    if received_checksum != computed_checksum:
        print("Checksum mismatch!")
        return None

    decoded_values = [round(struct.unpack(data_type, chunk)[0], 2)
                      for chunk in (full_data[sequence_bytes + i:sequence_bytes + i + struct.calcsize(data_type)]
                                    for i in range(0, len(full_data) - sequence_bytes - checksum_bytes,
                                                   struct.calcsize(data_type)))]
    return decoded_values


def send_keep_alive(client_socket):
    try:
        client_socket.send(b'\x00')
        return True
    except Exception as e:
        print(f"Failed to send keep-alive signal: {e}")
        return False


def request_sensor_data():
    return imu_manager.read_all_and_pack()


def save_data_with_timestamp(data):
    global data_buffer

    try:
        with data_save_lock:
            current_timestamp = datetime.datetime.now().timestamp()  # get UNIX timestamp including fractional seconds
            # Convert the float timestamp to a format that retains the microsecond precision
            # (e.g. multiply by 1e6 and cast to an integer)
            microsecond_timestamp = int(current_timestamp * 1e6)

            # append the timestamp to the data
            timestamped_data = struct.pack('<Q',
                                           microsecond_timestamp) + data  # 'Q' denotes an unsigned long long for the timestamp
            data_buffer.append(timestamped_data)

            if len(data_buffer) >= BUFFER_SIZE:
                with open(file_name, 'ab') as f:
                    for value in data_buffer:
                        f.write(value)
                print("saved data to file...")
                data_buffer.clear()
                return True
    except Exception as e:
        print(f"Error when saving data: {e}")
        return False


def save_remaining_data():
    # If there's remaining data in the buffer, save it to file
    if data_buffer:
        with open(file_name, 'ab') as f:
            for value in data_buffer:
                missing_values = 20 - (len(value) // 8 - 1)  # subtract 1 for the timestamp
                # add 0.0 doubles for missing values
                value += struct.pack('<{}d'.format(missing_values), *([0.0] * missing_values))
                f.write(value)
        data_buffer.clear()
        print("Saved remaining data.")


def handshake(client_socket, inputs, outputs):
    try:
        handshake_data = struct.pack('<3i', 0, outputs, inputs)  # send 0 to differentiate from Mevea
        client_socket.send(handshake_data)
        print(f"Handshake sent to the server with {inputs} input(s) and {outputs} output(s).")

        # no need to use the data
        data = client_socket.recv(12)  # 3x4 bytes
        # decoded_data = struct.unpack('<3i', data)
        # here could be extra check to see if the handshake matches

        print(f"Handshake done!")
        return True

    except Exception as e:
        print(f"\nHandshake Error: {e}")
        return False


def client_function(server_address, inputs, outputs):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(server_address)

        # TCP_NODELAY is apparently faster
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connected to server: {server_address}")

        success = handshake(client_socket, inputs, outputs)
        if not success:
            return

        while True:
            controller_data = receive_data(client_socket, inputs)

            # send controller data here to the driver
            controller.update_values(controller_data)

            if controller_data is None:
                break

            # print(request_sensor_data())
            # sensor_data = request_data()
            # save_data_with_timestamp(sensor_data)

            # Send keep-alive after receiving data
            if not send_keep_alive(client_socket):
                break

    except Exception as e:
        print(f"Error in client function: {e}")

    finally:
        # save_remaining_data()
        controller.reset()
        client_socket.close()


def main():
    controller.reset()
    server_address = (host, port)
    print(f"\nConnecting to: {server_address}...")

    while True:
        client_function(server_address, inputs, outputs)
        print(f"\nReconnecting to: {server_address}")
        sleep(5)


if __name__ == "__main__":
    main()
