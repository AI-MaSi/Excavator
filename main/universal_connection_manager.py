import socket
import struct
from datetime import datetime
import threading
import os

from time import sleep

# using these directly now. Add to __init__ if more flexibility needed
from config import *


# make proper exceptions you lazy man
class MasiSocketManager:
    def __init__(self):
        # lock needed if using threading in the future...
        self.data_save_lock = threading.Lock()
        self.checksum_bytes = struct.calcsize((endian_specifier + checksum_format))

        file_extension = ".bin" # not meant to be user changeable for now
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.filepath = os.path.join(dir_path, f"{base_filename}_{current_date}{file_extension}")

        self.local_socket = None
        self.connected_socket = None
        self.connected_addr = None
        self.recvd_id_number = None
        self.recvd_num_inputs = None
        self.recvd_num_outputs = None
        self.recv_bytes = None
        self.socket_type = None
        self.data_buffer = []

    @staticmethod
    def compute_checksum(packed_data):
        checksum = 0
        for byte in packed_data:
            checksum ^= byte
        return checksum

    @staticmethod
    def pack_data(data):
        # using fixed data_format as Mevea only accepts doubles!
        return struct.pack(endian_specifier + data_format * len(data), *data)

    def clear_file(self):
        with open(self.filepath, 'wb'):
            pass
        print("Cleared file!")

    def print_bin_file(self, num_doubles):
        format_str = endian_specifier + unix_format + data_format * num_doubles + checksum_format

        print(f"File format is: {str(format_str)}")
        sleep(2)
        data_size = struct.calcsize(format_str)

        if not os.path.exists(self.filepath):
            print(f"Error: File '{self.filepath}' does not exist.")
            # raise or create file
            return

        with open(self.filepath, 'rb') as f:
            while True:
                packed_data = f.read(data_size)
                if not packed_data:
                    break

                # Unpack the data
                data = struct.unpack(format_str, packed_data)
                print(f"Raw data: {data}")

    def setup_socket(self, socket_type):
        # add error checking
        self.socket_type = socket_type
        if not self.local_socket:
            self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.socket_type == 'server':
                self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.local_socket.bind((local_addr, port))
                self.local_socket.listen(1)
                print("Socket configured as a server!")
                print(f"\nSocket listening on {local_addr}:{port}")
                self.connected_socket, self.connected_addr = self.local_socket.accept()
                return True

            elif self.socket_type == 'client':
                # TCP_NODELAY is apparently faster
                self.local_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print("Socket configured as a client!")
                self.local_socket.connect((connect_addr, port))
                self.connected_socket = self.local_socket
                return True

            print("Check client type!")
            return False
        else:
            # socket already made, do something?
            print("socket already made!")
            return False

    def close_socket(self):
        if self.local_socket:
            self.local_socket.close()
            print("Socket closed successfully!")
        else:
            print("something wrong has happened somewhere")

    def add_checks(self, packed_data):
        # Compute checksum
        checksum = self.compute_checksum(packed_data)
        # Concatenate packed data with its checksum
        packed_values = packed_data + struct.pack((endian_specifier + checksum_format), checksum)
        # Sequence number removed
        return packed_values

    def send_data(self, data):

        # check how many inputs
        if outputs == 0:
            # no outputs, just send keep alive
            self.connected_socket.send(b'\x00')
            return True

        # pack data
        packed_data = self.pack_data(data)
        # add checks to packed data
        final_data = self.add_checks(packed_data)

        # send the data
        self.connected_socket.send(final_data)

        # save buffer data to file
        if len(self.data_buffer) >= BUFFER_SIZE:
            self.save_buffer()
        return final_data

    def receive_data(self):
        if self.recvd_num_outputs == 0:
            # sender has no outputs, just receive the keep alive
            keep_alive = self.connected_socket.recv(1)
            if len(keep_alive) == 1:
                return True
            else:
                print("Lost keep-alive signal.")
                return False

        # recv_bytes calculated in the handshake -method
        full_data = self.connected_socket.recv(self.recv_bytes)

        if len(full_data) != self.recv_bytes:
            print(f"Data received is shorter than expected: {len(full_data)} instead of {self.recv_bytes}.")
            # raise???
            return False

        # Extract and validate checksum
        received_checksum, = struct.unpack((endian_specifier + checksum_format), full_data[-self.checksum_bytes:])
        computed_checksum = self.compute_checksum(full_data[:-self.checksum_bytes])

        if received_checksum != computed_checksum:
            print("Checksum mismatch!")
            # raise???
            return False

        decoded_values = [round(struct.unpack((endian_specifier + data_format), chunk)[0], 2)
                          for chunk in (full_data[i:i + struct.calcsize((endian_specifier + data_format))]
                                        for i in range(0, len(full_data) - self.checksum_bytes,
                                                       struct.calcsize((endian_specifier + data_format))))]
        return decoded_values

    def add_data_to_buffer(self, packed_data):
        # add UNIX-timestamp and add to buffer
        with self.data_save_lock:
            current_timestamp = datetime.now().timestamp()
            microsecond_timestamp = int(current_timestamp * 1e6)

            timestamped_data = struct.pack((endian_specifier + unix_format), microsecond_timestamp) + packed_data
            self.data_buffer.append(timestamped_data)

        # check if the buffer is full
        if len(self.data_buffer) >= BUFFER_SIZE:
            self.save_buffer()

    def save_buffer(self):
        # save data from buffer to file
        with open(self.filepath, 'ab') as f:
            for value in self.data_buffer:
                f.write(value)
        print("saved data to file...")
        self.data_buffer.clear()
        return True

    def save_remaining_data(self, num_doubles):
        # If there's remaining data in the buffer, save it to file
        if self.data_buffer:
            with open(self.filepath, 'ab') as f:
                for value in self.data_buffer:
                    missing_values = num_doubles - (len(value) // 8 - 1)  # subtract 1 for the timestamp
                    value += struct.pack((endian_specifier + '{}' + data_format).format(missing_values), *([0.0] * missing_values))  # 0.0 doubles
                    f.write(value)
            self.data_buffer.clear()
            print("\nSaved remaining data.")
        else:
            print("no data in buffer to save!")

    def handshake(self):
        """
        Handshake is done so that it will work with Mevea.
        When the connection is accepted, Mevea will send 3x32bit integers
        containing [identification number], [number of outputs], [number of inputs].

        User needs to respond to this with [identification number], [number of inputs], [number of outputs]
        These values need to match!
        Please note that the response does not directly match the received data! (in / out flipped)
        """

        # handle handshake as server
        if self.socket_type == 'server':
            # receive, then send
            packed_handshake = self.connected_socket.recv(12)  # 3x4 bytes

            self.recvd_id_number, self.recvd_num_outputs, self.recvd_num_inputs = struct.unpack((endian_specifier + 3*handshake_format), packed_handshake)
            # send back the response handshake (in / out flipped)
            response = struct.pack((endian_specifier + 3 * handshake_format), identification_number, inputs, outputs)
            self.connected_socket.send(response)    # connected or local???

        # handle handshake as client
        elif self.socket_type == 'client':
            # send, then receive
            response = struct.pack((endian_specifier + 3 * handshake_format), identification_number, outputs, inputs)
            self.local_socket.send(response)

            # receive handshake
            packed_handshake = self.local_socket.recv(12)  # 3x4 bytes
            self.recvd_id_number, self.recvd_num_inputs, self.recvd_num_outputs = struct.unpack((endian_specifier + 3 * handshake_format), packed_handshake)

            # check the handshake
            # identification number cannot be the same
            if self.recvd_id_number == identification_number:
                print("cannot connect two of the same!")
                # raise
            # in / out needs to match
            if self.recvd_num_inputs != outputs or self.recvd_num_outputs != inputs:
                print("input / output mismatch")
                # raise

        # let the user know the handshake results
        if self.recvd_id_number == 0:
            print(f"Handshake received from Excavator with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")
        elif self.recvd_id_number == 1:
            print(f"Handshake received from Mevea with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")
        elif self.recvd_id_number == 2:
            print(f"Handshake received from Motion Platform with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")
        else:
            print(f"Unknown handshake received from {self.connected_addr} with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")
            return False

        # calculate how many bytes we are going to receive
        self.recv_bytes = struct.calcsize((endian_specifier + data_format)) * self.recvd_num_outputs + self.checksum_bytes
        return True
