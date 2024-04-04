"""
WOW huge update
-keep alive -system removed
-ability to use UDP or TCP, TCP side not tested yet tho
-own thread to monitor incoming control signals

-Back and forth -communication has been fucked because I don't understand UDP well enough haha

"""

import socket
import struct
from datetime import datetime
import threading
import os

from time import sleep

from config import *  # prööt, when more time create e.g. json-config

buffer_size = 100
extra_args = 3


class MasiSocketManager:
    def __init__(self, identification_number, inputs, outputs):

        # lock needed if using threading in the future...
        self.data_save_lock = threading.Lock()
        self.latest_data_lock = threading.Lock()

        self.checksum_bytes = struct.calcsize((endian_specifier + checksum_format))

        file_extension = ".bin"  # not meant to be user changeable for now
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.filepath = os.path.join(dir_path, f"{base_filename}_{current_date}{file_extension}")

        self.local_id_number = identification_number
        self.local_num_inputs = inputs
        self.local_num_outputs = outputs
        self.data_format = data_format
        self.data_buffer = []

        self.local_socket = None
        self.connected_socket = None
        self.local_addr = None
        self.connected_addr = None
        self.recvd_id_number = None
        self.recvd_num_inputs = None
        self.recvd_num_outputs = None
        self.recv_bytes = None
        self.socket_type = None
        self.network_protocol = None
        self.running = None
        self.data_reception_thread = None
        self.latest_recvd = None

    @staticmethod
    def compute_checksum(packed_data):
        checksum = 0
        for byte in packed_data:
            checksum ^= byte
        return checksum

    @staticmethod
    def prepare_extra_args(kwargs):
        # Prepare extra arguments for sending, ensuring exactly num_args are present
        extra_args_to_send = list(kwargs.values())[:extra_args]
        extra_args_to_send.extend([0] * (extra_args - len(extra_args_to_send)))  # Fill no data with 0
        return extra_args_to_send

    def pack_data(self, data):
        # ghetto packing using 1 byte integers

        int_data = []  # List to store converted integer values
        for value in data:
            clamped_value = max(-1.0, min(1.0, value))

            int_value = int(clamped_value * 100)  # haha ghetto convert
            int_data.append(int_value)

        packed_data = struct.pack(endian_specifier + self.data_format * len(int_data), *int_data)

        return packed_data

    def clear_file(self):
        with open(self.filepath, 'wb'):
            pass
        print("Cleared file!")

    def print_bin_file(self, num_doubles):
        format_str = endian_specifier + unix_format + self.data_format * num_doubles + checksum_format

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

    def setup_socket(self, addr, port, socket_type):
        if socket_type == 'client':
            self.setup_socket_client(addr, port)

        elif socket_type == 'server':
            self.setup_socket_server(addr, port)
        else:
            print("Check socket type!")
            return False
        return True

    def setup_socket_client(self, addr, port):
        if not self.local_socket:
            self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set up as client. TCP
            self.local_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Socket configured as a TCP-client!")
            self.local_socket.connect((addr, port))

            # Remember for later use
            self.socket_type = 'client'
            self.connected_addr = (addr, port)
            self.connected_socket = self.local_socket
            self.network_protocol = 'tcp'

        else:
            # socket already made, do something?
            print("socket already made!")

    def setup_socket_server(self, addr, port):
        if not self.local_socket:
            self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Set up as server. TCP
            self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.local_socket.bind((addr, port))
            print(f"Socket configured as a TCP-server! Listening on {addr}:{port}")
            self.local_socket.listen(1)
            self.connected_socket, self.connected_addr = self.local_socket.accept()

            self.socket_type = 'server'
            self.local_addr = (addr, port)
            self.network_protocol = 'tcp'

        else:
            # socket already made, do something?
            print("socket already made!")

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

        # pack data
        packed_data = self.pack_data(data)
        # add checks to packed data
        final_data = self.add_checks(packed_data)

        # send the data
        # self.connected_socket.send(final_data) TCP
        self.local_socket.sendto(final_data, self.connected_addr)

        # save buffer data to file
        if len(self.data_buffer) >= buffer_size:
            self.save_buffer()
        return final_data

    def start_data_recv_thread(self):
        if self.data_reception_thread is None or not self.data_reception_thread.is_alive():
            self.running = True
            self.data_reception_thread = threading.Thread(target=self.continuous_data_receiver, daemon=True)
            self.data_reception_thread.start()
            print("Started data receiving thread!")
        else:
            print("Data reception is already running.")

    def stop_data_recv_thread(self):
        self.running = False
        if self.data_reception_thread is not None:
            self.data_reception_thread.join()
            self.data_reception_thread = None

    def continuous_data_receiver(self, ultimate_relaxing_time=0.002):
        while self.running:
            try:
                data = self.receive_data()
                if data:
                    with self.latest_data_lock:
                        self.latest_recvd = data
                # sleep(ultimate_relaxing_time)  # Consistent delay for both data and no-data scenarios
            except Exception as e:
                print(f"Error receiving data: {e}")
                # sleep(ultimate_relaxing_time)  # Relax time in case of exceptions as well

    def receive_data(self):
        try:
            if self.network_protocol == 'tcp':
                full_data = self.connected_socket.recv(self.recv_bytes)
            else:
                full_data, addr = self.local_socket.recvfrom(self.recv_bytes)

            if not full_data or len(full_data) != self.recv_bytes:
                # Handles both keep-alive (len=1) and incomplete/missing data
                print("No new data or incomplete data received.")
                return None  # Unified return for no or incorrect data

            return full_data  # Returns the full, received data for further processing
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def get_latest_received(self):
        with self.latest_data_lock:
            if self.latest_recvd is None:
                return None  # Silently handle the no-data-yet case

            try:
                full_data = self.latest_recvd
                received_checksum, = struct.unpack((endian_specifier + checksum_format),
                                                   full_data[-self.checksum_bytes:])
                computed_checksum = self.compute_checksum(full_data[:-self.checksum_bytes])
                if received_checksum != computed_checksum:
                    print("Checksum mismatch!")
                    return None

                decoded_values = [(struct.unpack(endian_specifier + self.data_format, chunk)[0] / 100.0)
                                  for chunk in (full_data[i:i + struct.calcsize(endian_specifier + self.data_format)]
                                                for i in range(0, len(full_data) - self.checksum_bytes,
                                                               struct.calcsize(endian_specifier + self.data_format)))]
                self.latest_recvd = None  # Reset after processing
                return decoded_values
            except Exception as e:
                print(f"Error processing the latest received data: {e}")
                return None

    """
    def receive_data(self):
        if self.recvd_num_outputs == 0:

            # sender has no outputs, just receive the keep alive
            if self.network_protocol == 'tcp':
                keep_alive = self.connected_socket.recv(1)
            else:
                keep_alive, addr = self.local_socket.recvfrom(1)

            if len(keep_alive) == 1:
                return True
            else:
                print("Lost keep-alive signal.")
                return False

        # recv_bytes calculated in the handshake -method
        if self.network_protocol == 'tcp':
            full_data = self.connected_socket.recv(self.recv_bytes)
        else:
            full_data, addr = self.local_socket.recvfrom(self.recv_bytes)

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

        decoded_values = [(struct.unpack(endian_specifier + self.data_format, chunk)[0] / 100.0)
                          for chunk in (full_data[i:i + struct.calcsize(endian_specifier + self.data_format)]
                                        for i in range(0, len(full_data) - self.checksum_bytes,
                                                       struct.calcsize(endian_specifier + self.data_format)))]
        return decoded_values

    """

    def add_data_to_buffer(self, packed_data):
        # add UNIX-timestamp and add to buffer
        with self.data_save_lock:
            current_timestamp = datetime.now().timestamp()
            microsecond_timestamp = int(current_timestamp * 1e6)

            timestamped_data = struct.pack((endian_specifier + unix_format), microsecond_timestamp) + packed_data
            self.data_buffer.append(timestamped_data)

        # check if the buffer is full
        if len(self.data_buffer) >= buffer_size:
            self.save_buffer()

    def save_buffer(self):
        # save data from buffer to file
        with open(self.filepath, 'ab') as f:
            for value in self.data_buffer:
                f.write(value)
        print("saved data to file...")
        self.data_buffer.clear()
        return True

    def save_remaining_data(self, num_values):
        # If there's remaining data in the buffer, save it to file
        if self.data_buffer:
            with open(self.filepath, 'ab') as f:
                for value in self.data_buffer:
                    missing_values = num_values - (len(value) // 8 - 1)  # subtract 1 for the timestamp
                    value += struct.pack((endian_specifier + '{}' + self.data_format).format(missing_values),
                                         *([0.0] * missing_values))  # 0.0 doubles
                    f.write(value)
            self.data_buffer.clear()
            print("\nSaved remaining data.")
        else:
            print("no data in buffer to save!")

    def receive_extra_args(self, num_args):
        # Receive extra arguments from the TCP connection
        recvd_extra_args = []
        for _ in range(num_args):
            packed_arg = self.connected_socket.recv(struct.calcsize(handshake_format))
            arg, = struct.unpack(endian_specifier + handshake_format, packed_arg)
            recvd_extra_args.append(arg)
        return recvd_extra_args

    def identify(self, device_name, recvd_extra_args):
        # let the user know who is who
        # here you could also set some settings for different connections
        if device_name == "Undefined":
            print(
                f"Undefined handshake received from {self.connected_addr} with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")
        elif device_name == "Mevea":
            print(
                f"Handshake confirmed with Mevea device at {self.connected_addr} with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")

            self.data_format = 'd'  # 8 byte doubles for Mevea
        else:
            print(
                f"Handshake received from {device_name} with {self.recvd_num_inputs} inputs and {self.recvd_num_outputs} outputs.")

        print(f"Received extra arguments: {recvd_extra_args}")

    def handshake(self, **kwargs):
        """
        Handshake is done so that it will work with Mevea.
        When the connection is accepted, Mevea will send 3x32bit integers
        containing [identification number], [number of outputs], [number of inputs].

        User needs to respond to this with [identification number], [number of inputs], [number of outputs]
        These values need to match!
        Please note that the response does not directly match the received data! (in / out flipped)


        If not communicating with Mevea, you are able to send three extra arguments during handshake.
        These can be used to e.g. inform the receiving end about the sending rate or pump multiplier
         """

        recvd_extra_args = None

        # Additional arguments for non-Mevea connections
        for key, value in kwargs.items():
            print(f"Additional argument added: {key} with value: {value}")

        # handle handshake as server
        if self.socket_type == 'server':
            packed_handshake = self.connected_socket.recv(12)  # 3x4 bytes
            self.recvd_id_number, self.recvd_num_outputs, self.recvd_num_inputs = struct.unpack(
                endian_specifier + handshake_format * 3, packed_handshake)
            device_name = id_numbers.get(self.recvd_id_number, "Undefined")

            inputs_match = self.recvd_num_inputs == self.local_num_outputs
            outputs_match = self.recvd_num_outputs == self.local_num_inputs

            if not (inputs_match and outputs_match):
                # raise?
                print(f"Error: Mismatch in expected inputs/outputs with {device_name}.")
                return False

            if device_name != "Mevea":
                recvd_extra_args = self.receive_extra_args(extra_args)

            extra_args_to_send = self.prepare_extra_args(kwargs)
            response_format = endian_specifier + handshake_format * (3 + extra_args)
            response_values = [self.local_id_number, self.local_num_inputs, self.local_num_outputs] + extra_args_to_send
            response = struct.pack(response_format, *response_values)
            self.connected_socket.send(response)

        # handle handshake as client
        elif self.socket_type == 'client':
            extra_args_to_send = self.prepare_extra_args(kwargs)
            response = struct.pack(
                endian_specifier + handshake_format * (3 + extra_args),
                self.local_id_number, self.local_num_outputs, self.local_num_inputs, *extra_args_to_send)
            self.local_socket.send(response)

            packed_handshake = self.local_socket.recv(12)
            self.recvd_id_number, self.recvd_num_inputs, self.recvd_num_outputs = struct.unpack(
                endian_specifier + handshake_format * 3, packed_handshake)
            device_name = id_numbers.get(self.recvd_id_number, "Undefined")

            if device_name != "Mevea":
                recvd_extra_args = self.receive_extra_args(extra_args)

        else:  # not correct socket-type
            print("wrong socket type!")
            return False

        # calculate how many bytes we are going to receive (in drive loop)
        self.recv_bytes = struct.calcsize(
            (endian_specifier + self.data_format)) * self.recvd_num_outputs + self.checksum_bytes

        print(f"Received data should be: {self.recv_bytes} bytes")
        sleep(1)

        self.identify(device_name, recvd_extra_args)
        print(f"Handshake done!")
        print("\n------------------------------------------")
        return True, recvd_extra_args

    def tcp_to_udp(self, socket_buffer_size=64):
        print("Reconfiguring to UDP...")
        # Close existing TCP connection if it's a connected socket distinct from the listening socket
        if hasattr(self, 'connected_socket') and self.connected_socket:
            self.connected_socket.close()
            print("Closed existing TCP connected socket.")
            self.connected_socket = None  # Reset connected socket to avoid reuse

        # Close the listening socket if it exists
        if self.local_socket:
            self.local_socket.close()
            print("Closed existing TCP listening socket.")

        # Create a new socket for UDP
        self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_buffer_size)
        # self.local_socket.setblocking(False)

        # Set up as a UDP server
        if self.socket_type == 'server':
            self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.local_socket.bind(self.local_addr)
            print(f"Socket reconfigured as a UDP-server! Listening on {self.local_addr}...")
            self.network_protocol = 'udp'
            return True

        # Set up as a UDP client
        elif self.socket_type == 'client':
            # For UDP client, no need to connect
            print("Socket reconfigured as a UDP-client!")
            self.network_protocol = 'udp'
            return True
