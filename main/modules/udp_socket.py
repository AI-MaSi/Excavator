import socket
import struct
import threading
import time
from typing import Optional, List, Tuple


class UDPSocket:
    """
    UDP socket with heartbeat safety mechanism.
    - Adds timestamp to each packet
    - Returns None if data is too old
    - Configurable timeout for safety
    """

    def __init__(self, local_id=0, max_age_seconds=0.5):
        self.socket = None
        self.remote_addr = None
        self.local_id = local_id
        self.num_inputs = 0
        self.num_outputs = 0
        self.max_age_seconds = max_age_seconds

        # For receiving data
        self.latest_data = None
        self.latest_timestamp = 0.0
        self.data_lock = threading.Lock()
        self.recv_thread = None
        self.running = False

        # Statistics
        self.packets_received = 0
        self.packets_expired = 0
        self.last_packet_time = 0.0

        # Pre-computed format strings (filled after handshake)
        self.send_format = None
        self.recv_format = None

    def setup(self, host, port, num_inputs, num_outputs, is_server=False):
        """Set up UDP socket with heartbeat protocol."""
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(1.0)

        if is_server:
            self.socket.bind((host, port))
            print(f"UDP Server listening on {host}:{port}")
        else:
            self.remote_addr = (host, port)
            print(f"UDP Client ready to send to {host}:{port}")

        # Format: timestamp (4 bytes) + data (N bytes)
        # Using 32-bit timestamp (milliseconds since epoch % 2^32)
        # TODO: Allow user to specify struct pack format as argument (e.g., 'h' for int16, 'f' for float)
        #       recv_size = num_outputs * sizeof(datatype)
        self.send_format = f'<I{self.num_outputs}b'  # Timestamp + signed bytes
        self.recv_format = f'<I{self.num_inputs}b'

        return True

    # ================================
    # Conversion helpers (int <-> float)
    # ================================
    @staticmethod
    def ints_to_floats(values: List[int]) -> List[float]:
        """Convert signed int8 [-128..127] to floats in [-1.0..1.0] with clamping.

        Values outside int8 range are clamped. Uses symmetric scale with 127.
        """
        floats: List[float] = []
        for v in values:
            try:
                iv = int(v)
            except Exception:
                iv = 0
            if iv < -128:
                iv = -128
            if iv > 127:
                iv = 127
            # Map to [-1, 1]; prefer 127 for symmetry
            floats.append(max(-1.0, min(1.0, iv / 127.0)))
        return floats

    @staticmethod
    def floats_to_ints(values: List[float]) -> List[int]:
        """Convert floats in [-1.0..1.0] to signed int8 [-128..127].

        Uses 127 scale with rounding and clamps to [-128, 127].
        """
        ints: List[int] = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            fv = max(-1.0, min(1.0, fv))
            iv = int(round(fv * 127.0))
            if iv < -128:
                iv = -128
            if iv > 127:
                iv = 127
            ints.append(iv)
        return ints

    def get_latest_floats(self) -> Optional[List[float]]:
        """Return latest data converted to floats [-1..1] if fresh, else None."""
        latest = self.get_latest()
        if latest is None:
            return None
        return UDPSocket.ints_to_floats(latest)

    def send_floats(self, values: List[float]) -> bool:
        """Send float values in [-1..1] (converted to signed bytes)."""
        return self.send(UDPSocket.floats_to_ints(values))

    def handshake(self, timeout=5.0):
        """Enhanced handshake that includes max_age_seconds."""
        # Pack: [id, num_outputs, num_inputs, max_age_ms] as 4 bytes + 2 bytes
        max_age_ms = int(self.max_age_seconds * 1000)
        our_info = struct.pack('<4BH',
                               self.local_id,
                               self.num_outputs,
                               self.num_inputs,
                               0,  # Reserved byte
                               max_age_ms)

        if self.remote_addr:  # Client mode
            self.socket.sendto(our_info, self.remote_addr)

            self.socket.settimeout(timeout)
            try:
                data, addr = self.socket.recvfrom(6)
                self.remote_addr = addr
            except socket.timeout:
                print("Handshake timeout!")
                return False
        else:  # Server mode
            print("Waiting for handshake...")
            self.socket.settimeout(timeout)
            try:
                data, addr = self.socket.recvfrom(6)
                self.remote_addr = addr
                self.socket.sendto(our_info, self.remote_addr)
            except socket.timeout:
                print("Handshake timeout!")
                return False

        # Verify match
        remote_id, remote_outputs, remote_inputs, _, remote_max_age_ms = struct.unpack('<4BH', data)

        if remote_inputs != self.num_outputs:
            print(f"Mismatch: They expect {remote_inputs} inputs, we send {self.num_outputs}")
            return False
        if remote_outputs != self.num_inputs:
            print(f"Mismatch: They send {remote_outputs} outputs, we expect {self.num_inputs}")
            return False

        print(f"Handshake OK with device ID {remote_id} (max_age: {remote_max_age_ms}ms)")
        self.socket.settimeout(1.0)
        return True

    def send(self, values):
        """Send values with timestamp."""
        if not self.remote_addr:
            print("No remote address set!")
            return False

        if len(values) != self.num_outputs:
            print(f"Expected {self.num_outputs} values, got {len(values)}")
            return False

        # Create timestamp (milliseconds since epoch, wrapped to 32-bit)
        timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF

        # Clamp values to 8-bit range
        clamped = [max(-128, min(127, int(v))) for v in values]

        # Pack timestamp + data and send
        data = struct.pack(self.send_format, timestamp_ms, *clamped)
        self.socket.sendto(data, self.remote_addr)
        return True

    def get_latest(self) -> Optional[List[int]]:
        """
        Get latest data only if it's fresh enough.
        Returns None if data is too old or no data received.
        """
        with self.data_lock:
            if self.latest_data is None:
                return None

            # Check if data is too old
            age = time.time() - self.latest_timestamp
            if age > self.max_age_seconds:
                self.packets_expired += 1
                #if self.packets_expired % 10 == 1:  # Log every 10th expiration
                    #print(f"WARNING: Data expired (age: {age:.3f}s > {self.max_age_seconds}s)")
                return None

            return self.latest_data.copy()

    def get_connection_stats(self) -> dict:
        """Get connection statistics for monitoring."""
        with self.data_lock:
            current_time = time.time()
            age = current_time - self.latest_timestamp if self.latest_timestamp > 0 else float('inf')
            time_since_last = current_time - self.last_packet_time if self.last_packet_time > 0 else float('inf')

            return {
                'packets_received': self.packets_received,
                'packets_expired': self.packets_expired,
                'data_age_seconds': age,
                'time_since_last_packet': time_since_last,
                'is_connected': age < self.max_age_seconds,
                'has_data': self.latest_data is not None
            }

    def start_receiving(self):
        """Start the receive thread."""
        if not self.recv_thread or not self.recv_thread.is_alive():
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()
            print("Started receive thread")

    def stop_receiving(self):
        """Stop the receive thread."""
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=2.0)
            print("Stopped receive thread")

    def _receive_loop(self):
        """Background thread to continuously receive data with timestamps."""
        expected_size = 4 + self.num_inputs  # 4 bytes timestamp + N bytes data

        while self.running:
            try:
                data, addr = self.socket.recvfrom(expected_size)

                if not self.remote_addr:
                    self.remote_addr = addr

                if len(data) == expected_size:
                    # Unpack timestamp and values
                    unpacked = struct.unpack(self.recv_format, data)
                    timestamp_ms = unpacked[0]
                    values = list(unpacked[1:])

                    # Use arrival time as packet timestamp - much simpler and more reliable
                    # than trying to handle 32-bit wraparound across network
                    arrival_time = time.time()

                    with self.data_lock:
                        self.latest_data = values
                        self.latest_timestamp = arrival_time
                        self.packets_received += 1
                        self.last_packet_time = arrival_time

                else:
                    print(f"Wrong packet size: expected {expected_size}, got {len(data)}")

            except socket.timeout:
                continue  # Normal timeout, keep trying
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")

    def close(self):
        """Clean shutdown."""
        self.stop_receiving()
        if self.socket:
            self.socket.close()
            print("Socket closed")
