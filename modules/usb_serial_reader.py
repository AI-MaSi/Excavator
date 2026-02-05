import serial
import time
import logging
import serial.tools.list_ports
import struct
import math


class USBSerialReader:
    """Read IMU quaternion data from Pico over USB serial (binary protocol)."""

    NUM_SENSORS = 3
    FLOATS_PER_SENSOR = 7  # w, x, y, z, gx, gy, gz

    # Binary protocol message types (version=2 control frames)
    MSG_TYPE_CFG_OK = 0x01
    MSG_TYPE_CFG_WAIT = 0x02
    MSG_TYPE_ZERO_ACK = 0x03
    MSG_TYPE_ERROR = 0x04

    def __init__(self, baud_rate=115200, timeout=1.0, simulation_mode=False,
                 log_level: str = "INFO", port: str = None, debug: bool = False,
                 verify_checksum: bool = True):
        self.logger = logging.getLogger(f"{__name__}.USBSerialReader")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.port = port
        self.simulation_mode = simulation_mode
        self.debug = debug
        self.verify_checksum = verify_checksum
        self.sim_time = 0.0
        self.last_timestamp_us = None
        self._bin_buf = bytearray()
        self._text_buf = bytearray()
        self._last_config = None
        self._checksum_failures = 0
        self._header_failures = 0

        if not simulation_mode:
            self.connect()
        else:
            self.logger.info("Simulation mode: generating synthetic IMU data")

    def find_pico_port(self):
        """Try to find a likely XIAO RP2040 / Pico port automatically."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.logger.error("No serial ports found.")
            return None

        keywords = ("XIAO", "RP2040", "Pico", "USB Serial", "CDC", "ACM")
        vidpid_markers = ("VID:PID=2E8A:", "VID:PID=2886:")

        for port in ports:
            desc = port.description or ""
            hwid = port.hwid or ""
            if any(k in desc for k in keywords) or any(m in hwid for m in vidpid_markers):
                self.logger.info(f"Found device on port: {port.device}")
                return port.device

        self.logger.warning(f"No descriptor match; defaulting to {ports[0].device}")
        return ports[0].device

    def connect(self):
        """Connect to the serial port."""
        if not self.port:
            self.port = self.find_pico_port()
        if not self.port:
            raise serial.SerialException("No serial port found")

        self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
        time.sleep(0.1)

    def send_config(
        self,
        sample_rate=200,
        gyro_dps=500,
        gain=0.5,
        accel_rejection=10.0,
        recovery_s=1.0,
        offset_s=1.0,
    ):
        """
        Send configuration to Pico.

        Args:
            sample_rate: Sample rate in Hz (10-1000)
            gyro_dps: Gyro full-scale range (125/250/500/1000/2000)
            gain: AHRS filter gain (higher = faster response, more noise)
            accel_rejection: Reject accel when exceeding this deviation (degrees)
            recovery_s: Time before re-enabling accel after rejection (seconds)
            offset_s: Gyro offset tracking timeout (seconds)
        """
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot send config")
            return False

        config = (
            f"SR={sample_rate}|"
            f"GYRO_DPS={gyro_dps}|"
            f"GAIN={gain}|"
            f"ACC_REJ={accel_rejection}|"
            f"RECOV_S={recovery_s}|"
            f"OFFSET_S={offset_s}|\n"
        )
        self._last_config = config
        self.ser.write(config.encode("utf-8"))
        self.logger.info(f"Sent config: {config.strip()}")
        return True

    def wait_for_cfg_ok(self, timeout_s=5.0, resend_s=0.5):
        """Wait for binary CFG_OK from Pico, resending config periodically."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot wait for CFG_OK")
            return False

        start = time.time()
        last_send = 0.0

        while True:
            now = time.time()
            if timeout_s is not None and (now - start) >= timeout_s:
                return False

            if resend_s is not None and (now - last_send) >= resend_s:
                # Resend last config if available
                if self._last_config:
                    self.ser.write(self._last_config.encode("utf-8"))
                last_send = now

            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                if self.debug and chunk:
                    self.logger.info("RX(wait): %r", chunk)
                self._bin_buf.extend(chunk)

                # Look for binary CFG_OK: [0xAA 0x55 0x02 0x01 checksum16]
                found_cfg_ok = False
                idx = 0
                while idx <= len(self._bin_buf) - 6:
                    if self._bin_buf[idx] == 0xAA and self._bin_buf[idx + 1] == 0x55:
                        version = self._bin_buf[idx + 2]
                        if version == 2:  # Control message
                            msg_type = self._bin_buf[idx + 3]
                            expected_cs = struct.unpack_from("<H", self._bin_buf, idx + 4)[0]
                            calc_cs = (version + msg_type) & 0xFFFF
                            if calc_cs == expected_cs and msg_type == self.MSG_TYPE_CFG_OK:
                                found_cfg_ok = True
                                # Preserve any data after this frame
                                self._bin_buf = bytearray(self._bin_buf[idx + 6:])
                                break
                            # Skip this frame (bad checksum or different msg type)
                            idx += 6
                        else:
                            # Data frame - skip past sync for now
                            idx += 1
                    else:
                        idx += 1

                if found_cfg_ok:
                    return True

                # Keep buffer bounded
                if len(self._bin_buf) > 4096:
                    del self._bin_buf[:-1024]

            time.sleep(0.01)

    def send_zero(self):
        """Request re-zero (sets current pitch as zero reference)."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot send ZERO")
            return False

        self.ser.write(b"CMD=ZERO\n")
        self.logger.info("Sent ZERO command")
        return True

    def generate_simulation_data(self):
        """Generate synthetic IMU data for testing."""
        self.sim_time += 0.0083  # ~120Hz

        # Simulate rotation from 0 to 45 degrees over 5 seconds
        angle = min(45.0, (self.sim_time / 5.0) * 45.0) * math.pi / 180.0
        angular_velocity = (45.0 * math.pi / 180.0) / 5.0 if self.sim_time < 5.0 else 0.0

        data = []
        for i in range(self.NUM_SENSORS):
            # Each IMU at slightly different angle
            offset = i * 30.0 * math.pi / 180.0
            a = angle + offset
            w = math.cos(a / 2)
            y = math.sin(a / 2)
            data.append([w, 0.0, y, 0.0, 0.0, angular_velocity * 180.0 / math.pi, 0.0])

        self.last_timestamp_us = int(self.sim_time * 1e6)
        return data

    def _read_binary_frame(self):
        """Parse binary IMU frame from serial buffer."""
        if not self.ser or not self.ser.is_open:
            return None

        # Read available data into buffer
        if self.ser.in_waiting > 0:
            self._bin_buf.extend(self.ser.read(self.ser.in_waiting))

        # Frame: 0xAA 0x55 | ver(1) | count(1) | ts_us(4) | payload | checksum(2)
        while True:
            if len(self._bin_buf) < 4:
                return None

            # Find sync bytes
            sync = self._bin_buf.find(b"\xAA\x55")
            if sync < 0:
                # Keep a trailing 0xAA in case sync splits across reads
                if self._bin_buf.endswith(b"\xAA"):
                    self._bin_buf[:] = b"\xAA"
                else:
                    self._bin_buf.clear()
                return None
            if sync > 0:
                del self._bin_buf[:sync]
                if len(self._bin_buf) < 4:
                    return None

            # Parse header
            if len(self._bin_buf) < 8:
                return None

            version = self._bin_buf[2]

            # Handle control frames (version=2) - skip them silently
            if version == 2:
                if len(self._bin_buf) < 6:
                    return None
                # Control frame is always 6 bytes, skip it
                del self._bin_buf[:6]
                continue

            sensor_count = self._bin_buf[3]

            if version != 1 or sensor_count != self.NUM_SENSORS:
                # Drop one byte and rescan for sync
                self._header_failures += 1
                del self._bin_buf[0:1]
                continue

            # Calculate frame length
            payload_len = 4 + sensor_count * self.FLOATS_PER_SENSOR * 4
            frame_len = 2 + 1 + 1 + payload_len + 2

            if len(self._bin_buf) < frame_len:
                return None

            # Verify checksum
            checksum_offset = 2 + 1 + 1 + payload_len
            expected = struct.unpack_from("<H", self._bin_buf, checksum_offset)[0]
            calc = sum(self._bin_buf[2:checksum_offset]) & 0xFFFF

            if calc != expected:
                self._checksum_failures += 1
                if self.debug and (self._checksum_failures % 50 == 0):
                    self.logger.warning(
                        "Checksum mismatch (count=%d): calc=0x%04X expected=0x%04X",
                        self._checksum_failures,
                        calc,
                        expected,
                    )
                if not self.verify_checksum:
                    break
                # Drop one byte and rescan for sync
                del self._bin_buf[0:1]
                continue

            # Parse timestamp
            ts = struct.unpack_from("<I", self._bin_buf, 4)[0]
            self.last_timestamp_us = int(ts)

            # Parse sensor data
            imu_data = []
            offset = 8
            for _ in range(sensor_count):
                vals = struct.unpack_from("<fffffff", self._bin_buf, offset)
                imu_data.append(list(vals))
                offset += 28

            del self._bin_buf[:frame_len]
            return imu_data

    def read_imus(self):
        """
        Read IMU data (returns latest frame, discarding any backlog).

        Returns:
            List of 3 arrays, each containing [w, x, y, z, gx, gy, gz]
            (quaternion + gyro in dps), or None if no data available.
        """
        if self.simulation_mode:
            return self.generate_simulation_data()

        if self.debug and self.ser and self.ser.in_waiting > 0:
            peek = self.ser.read(self.ser.in_waiting)
            if peek:
                self.logger.info("RX(data): %r", peek)
            # Put bytes back into binary buffer for parsing
            self._bin_buf.extend(peek)

        # Drain buffer and return only the LATEST frame to avoid latency
        latest_data = None
        while True:
            data = self._read_binary_frame()
            if data is None:
                break
            latest_data = data

        return latest_data

    def quaternion_to_pitch(self, w, x, y, z):
        """Extract pitch angle (Y-axis rotation) from quaternion in degrees."""
        # Y-twist: 2*atan2(y, w)
        pitch = 2.0 * math.atan2(y, w) * 180.0 / math.pi
        if pitch <= -180.0:
            pitch += 360.0
        elif pitch > 180.0:
            pitch -= 360.0
        return pitch

    def iter_stream(self, sleep_s=0.001):
        """Generator that yields parsed IMU data as it arrives."""
        try:
            while True:
                data = self.read_imus()
                if data is not None:
                    yield data
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.close()

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Connection closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read 3x IMU quaternions over USB serial")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--sr", type=int, default=200, help="Sample rate Hz")
    parser.add_argument("--gyro-dps", type=int, default=500, help="Gyro range (125/250/500/1000/2000)")
    parser.add_argument("--gain", type=float, default=0.5, help="AHRS gain")
    parser.add_argument("--accel-rej", type=float, default=10.0, help="Accel rejection threshold (deg)")
    parser.add_argument("--recovery-s", type=float, default=1.0, help="Recovery period (s)")
    parser.add_argument("--offset-s", type=float, default=1.0, help="Gyro offset timeout (s)")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    parser.add_argument("--debug", action="store_true", help="Print raw serial data")
    parser.add_argument("--no-checksum", action="store_true",
                        help="Skip checksum verification (debug only)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for CFG_OK (useful if Pico is already streaming)")
    args = parser.parse_args()

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        simulation_mode=args.sim,
        debug=args.debug,
        verify_checksum=not args.no_checksum,
    )

    if not args.sim:
        reader.send_config(
            sample_rate=args.sr,
            gyro_dps=args.gyro_dps,
            gain=args.gain,
            accel_rejection=args.accel_rej,
            recovery_s=args.recovery_s,
            offset_s=args.offset_s,
        )
        if not args.no_wait:
            if not reader.wait_for_cfg_ok(timeout_s=5.0, resend_s=0.5):
                reader.logger.warning("CFG_OK not received; continuing anyway")

    print(f"Reading IMUs @ {args.sr} Hz (Ctrl+C to stop)...")
    t0 = time.time()
    n = 0

    try:
        while True:
            data = reader.read_imus()
            if data is not None:
                n += 1
                if time.time() - t0 >= 1.0:
                    # Print pitch angles
                    pitches = []
                    for i, imu in enumerate(data):
                        w, x, y, z = imu[0:4]
                        pitch = reader.quaternion_to_pitch(w, x, y, z)
                        pitches.append(f"IMU{i}:{pitch:+6.1f}")
                    ts = reader.last_timestamp_us
                    print(f"SPS={n:3d}  ts={ts:10d}  {' | '.join(pitches)}")
                    n = 0
                    t0 = time.time()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
