import serial
import time
import logging
import serial.tools.list_ports


class USBSerialReader:
    def __init__(self, baud_rate=115200, timeout=1.0, simulation_mode=False, log_level: str = "INFO",
                 port: str = None, data_format: str = "CSV"):
        """Initialize USB serial reader.

        Args:
            baud_rate: Serial baud rate
            timeout: Serial timeout in seconds
            simulation_mode: If True, generate synthetic data instead of reading hardware
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            port: Optional serial port override (e.g., "/dev/ttyACM0" or "COM3")
            data_format: "CSV" or "BIN" (binary framed)
        """
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.USBSerialReader")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.port = None
        self.simulation_mode = simulation_mode
        self.port = port
        self.data_format = (data_format or "CSV").upper()
        self.sim_time = 0.0
        self.last_timestamp_us = None
        self._bin_buf = bytearray()
        if not simulation_mode:
            self.connect()
        else:
            self.logger.info("Simulation mode: generating synthetic IMU data")

    def find_pico_port(self):
        """Try to find a likely XIAO RP2040 port automatically."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.logger.error("No serial ports found. Please specify manually (e.g., COM3).")
            return None

        keywords = ("XIAO", "RP2040", "Pico", "USB Serial", "CDC", "ACM")
        vidpid_markers = ("VID:PID=2E8A:", "VID:PID=2886:")
        for port in ports:
            desc = (port.description or "")
            hwid = (port.hwid or "")
            if any(k in desc for k in keywords) or any(m in hwid for m in vidpid_markers):
                self.logger.info(f"Found XIAO/RP2040 on port: {port.device}")
                return port.device

        # Default fallback: pick first available port on this system
        self.logger.warning(f"No descriptor match; defaulting to {ports[0].device}")
        return ports[0].device

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def connect(self):
        """Connect to the serial port"""
        if not self.port:
            self.port = self.find_pico_port()
        if not self.port:
            raise serial.SerialException("No serial port found")
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            self.logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
            # Wait a moment for connection to stabilize
            time.sleep(0.1)
        except serial.SerialException as e:
            self.logger.error(f"Serial connection error: {e}")
            raise

    def send_handshake_config(self, sensor_count=3, sample_rate=120, qmode="FULL",
                              lpf_enabled=1, lpf_alpha=0.995):
        """
        Send minimal configuration handshake to Pico.
        
        Args:
            sensor_count: Number of sensors
            lpf_enabled: Low-pass filter enabled
            lpf_alpha: Low-pass filter alpha value
            sample_rate: Sample rate in Hz
        """
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Serial port not connected - cannot send handshake")
            return False
        try:
            # CSV by default; allow binary framed stream when requested
            qm = (qmode or "FULL").upper()
            if qm not in ("FULL", "PITCH", "0", "1"):
                qm = "FULL"
            # NOTE (no flash needed): To change software smoothing, tweak lpf_alpha here.
            # Current filter is y = (1-alpha)*y_prev + alpha*x. For smoothing, use alpha ~0.05-0.15 at 120-200 Hz.
            # Example: set lpf_alpha=0.1 to reduce vibration; set 0.0 or disable LPF for maximum responsiveness.
            fmt = "BIN" if self.data_format == "BIN" else "CSV"
            format_response = (
                f"SC={sensor_count}|LPF_ENABLED={lpf_enabled}|LPF_ALPHA={lpf_alpha}|"
                f"SR={sample_rate}|QMODE={qm}|FMT={fmt}|\n"
            )
            self.ser.write(format_response.encode("utf-8"))
            self.logger.debug(f"Sent handshake config: {format_response.strip()}")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Error sending handshake: {e}")
            return False

    def send_zero(self):
        """Request re-zero from device (sets current pitch as zero)."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Serial port not connected - cannot send ZERO")
            return False
        try:
            self.ser.write(b"CMD=ZERO\n")
            self.logger.info("Sent ZERO command")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Error sending ZERO: {e}")
            return False

    def parse_imu_line(self, line):
        """
        Parse compact CSV line: timestamp_us, then per IMU -> w,x,y,z,gx,gy,gz
        Backward-compatible: if no leading timestamp, still parses.
        Returns list of per-IMU arrays and sets self.last_timestamp_us.
        """
        try:
            vals = [v for v in line.strip().split(',') if v]
            floats = []
            for v in vals:
                try:
                    floats.append(float(v))
                except Exception:
                    return None
            imu_data = []
            if not floats:
                return None
            if (len(floats) % 7) == 0:
                # No timestamp
                self.last_timestamp_us = None
                start = 0
            elif (len(floats) - 1) % 7 == 0:
                # Leading timestamp
                self.last_timestamp_us = int(floats[0])
                start = 1
            else:
                return None
            for i in range(start, len(floats), 7):
                chunk = floats[i:i+7]
                if len(chunk) < 7:
                    return None
                imu_data.append([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6]])
            return imu_data
        except Exception:
            return None

    def generate_simulation_data(self):
        """Generate synthetic IMU data for testing"""
        import math

        # Simulate Link0 rotating from 0 to 42 degrees over 10 seconds
        self.sim_time += 0.0167  # ~60Hz
        link0_angle = min(42.0, (self.sim_time / 10.0) * 42.0) * math.pi / 180.0

        # Calculate angular velocity (rad/s) for gyro simulation
        angular_velocity = (42.0 * math.pi / 180.0) / 10.0 if self.sim_time < 10.0 else 0.0

        # Generate realistic quaternions for each IMU
        # IMU0: Rotating around Y-axis (Link0)
        w0 = math.cos(link0_angle / 2)
        x0 = 0.0
        y0 = math.sin(link0_angle / 2)
        z0 = 0.0
        gx0, gy0, gz0 = 0.0, angular_velocity, 0.0  # Rotating around Y-axis

        # IMU1: Fixed relative to IMU0 (-58 degrees)
        rel_angle1 = -58.0 * math.pi / 180.0 + link0_angle
        w1 = math.cos(rel_angle1 / 2)
        x1 = 0.0
        y1 = math.sin(rel_angle1 / 2)
        z1 = 0.0
        gx1, gy1, gz1 = 0.0, angular_velocity, 0.0  # Same angular velocity

        # IMU2: Fixed relative to IMU0 (+65 degrees)
        rel_angle2 = 65.0 * math.pi / 180.0 + link0_angle
        w2 = math.cos(rel_angle2 / 2)
        x2 = 0.0
        y2 = math.sin(rel_angle2 / 2)
        z2 = 0.0
        gx2, gy2, gz2 = 0.0, angular_velocity, 0.0  # Same angular velocity

        # Always include gyro in simulation
        return [[w0, x0, y0, z0, gx0, gy0, gz0],
                [w1, x1, y1, z1, gx1, gy1, gz1],
                [w2, x2, y2, z2, gx2, gy2, gz2]]

    def _read_imus_binary(self):
        """Read framed binary IMU packets."""
        import struct

        if not self.ser or not self.ser.is_open:
            return None

        if self.ser.in_waiting > 0:
            self._bin_buf.extend(self.ser.read(self.ser.in_waiting))

        # Frame: 0xAA 0x55 | ver(1) | count(1) | ts_us(4) | payload | checksum(2)
        while True:
            if len(self._bin_buf) < 4:
                return None
            sync = self._bin_buf.find(b"\xAA\x55")
            if sync < 0:
                self._bin_buf.clear()
                return None
            if sync > 0:
                del self._bin_buf[:sync]
                if len(self._bin_buf) < 4:
                    return None
            if len(self._bin_buf) < 6:
                return None
            version = self._bin_buf[2]
            sensor_count = self._bin_buf[3]
            if version != 1 or sensor_count < 1 or sensor_count > 16:
                del self._bin_buf[0:2]
                continue
            payload_len = 4 + sensor_count * 7 * 4
            frame_len = 2 + 1 + 1 + payload_len + 2
            if len(self._bin_buf) < frame_len:
                return None
            checksum_offset = 2 + 1 + 1 + payload_len
            expected = struct.unpack_from("<H", self._bin_buf, checksum_offset)[0]
            calc = sum(self._bin_buf[2:checksum_offset]) & 0xFFFF
            if calc != expected:
                del self._bin_buf[0:2]
                continue
            ts = struct.unpack_from("<I", self._bin_buf, 4)[0]
            self.last_timestamp_us = int(ts)
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
        Read raw IMU data from serial port or simulation.

        Returns:
            list of N arrays, each containing [w, x, y, z, gx, gy, gz] (quaternion + gyro)
            or None if no valid data available
        """
        if self.simulation_mode:
            return self.generate_simulation_data()

        if not self.ser or not self.ser.is_open:
            return None

        if self.data_format == "BIN":
            return self._read_imus_binary()

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    return self.parse_imu_line(line)
            return None
        except serial.SerialException as e:
            self.logger.error(f"Serial read error: {e}")
            return None


    def iter_stream(self, sleep_s=0.0):
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
        """Close the serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Serial connection closed")

    def __del__(self):
        """Cleanup on object destruction"""
        self.close()


if __name__ == "__main__":
    # Minimal interactive example: print SPS and raw quaternions
    import argparse

    parser = argparse.ArgumentParser(description="Read mock IMU data over USB serial.")
    parser.add_argument("--port", help="Serial port override (e.g., /dev/ttyACM0 or COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Read timeout seconds")
    parser.add_argument("--sps", type=int, help="Sample rate in Hz for mock firmware handshake")
    parser.add_argument("--binary", action="store_true", help="Use binary framed stream")
    args = parser.parse_args()

    data_format = "BIN" if args.binary else "CSV"
    reader = USBSerialReader(baud_rate=args.baud, timeout=args.timeout, port=args.port,
                             data_format=data_format)
    sample_rate = args.sps if args.sps else 120
    reader.send_handshake_config(sensor_count=3, sample_rate=sample_rate, qmode="FULL",
                                 lpf_enabled=1, lpf_alpha=0.995)
    t0 = time.time()
    n = 0
    try:
        while True:
            data = reader.read_imus()
            if data is not None:
                n += 1
                if time.time() - t0 >= 1.0:
                    # Print raw quaternions
                    quats = []
                    gyro = []
                    for idx, imu in enumerate(data):
                        if len(imu) >= 4:
                            w, x, y, z = imu[0:4]
                            gx, gy, gz = 0.0, 0.0, 0.0
                            quats.append(f"[{w:.3f},{x:.3f},{y:.3f},{z:.3f}]")
                            gyro.append(f"[{gx:.2f},{gy:.2f},{gz:.2f}]")
                    ts = reader.last_timestamp_us
                    if ts is not None:
                        reader.logger.info(f"SPS={n}, ts_us={ts}, quats={quats}, gyro={gyro}")
                    else:
                        reader.logger.info(f"[No timestamp] SPS={n}")
                    n = 0
                    t0 = time.time()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
