import serial
import time
import serial.tools.list_ports


class USBSerialReader:
    def __init__(self, baud_rate=115200, timeout=1.0, simulation_mode=False):
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.port = None
        self.simulation_mode = simulation_mode
        self.sim_time = 0.0
        self.last_timestamp_us = None
        if not simulation_mode:
            self.connect()
        else:
            print("Simulation mode: generating synthetic IMU data")

    def find_pico_port(self):
        """Try to find the Pico automatically"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'Pico' in port.description or 'USB Serial' in port.description:
                print(f"Found Pico on port: {port.device}")
                return port.device

        # Default fallback: pick first available port on this system
        if ports:
            print(f"No Pico descriptor match; defaulting to {ports[0].device}")
            return ports[0].device
        print("No serial ports found. Please specify manually (e.g., COM3).")
        return 'COM3'

    def connect(self):
        """Connect to the serial port"""
        self.port = self.find_pico_port()
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            # Wait a moment for connection to stabilize
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
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
            print("Serial port not connected - cannot send handshake")
            return False
        try:
            # Always CSV stream on-device; minimal fields
            qm = (qmode or "FULL").upper()
            if qm not in ("FULL", "PITCH", "0", "1"):
                qm = "FULL"
            # NOTE (no flash needed): To change software smoothing, tweak lpf_alpha here.
            # Current filter is y = (1-alpha)*y_prev + alpha*x. For smoothing, use alpha ~0.05–0.15 at 120–200 Hz.
            # Example: set lpf_alpha=0.1 to reduce vibration; set 0.0 or disable LPF for maximum responsiveness.
            format_response = (
                f"SC={sensor_count}|LPF_ENABLED={lpf_enabled}|LPF_ALPHA={lpf_alpha}|SR={sample_rate}|QMODE={qm}|\n"
            )
            self.ser.write(format_response.encode("utf-8"))
            print(f"Sent handshake config: {format_response.strip()}")
            return True
        except serial.SerialException as e:
            print(f"Error sending handshake: {e}")
            return False

    def send_zero(self):
        """Request re-zero from device (sets current pitch as zero)."""
        if not self.ser or not self.ser.is_open:
            print("Serial port not connected - cannot send ZERO")
            return False
        try:
            self.ser.write(b"CMD=ZERO\n")
            print("Sent ZERO command")
            return True
        except serial.SerialException as e:
            print(f"Error sending ZERO: {e}")
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

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    return self.parse_imu_line(line)
            return None
        except serial.SerialException as e:
            print(f"Serial read error: {e}")
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
            print("Serial connection closed")

    def __del__(self):
        """Cleanup on object destruction"""
        self.close()


if __name__ == "__main__":
    # Minimal interactive example: print SPS and raw quaternions
    reader = USBSerialReader()
    reader.send_handshake_config(sensor_count=3, sample_rate=120, qmode="FULL", lpf_enabled=1, lpf_alpha=0.995)
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
                    for idx, imu in enumerate(data):
                        if len(imu) >= 4:
                            w, x, y, z = imu[0:4]
                            quats.append(f"[{w:.3f},{x:.3f},{y:.3f},{z:.3f}]")
                    ts = reader.last_timestamp_us
                    if ts is not None:
                        print(f"SPS={n}, ts_us={ts}, raw quats={quats}")
                    else:
                        print(f"SPS={n}, raw quats={quats}")
                    n = 0
                    t0 = time.time()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
