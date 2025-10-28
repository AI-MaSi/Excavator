import serial
import json
import time
import sys
import re
import math
import serial.tools.list_ports


class USBSerialReader:
    def __init__(self, baud_rate=115200, timeout=1.0, simulation_mode=False):
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.port = None
        self.simulation_mode = simulation_mode
        self.sim_time = 0.0
        
        if not simulation_mode:
            self.connect()
        else:
            print("Running in simulation mode - generating synthetic IMU data")

    def find_pico_port(self):
        """Try to find the Pico automatically"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'Pico' in port.description or 'USB Serial' in port.description:
                return port.device
        
        # Different fallbacks for different platforms
        if sys.platform.startswith('win'):
            return 'COM3'  # Common Windows COM port
        else:
            return '/dev/ttyACM0'  # Linux/Mac fallback

    def connect(self):
        """Connect to the serial port"""
        self.port = self.find_pico_port()
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            # Wait a moment for connection to stabilize
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            raise

    def send_handshake_config(self, sensor_count=3, lpf_enabled=1, lpf_alpha=0.99, sample_rate=120):
        """
        Send configuration handshake to Pico with IMU settings.
        
        Args:
            sensor_count: Number of sensors (default: 3)
            lpf_enabled: Low-pass filter enabled (1/0, default: 1)
            lpf_alpha: Low-pass filter alpha value (default: 0.95)
            sample_rate: Sample rate in Hz (default: 120)
        """
        if not self.ser or not self.ser.is_open:
            print("Serial port not connected - cannot send handshake")
            return False
            
        try:
            format_response = f"SC={sensor_count}|LPF_ENABLED={lpf_enabled}|LPF_ALPHA={lpf_alpha}|SR={sample_rate}|\n"
            self.ser.write(format_response.encode("utf-8"))
            print(f"Sent handshake config: {format_response.strip()}")
            return True
        except serial.SerialException as e:
            print(f"Error sending handshake: {e}")
            return False

    def parse_imu_line(self, line):
        """
        Parse a line like:
        w0: -0.5033, x0: -0.2855, y0: -0.1870, z0: 0.7940 | w1: -0.3059, x1: 0.1590, y1: 0.0411, z1: 0.9379 | w2: -0.3349, x2: 0.6686, y2: 0.4297, z2: 0.5065 |

        Returns: list of [w, x, y, z] quaternion arrays for each IMU
        """
        try:
            # Split by | to get each IMU section
            imu_sections = line.split('|')[:-1]  # Remove last empty element

            imu_data = []
            for section in imu_sections:
                # Extract numbers using regex
                # Look for pattern: w0: number, x0: number, y0: number, z0: number
                matches = re.findall(r'[wxyz]\d+:\s*([-+]?\d*\.?\d+)', section.strip())

                if len(matches) == 4:
                    quaternion = [float(val) for val in matches]  # [w, x, y, z]
                    imu_data.append(quaternion)
                else:
                    # Silently skip malformed sections
                    return None

            return imu_data if len(imu_data) == 3 else None

        except Exception as e:
            print(f"Error parsing line '{line}': {e}")
            return None

    def generate_simulation_data(self):
        """Generate synthetic IMU data for testing"""
        import math
        
        # Simulate Link0 rotating from 0 to 42 degrees over 10 seconds
        self.sim_time += 0.0167  # ~60Hz
        link0_angle = min(42.0, (self.sim_time / 10.0) * 42.0) * math.pi / 180.0
        
        # Generate realistic quaternions for each IMU
        # IMU0: Rotating around Y-axis (Link0)
        w0 = math.cos(link0_angle / 2)
        x0 = 0.0
        y0 = math.sin(link0_angle / 2)
        z0 = 0.0
        
        # IMU1: Fixed relative to IMU0 (-58 degrees)
        rel_angle1 = -58.0 * math.pi / 180.0 + link0_angle
        w1 = math.cos(rel_angle1 / 2)
        x1 = 0.0
        y1 = math.sin(rel_angle1 / 2)
        z1 = 0.0
        
        # IMU2: Fixed relative to IMU0 (+65 degrees)
        rel_angle2 = 65.0 * math.pi / 180.0 + link0_angle
        w2 = math.cos(rel_angle2 / 2)
        x2 = 0.0
        y2 = math.sin(rel_angle2 / 2)
        z2 = 0.0
        
        return [[w0, x0, y0, z0], [w1, x1, y1, z1], [w2, x2, y2, z2]]

    def read_imus(self):
        """
        Read IMU data from serial port or simulation.
        Returns: list of 3 quaternion arrays [[w0,x0,y0,z0], [w1,x1,y1,z1], [w2,x2,y2,z2]]
        or None if no valid data available
        """
        if self.simulation_mode:
            return self.generate_simulation_data()
            
        if not self.ser or not self.ser.is_open:
            print("Serial port not connected")
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
        except Exception as e:
            print(f"Unexpected error reading serial: {e}")
            return None

    def close(self):
        """Close the serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")

    def __del__(self):
        """Cleanup on object destruction"""
        self.close()


def quat_to_euler_nwu(w, x, y, z):
    """
    Convert quaternion to Euler angles in NWU convention.
    Returns: (roll, pitch, yaw) in degrees
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def check_yaw_constraint(imu_data):
    """
    Check if yaw is properly constrained to ~0° for all IMUs.
    Prints yaw angles and returns max absolute yaw.
    """
    max_yaw = 0.0
    print("Yaw constraint check:")
    
    for i, quat in enumerate(imu_data):
        w, x, y, z = quat
        roll, pitch, yaw = quat_to_euler_nwu(w, x, y, z)
        abs_yaw = abs(yaw)
        max_yaw = max(max_yaw, abs_yaw)
        
        print(f"  IMU{i}: Yaw={yaw:6.2f}°, Pitch={pitch:6.2f}°, Roll={roll:6.2f}°")
    
    if max_yaw < 5.0:
        print(f"✓ Yaw constraint working well (max: {max_yaw:.2f}°)")
    else:
        print(f"⚠ Yaw constraint may have issues (max: {max_yaw:.2f}°)")
    
    return max_yaw


def analyze_quaternion_components(imu_data):
    """
    Analyze which quaternion components are changing to understand rotation axes.
    """
    print("Quaternion component analysis:")
    
    for i, quat in enumerate(imu_data):
        w, x, y, z = quat
        magnitude = math.sqrt(w*w + x*x + y*y + z*z)
        
        print(f"  IMU{i}: |q|={magnitude:.4f}, w={w:7.4f}, x={x:7.4f}, y={y:7.4f}, z={z:7.4f}")
        
        # Dominant component analysis
        components = {'w': abs(w), 'x': abs(x), 'y': abs(y), 'z': abs(z)}
        dominant = max(components, key=components.get)
        print(f"    Dominant component: {dominant} ({components[dominant]:.4f})")
    print()


def receive_pico_data_debug():
    """Debug function to continuously display incoming data"""
    reader = USBSerialReader()
    counter = 0
    start_time = time.time()

    try:
        print("Starting debug data reception...")
        while True:
            imu_data = reader.read_imus()
            if imu_data is not None:
                elapsed_time = time.time() - start_time
                counter += 1

                if elapsed_time >= 1:
                    print(f"Samples per second: {counter}")
                    counter = 0
                    start_time = time.time()

                # Print formatted IMU data with analysis
                print("IMU Data:")
                for i, imu in enumerate(imu_data):
                    print(f"  IMU{i}: w={imu[0]:7.4f}, x={imu[1]:7.4f}, y={imu[2]:7.4f}, z={imu[3]:7.4f}")
                
                # Check yaw constraint and analyze components
                check_yaw_constraint(imu_data)
                analyze_quaternion_components(imu_data)
                print("-" * 60)

            time.sleep(0.001)  # 1ms delay for 60Hz real-time control

    except KeyboardInterrupt:
        print("Stopping debug reception...")
    finally:
        reader.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        receive_pico_data_debug()
    else:
        print("USB Serial Reader Module")
        print("Usage: python usb_serial_reader.py [debug]")
        print("  debug: Run continuous debug output")