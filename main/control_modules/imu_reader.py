# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/

import board
import adafruit_tca9548a
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
import numpy as np
import time
from adafruit_lsm6ds import Rate, AccelRange, GyroRange
import numba


@numba.njit(fastmath=True)
def _quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (body frame) 
    q1, q2: quaternions in [w, x, y, z] format
    """
    # Ensure float32 inputs
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float32)


@numba.njit(fastmath=True)
def _quaternion_normalize(q):
    """Normalize a quaternion to unit magnitude """
    # Convert input to float32 to ensure consistent types
    q = np.asarray(q, dtype=np.float32)

    norm = np.sqrt(np.sum(q * q))
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Explicitly ensure float32 output
    return np.asarray(q / norm, dtype=np.float32)


@numba.njit(fastmath=True)
def _quaternion_exponential(v, scalar=0.0):
    """
    Compute quaternion exponential exp(q) 
    """
    # Ensure float32 input
    v = np.asarray(v, dtype=np.float32)
    scalar = np.float32(scalar)

    v_norm = np.sqrt(np.sum(v * v))

    # Handle small vector norm case
    if v_norm < 1e-10:
        return np.array([np.exp(scalar), 0.0, 0.0, 0.0], dtype=np.float32)

    # Calculate the exponential
    exp_scalar = np.float32(np.exp(scalar))
    factor = exp_scalar * np.sin(v_norm) / v_norm

    return np.array([
        exp_scalar * np.cos(v_norm),
        factor * v[0],
        factor * v[1],
        factor * v[2]
    ], dtype=np.float32)


@numba.njit(fastmath=True)
def _integrate_quaternion_exp(q, gyro, dt):
    """Integrate quaternion with exponential """
    # Ensure float32 inputs
    q = np.asarray(q, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    dt = np.float32(dt)

    half_angle = np.float32(0.5) * dt
    omega_vector = np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float32) * half_angle
    q_exp = _quaternion_exponential(omega_vector)
    q_new = _quaternion_multiply(q, q_exp)
    return _quaternion_normalize(q_new)


@numba.njit(fastmath=True)
def _rotate_vector_by_quaternion(v, q):
    """Rotate vector by quaternion """
    # Ensure float32 inputs
    v = np.asarray(v, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    p = np.array([0, v[0], v[1], v[2]], dtype=np.float32)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    qp = _quaternion_multiply(q, p)
    rotated_p = _quaternion_multiply(qp, q_conj)
    return rotated_p[1:4]


@numba.njit(fastmath=True)
def _madgwick_step(q, accel, gyro, beta, dt):
    """
    Single Madgwick filter step 
    Returns updated quaternion
    """
    # Ensure float32 inputs
    q = np.asarray(q, dtype=np.float32)
    accel = np.asarray(accel, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    beta = np.float32(beta)
    dt = np.float32(dt)

    # Normalize accelerometer
    acc_norm = np.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
    if acc_norm < 0.01:
        # Not enough acceleration, just integrate gyro
        gyro_quat = np.array([0, gyro[0], gyro[1], gyro[2]], dtype=np.float32)
        qDot = np.float32(0.5) * _quaternion_multiply(q, gyro_quat)
        q_new = q + qDot * dt
        return _quaternion_normalize(q_new)

    # Normalize accelerometer
    ax = accel[0] / acc_norm
    ay = accel[1] / acc_norm
    az = accel[2] / acc_norm

    # Objective function F
    F = np.zeros(3, dtype=np.float32)
    F[0] = 2.0 * (q[1] * q[3] - q[0] * q[2]) - ax
    F[1] = 2.0 * (q[0] * q[1] + q[2] * q[3]) - ay
    F[2] = 2.0 * (0.5 - q[1] ** 2 - q[2] ** 2) - az

    # Jacobian matrix J
    J = np.zeros((3, 4), dtype=np.float32)
    J[0, 0] = -2.0 * q[2]
    J[0, 1] = 2.0 * q[3]
    J[0, 2] = -2.0 * q[0]
    J[0, 3] = 2.0 * q[1]

    J[1, 0] = 2.0 * q[1]
    J[1, 1] = 2.0 * q[0]
    J[1, 2] = 2.0 * q[3]
    J[1, 3] = 2.0 * q[2]

    J[2, 0] = 0.0
    J[2, 1] = -4.0 * q[1]
    J[2, 2] = -4.0 * q[2]
    J[2, 3] = 0.0

    # Calculate gradient
    gradient = np.zeros(4, dtype=np.float32)
    gradient[0] = J[0, 0] * F[0] + J[1, 0] * F[1] + J[2, 0] * F[2]
    gradient[1] = J[0, 1] * F[0] + J[1, 1] * F[1] + J[2, 1] * F[2]
    gradient[2] = J[0, 2] * F[0] + J[1, 2] * F[1] + J[2, 2] * F[2]
    gradient[3] = J[0, 3] * F[0] + J[1, 3] * F[1] + J[2, 3] * F[2]

    # Normalize gradient
    gradient_norm = np.sqrt(np.sum(gradient * gradient))
    if gradient_norm > 1e-10:
        gradient = gradient / gradient_norm

    # Rate of change from gyroscope
    gyro_quat = np.array([0, gyro[0], gyro[1], gyro[2]], dtype=np.float32)
    qDot = np.float32(0.5) * _quaternion_multiply(q, gyro_quat) - beta * gradient

    # Integrate
    q_new = q + qDot * dt

    # Normalize and return
    return _quaternion_normalize(q_new)


class QuatIMUReader:
    """
    Quaternion-based IMU reader optimized for small angular velocities.
    Provides direct quaternion outputs using advanced integration techniques.
    """

    def __init__(self,
                 filter_alpha: float = 1.0,
                 beta: float = 1.00,
                 spike_threshold: float = 45.0,
                 max_imus: int = 8,
                 integration_method: str = "exponential",
                 mux_a0: bool = False,
                 mux_a1: bool = False,
                 mux_a2: bool = False,
                 dead_band: float = 0.0,
                 moving_avg_samples: int = 10):
        """
        Initialize the quaternion-optimized IMU reader

        Args:
            filter_alpha: Filter strength for accelerometer/gyro data (0-1)
                          Lower values better for small movements
            beta: Madgwick filter parameter (0-?) - This was not limited to 0...1 fuck me
                  Lower values better for stability with small movements
            spike_threshold: deg/s - max allowed sudden change
            max_imus: Maximum number of IMUs to support
            integration_method: Method to use for quaternion integration
                                "madgwick" - Madgwick's gradient descent algorithm
                                "exponential" - Direct quaternion exponential integration (best for small movements)
                                "adaptive" - Adaptive algorithm that switches based on motion
            mux_a0: Boolean value for A0 pin of multiplexer (adds 1 to base address if True)
            mux_a1: Boolean value for A1 pin of multiplexer (adds 2 to base address if True)
            mux_a2: Boolean value for A2 pin of multiplexer (adds 4 to base address if True)
            dead_band: Dead band threshold for gyro readings in deg/s
            moving_avg_samples: Number of samples to use in moving average filter
        """

        # self.accel_rate = Rate.RATE_416_HZ
        # self.gyro_rate = Rate.RATE_416_HZ
        # self.accel_range = AccelRange.RANGE_2G
        # self.gyro_range = GyroRange.RANGE_125_DPS

        # Calculate multiplexer address based on A0, A1, A2 pins
        mux_addr = 0x70  # Base address
        if mux_a0:
            mux_addr += 1
        if mux_a1:
            mux_addr += 2
        if mux_a2:
            mux_addr += 4

        print(f"Using multiplexer address: 0x{mux_addr:02X}")

        # Initialize I2C and multiplexer
        self.i2c = board.I2C()
        self.mux = adafruit_tca9548a.TCA9548A(self.i2c, address=mux_addr)

        # Pre-allocate with maximum size
        self.max_imus = int(max_imus)
        self.imus = {}
        self.imu_count = 0

        # Initialize IMUs
        self.init_imus()

        # Filter settings
        self.alpha = float(filter_alpha)  # Low-pass filter strength (lower for small movements)
        self.beta = float(beta)  # Madgwick filter parameter (lower for small movements)
        self.min_beta = beta / 4  # For adaptive filtering
        self.max_beta = beta * 2  # For adaptive filtering

        # Dead band for very small movements (deg/s)
        self.dead_band = float(dead_band)

        # Select integration method
        self.integration_method = str(integration_method)

        # Bias compensation (updated during static periods)
        self.gyro_bias = None  # Will be initialized after first readings

        # Store moving average samples
        self.moving_avg_samples = int(moving_avg_samples)

        # Initialize timing reference
        self.last_time = time.monotonic()

        # Only allocate arrays if IMUs were found
        if self.imu_count > 0:
            # Raw sensor data: [num_imus, 6] for accel_x/y/z, gyro_x/y/z
            self.data_array = np.zeros((self.imu_count, 6), dtype=np.float32)
            self.prev_data = np.zeros((self.imu_count, 6), dtype=np.float32)

            # Filtered data storage
            self.filtered_data = np.zeros((self.imu_count, 6), dtype=np.float32)

            # Moving average storage for gyro (configurable samples)
            self.gyro_history = np.zeros((self.imu_count, self.moving_avg_samples, 3), dtype=np.float32)
            self.history_index = 0

            # Quaternion data: [num_imus, 4] for w, x, y, z
            self.quaternions = np.zeros((self.imu_count, 4), dtype=np.float32)
            # Initialize all quaternions to identity [1, 0, 0, 0]
            self.quaternions[:, 0] = 1.0

            # Fixed time step for more consistent quaternion integration
            self.fixed_dt = 1.0 / 100.0  # 100Hz internal update rate

            # Performance tracking
            self.read_times = np.zeros(10)
            self.read_time_idx = 0

            # Quaternion magnitudes for diagnostics
            self.quat_magnitudes = np.zeros(self.imu_count, dtype=np.float32)

            # Motion detection
            self.motion_threshold = 0.5  # deg/s - threshold to detect static state (lower for small movements)
            self.is_static = np.zeros(self.imu_count, dtype=bool)
            self.static_duration = np.zeros(self.imu_count, dtype=np.float32)

            # Calibration counters
            self.last_bias_update = time.monotonic()
            self.gyro_bias = np.zeros((self.imu_count, 3), dtype=np.float32)
            self.bias_samples = np.zeros(self.imu_count, dtype=np.int32)

            # Spike detection
            self.spike_threshold = spike_threshold

    def init_imus(self, check_both_addresses=True):
        """
        Initialize all available IMUs on multiplexer
        """

        # Define possible IMU addresses
        imu_addresses = [0x6A]  # Default address
        if check_both_addresses:
            imu_addresses.append(0x6B)  # Address when A0 pin is high

        # Try to configure each multiplexer channel
        for channel in range(8):  # TCA9548A has 8 channels
            channel_bus = self.mux[channel]

            # Try each possible IMU address
            for addr in imu_addresses:
                try:
                    print(f"Trying IMU at channel {channel}, address 0x{addr:02X}...")

                    # Try to initialize IMU at specific address
                    imu = ISM330DHCX(channel_bus, address=addr)

                    # Configure for specified rates and ranges
                    # imu.accelerometer_data_rate = self.accel_rate
                    # imu.gyro_data_rate = self.gyro_rate
                    # imu.accelerometer_range = self.accel_range
                    # imu.gyro_range = self.gyro_range

                    # Test read to verify it works
                    accel = imu.acceleration
                    gyro = imu.gyro

                    # Verify sensor readings are valid (not zero or NaN)
                    if (np.isnan(accel).any() or np.isnan(gyro).any() or
                            (accel[0] == 0 and accel[1] == 0 and accel[2] == 0)):
                        print(f"Skipping IMU on channel {channel} address 0x{addr:02X}: Invalid initial readings")
                        continue

                    # Store IMU with successful initialization
                    self.imus[self.imu_count] = {
                        'channel': channel,
                        'imu': imu,
                        'address': addr
                    }

                    print(f"Initialized IMU_{self.imu_count} on channel {channel}, address 0x{addr:02X}")
                    print(f"  Accel: {accel}, Gyro: {gyro}")
                    # print(f"  Configured with settings: accel={accel_rate}, gyro={gyro_rate}, "
                    #      f"accel_range={accel_range}, gyro_range={gyro_range}")

                    self.imu_count += 1

                    # delay between IMU initializations to avoid I2C bus issues
                    time.sleep(0.01)

                    # No need to try second address if we already found an IMU on this channel
                    # Comment this out if you want to support 2 IMUs per channel
                    break

                except Exception as e:
                    # Skip non-responsive channels/addresses
                    print(f"Failed to initialize IMU on channel {channel}, address 0x{addr:02X}: {e}")
                    continue

    def set_imu_config(self, accel_range=None, gyro_range=None, accel_rate=None, gyro_rate=None):
        """
        Update accelerometer and gyroscope ranges and rates for all IMUs

        Args:
            accel_range: New accelerometer range (from AccelRange)
            gyro_range: New gyroscope range (from GyroRange)
            accel_rate: New accelerometer data rate (from Rate)
            gyro_rate: New gyroscope data rate (from Rate)

        Returns:
            True if successful, False otherwise
        """
        success = True

        for imu_num, imu_data in self.imus.items():
            try:
                imu = imu_data['imu']

                if accel_range is not None:
                    imu.accelerometer_range = accel_range
                    self.accel_range = accel_range

                if gyro_range is not None:
                    imu.gyro_range = gyro_range
                    self.gyro_range = gyro_range

                if accel_rate is not None:
                    imu.accelerometer_data_rate = accel_rate
                    self.accel_rate = accel_rate

                if gyro_rate is not None:
                    imu.gyro_data_rate = gyro_rate
                    self.gyro_rate = gyro_rate

                # Test read to verify it works
                accel = imu.acceleration
                gyro = imu.gyro

                print(f"Updated IMU_{imu_num} config:")
                print(f"  Rates: accel={imu.accelerometer_data_rate}, gyro={imu.gyro_data_rate}")
                print(f"  Ranges: accel={imu.accelerometer_range}, gyro={imu.gyro_range}")

            except Exception as e:
                print(f"Failed to update configuration for IMU_{imu_num}: {e}")
                success = False

        return success

    def apply_low_pass_filter(self, new_data):
        """
        Apply simple low-pass filter to smooth sensor data
        Uses the formula: filtered = alpha * new_data + (1 - alpha) * filtered_old
        """
        # If this is the first reading, initialize filtered data with raw values
        if np.all(self.filtered_data == 0):
            np.copyto(self.filtered_data, new_data)  # In-place copy
        else:
            # Apply exponential low-pass filter (vectorized operation)
            # Using in-place operation to avoid temporary array creation
            self.filtered_data *= (1 - self.alpha)
            self.filtered_data += self.alpha * new_data

        return self.filtered_data

    def apply_moving_average(self, new_data):
        """
        Apply moving average filter to gyro data
        Helps reduce noise for small angular velocities
        """
        # Update history buffer
        for i in range(self.imu_count):
            # Store new gyro readings in history
            self.gyro_history[i, self.history_index] = new_data[i, 3:6]

            # Calculate moving average and update filtered data
            avg_gyro = np.mean(self.gyro_history[i], axis=0)
            self.filtered_data[i, 3:6] = avg_gyro

        # Update history index
        self.history_index = (self.history_index + 1) % self.moving_avg_samples

        return self.filtered_data

    def apply_spike_detection(self, new_data):
        """
        Detect and remove sudden spikes in gyro data
        Important for small angular velocity measurement
        """
        # Skip spike detection if threshold is 0.0 or negative
        if self.spike_threshold <= 0.0:
            # Still store current data as previous for next time
            np.copyto(self.prev_data, new_data)
            return new_data

        for i in range(self.imu_count):
            # Compute difference with previous reading
            gyro_diff = np.abs(new_data[i, 3:6] - self.prev_data[i, 3:6])

            # Check for spikes
            if np.max(gyro_diff) > self.spike_threshold:
                # Spike detected - use previous value
                new_data[i, 3:6] = self.prev_data[i, 3:6]
                print(f"Spike detected in IMU_{i}, value rejected")

        # Store current data as previous for next time
        np.copyto(self.prev_data, new_data)

        return new_data

    def apply_dead_band(self, data):
        """
        Apply dead band to gyro readings
        Any value below threshold is set to exactly zero
        """
        for i in range(self.imu_count):
            # Get gyro data
            gyro = data[i, 3:6]

            # Apply dead band
            mask = np.abs(gyro) < self.dead_band
            gyro[mask] = 0.0

            # Update data
            data[i, 3:6] = gyro

        return data

    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions (body frame)
        q1, q2: quaternions in [w, x, y, z] format
        q1 ⊗ q2: Apply rotation q2 after rotation q1
        """
        return _quaternion_multiply(q1, q2)

    @staticmethod
    def quaternion_normalize(q):
        """Normalize a quaternion to unit magnitude"""
        return _quaternion_normalize(q)

    @staticmethod
    def quaternion_exponential(v, scalar=0.0):
        """
        Compute quaternion exponential exp(q) for quaternion q = [scalar, v]
        Based on formula from 'How to Integrate Quaternions' paper
        """
        return _quaternion_exponential(v, scalar)

    @staticmethod
    def integrate_quaternion_exp(q, gyro, dt):
        """
        Integrate angular velocity using quaternion exponential method

        Args:
            q: Current quaternion [w, x, y, z]
            gyro: Angular velocity [gx, gy, gz] in rad/s
            dt: Time step in seconds

        Returns:
            Updated quaternion
        """
        return _integrate_quaternion_exp(q, gyro, dt)

    def detect_motion_state(self):
        """
        Detect if the IMU is in a static or moving state
        Updates is_static and static_duration arrays
        Used for auto-calibration and adaptive filtering
        """
        current_time = time.monotonic()
        dt = current_time - self.last_time

        for i in range(self.imu_count):
            # Get gyro magnitude
            gx, gy, gz = self.filtered_data[i, 3:6]
            gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

            # Check if static
            if gyro_mag < self.motion_threshold:
                # IMU is static
                if not self.is_static[i]:
                    # Just became static
                    self.is_static[i] = True
                    self.static_duration[i] = 0
                else:
                    # Update static duration
                    self.static_duration[i] += dt
            else:
                # IMU is moving
                self.is_static[i] = False
                self.static_duration[i] = 0

        # Update gyro bias if there's been enough static time
        if current_time - self.last_bias_update > 2.0:  # Every 2 seconds
            self.update_gyro_bias()
            self.last_bias_update = current_time

    def update_gyro_bias(self):
        """Update gyroscope bias during static periods"""
        for i in range(self.imu_count):
            # Only update bias during extended static periods
            if self.is_static[i] and self.static_duration[i] > 1.0:  # After 1 second static
                # Get current gyro readings
                gyro = self.filtered_data[i, 3:6]

                # Update bias using incremental average
                if self.bias_samples[i] < 100:  # Limit number of samples
                    self.bias_samples[i] += 1
                    # Incremental average formula
                    self.gyro_bias[i] = self.gyro_bias[i] + (gyro - self.gyro_bias[i]) / self.bias_samples[i]

                    if self.bias_samples[i] % 10 == 0:
                        print(f"Updated gyro bias for IMU_{i}: {self.gyro_bias[i]}")

    def apply_gyro_bias(self, data_array):
        """Apply gyro bias correction to data array"""
        if self.gyro_bias is not None:
            for i in range(self.imu_count):
                if self.bias_samples[i] > 5:  # Only apply if we have enough samples
                    # Apply bias correction to gyro data
                    data_array[i, 3:6] -= self.gyro_bias[i]
        return data_array

    def update_quaternions_madgwick(self, dt):
        """
        Update quaternion orientation using Madgwick's gradient descent algorithm
        This algorithm integrates gyroscope measurements and corrects with accelerometer
        """
        for i in range(self.imu_count):
            # Get current quaternion
            q = self.quaternions[i]

            # Get filtered accelerometer and gyroscope data
            accel = self.filtered_data[i, 0:3]
            gyro = self.filtered_data[i, 3:6]

            # Convert gyro from deg/s to rad/s
            gyro = np.radians(gyro)

            # Use Numba-optimized Madgwick step
            self.quaternions[i] = _madgwick_step(q, accel, gyro, self.beta, dt)
            self.quat_magnitudes[i] = np.sqrt(np.sum(self.quaternions[i] * self.quaternions[i]))

        return self.quaternions

    def update_quaternions_exponential(self, dt):
        """
        Update quaternions using direct quaternion exponential integration
        More accurate with small angular velocities
        """
        for i in range(self.imu_count):
            # Get current quaternion
            q = self.quaternions[i]

            # Get gyroscope data in rad/s
            gyro = np.radians(self.filtered_data[i, 3:6])

            # Directly integrate using quaternion exponential formula
            q_new = self.integrate_quaternion_exp(q, gyro, dt)

            # Apply accelerometer correction if possible
            accel = self.filtered_data[i, 0:3]
            acc_norm = np.sqrt(np.sum(accel ** 2))

            if acc_norm > 0.01:  # Only correct if we have significant acceleration
                # Normalize accelerometer
                accel = accel / acc_norm

                # Simple accelerometer correction toward gravity
                # Create a rotation that would align current gravity vector with measured acceleration
                current_down = QuatIMUReader.rotate_vector_by_quaternion(np.array([0, 0, 1]), q_new)

                correction_strength = 0.005  # Smaller for fine movements

                # Calculate rotation axis between vectors (cross product)
                rotation_axis = np.cross(current_down, accel)
                rotation_angle = np.arccos(np.clip(np.dot(current_down, accel), -1.0, 1.0))

                # Apply small correction - create correction quaternion
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    correction_angle = rotation_angle * correction_strength
                    correction_quat = QuatIMUReader.quaternion_from_axis_angle(rotation_axis, correction_angle)

                    # Apply correction
                    q_new = QuatIMUReader.quaternion_multiply(correction_quat, q_new)

            # Store updated and normalized quaternion
            self.quaternions[i] = QuatIMUReader.quaternion_normalize(q_new)

            # Store quaternion magnitude for diagnostics
            self.quat_magnitudes[i] = np.sqrt(np.sum(self.quaternions[i] * self.quaternions[i]))

        return self.quaternions

    def update_quaternions_adaptive(self, dt):
        """
        Adaptive quaternion integration method
        - Uses exponential integration for most cases
        - Uses Madgwick when more stability is needed
        """
        for i in range(self.imu_count):
            # Determine motion intensity to select algorithm
            gx, gy, gz = self.filtered_data[i, 3:6]
            gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

            # Get accelerometer data
            accel = self.filtered_data[i, 0:3]
            acc_norm = np.sqrt(np.sum(accel ** 2))

            # Choose algorithm based on conditions
            if gyro_mag < 0.2:  # Very small movement - use exponential for accuracy
                gyro = np.radians([gx, gy, gz])
                q = self.quaternions[i]
                q_new = QuatIMUReader.integrate_quaternion_exp(q, gyro, dt)
                self.quaternions[i] = QuatIMUReader.quaternion_normalize(q_new)

            elif acc_norm < 0.1:  # Free-fall or large acceleration - just use gyro
                gyro = np.radians([gx, gy, gz])
                q = self.quaternions[i]
                gyro_quat = np.array([0, gyro[0], gyro[1], gyro[2]], dtype=np.float32)
                qDot = 0.5 * QuatIMUReader.quaternion_multiply(q, gyro_quat)
                q_new = q + qDot * dt
                self.quaternions[i] = QuatIMUReader.quaternion_normalize(q_new)

            else:  # Normal movement with good accel - use Madgwick
                q = self.quaternions[i]
                gyro = np.radians([gx, gy, gz])
                self.quaternions[i] = _madgwick_step(q, accel, gyro, self.beta, dt)

            # Store quaternion magnitude for diagnostics
            self.quat_magnitudes[i] = np.sqrt(np.sum(self.quaternions[i] * self.quaternions[i]))

        return self.quaternions

    @staticmethod
    def rotate_vector_by_quaternion(v, q):
        """
        Rotate vector v by quaternion q
        v: 3D vector
        q: quaternion [w, x, y, z]

        Returns: rotated vector
        """
        return _rotate_vector_by_quaternion(v, q)

    @staticmethod
    def quaternion_from_axis_angle(axis, angle):
        """
        Create quaternion from rotation axis and angle

        Args:
            axis: normalized 3D vector representing rotation axis
            angle: rotation angle in radians

        Returns:
            quaternion [w, x, y, z]
        """
        half_angle = angle * 0.5
        sin_half = np.sin(half_angle)

        w = np.cos(half_angle)
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half

        return np.array([w, x, y, z], dtype=np.float32)

    def read_quats(self, decimals=4, apply_filter=True):
        """
        Read quaternions from all IMUs - this is the main method to use

        Args:
            decimals: Number of decimal places to round quaternion values
            apply_filter: Whether to apply low-pass filtering to sensor data

        Returns:
            numpy array of shape [num_imus, 4] with quaternion values [w, x, y, z]
            None if no IMUs are available
        """
        # Start timing the read operation
        start_time = time.monotonic()

        if self.imu_count == 0:
            return None

        has_errors = False

        # Read raw data from all IMUs
        for imu_num, imu_data in self.imus.items():
            try:
                imu = imu_data['imu']

                # Read data - this is the potential bottleneck on I2C
                accel = imu.acceleration
                gyro = imu.gyro

                # Store in NumPy array
                self.data_array[imu_num, 0:3] = accel
                self.data_array[imu_num, 3:6] = gyro

            except Exception as e:
                print(f"Error reading IMU_{imu_num}: {e}")
                has_errors = True
                continue

        # Round raw data (vectorized operation)
        np.round(self.data_array, decimals, out=self.data_array)

        # Apply spike detection to remove spurious readings
        self.data_array = self.apply_spike_detection(self.data_array)

        # Apply gyro bias correction
        self.data_array = self.apply_gyro_bias(self.data_array)

        # Apply dead band to very small readings
        self.data_array = self.apply_dead_band(self.data_array)

        # Apply low-pass filter if requested
        if apply_filter:
            self.filtered_data = self.apply_low_pass_filter(self.data_array)
            # Apply additional moving average to gyro for very small movements
            self.filtered_data = self.apply_moving_average(self.filtered_data)
        else:
            self.filtered_data = self.data_array.copy()

        # Update motion state detection
        self.detect_motion_state()

        # Get current time and compute time delta
        current_time = time.monotonic()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Use fixed time step for more consistent updates
        if dt > 0.1:  # If we've been paused for a while, use a reasonable default
            dt = self.fixed_dt

        # Constrain dt to prevent instability
        dt = min(max(dt, 0.001), 0.05)  # Constrain between 1ms and 50ms

        # Update quaternions using selected method
        if self.integration_method == "madgwick":
            quaternions = self.update_quaternions_madgwick(dt)
        elif self.integration_method == "exponential":
            quaternions = self.update_quaternions_exponential(dt)
        elif self.integration_method == "adaptive":
            quaternions = self.update_quaternions_adaptive(dt)
        else:
            # Default to exponential if unknown method
            quaternions = self.update_quaternions_exponential(dt)

        # Round quaternions to desired precision
        np.round(quaternions, decimals, out=quaternions)

        # Store timing statistics for performance monitoring
        elapsed = time.monotonic() - start_time
        self.read_times[self.read_time_idx] = elapsed
        self.read_time_idx = (self.read_time_idx + 1) % len(self.read_times)

        # Print timing occasionally for debugging
        # if self.read_time_idx == 0:
        #    avg_time = np.mean(self.read_times[self.read_times > 0])
        #    print(f"Quat read time: {avg_time * 1000:.2f}ms ({1.0 / avg_time:.1f} Hz)")

        # If there were errors reading some IMUs, warn that arrays may contain invalid data
        if has_errors:
            print("Warning: Some IMUs had read errors, quaternion array may contain invalid data")

        return quaternions

    def get_quaternion_magnitudes(self):
        """
        Get the magnitude of each quaternion for diagnostic purposes
        Should be very close to 1.0 for valid quaternions

        Returns:
            numpy array of shape [num_imus] with quaternion magnitudes
        """
        return self.quat_magnitudes

    def set_filter_strength(self, alpha):
        """
        Adjust the filter strength (0-1)
        Lower values = more filtering, higher values = faster response
        For small movements, values around 0.05 are recommended
        """
        if 0 <= alpha <= 1:
            self.alpha = alpha
            return True
        return False

    def set_madgwick_beta(self, beta):
        """
        Adjust the Madgwick filter beta parameter (0-1)
        Higher values = faster convergence but more noise
        Lower values = smoother but slower to respond
        For small movements, values around 0.05 are recommended
        """
        if 0 <= beta <= 1:
            self.beta = beta
            self.min_beta = beta / 4
            self.max_beta = beta * 2
            return True
        return False

    def set_integration_method(self, method):
        """
        Change the quaternion integration method

        Args:
            method: One of ["madgwick", "exponential", "adaptive"]

        Returns:
            True if successful, False if invalid method
        """
        valid_methods = ["madgwick", "exponential", "adaptive"]
        if method in valid_methods:
            self.integration_method = method
            print(f"Changed integration method to {method}")
            return True
        return False

    def set_dead_band(self, threshold):
        """
        Set dead band threshold for very small movements

        Args:
            threshold: Dead band in deg/s (typically 0.05-0.2)

        Returns:
            True if successful
        """
        if threshold >= 0:
            self.dead_band = threshold
            print(f"Set dead band to {threshold} deg/s")
            return True
        return False

    def set_spike_threshold(self, threshold):
        """
        Set spike detection threshold

        Args:
            threshold: Maximum allowed sudden change in deg/s

        Returns:
            True if successful
        """
        if threshold > 0:
            self.spike_threshold = threshold
            print(f"Set spike threshold to {threshold} deg/s")
            return True
        return False

    def set_moving_avg_samples(self, samples):
        """
        Set the number of samples used in the moving average filter

        Args:
            samples: Number of samples (integer > 0)

        Returns:
            True if successful, False otherwise
        """
        if samples <= 0:
            return False

        # Store new value
        samples = int(samples)

        if samples == self.moving_avg_samples:
            return True  # No change needed

        # Create new history array with new size
        new_history = np.zeros((self.imu_count, samples, 3), dtype=np.float32)

        # Reset history index
        self.history_index = 0
        self.moving_avg_samples = samples
        self.gyro_history = new_history

        print(f"Changed moving average filter to use {samples} samples")
        return True

    def get_performance_stats(self):
        """
        Return performance statistics
        """
        valid_times = self.read_times[self.read_times > 0]
        if len(valid_times) == 0:
            return {
                'avg_time_ms': 0,
                'avg_rate_hz': 0,
                'min_time_ms': 0,
                'max_time_ms': 0
            }

        avg_time = np.mean(valid_times)
        return {
            'avg_time_ms': avg_time * 1000,
            'avg_rate_hz': 1.0 / avg_time if avg_time > 0 else 0,
            'min_time_ms': np.min(valid_times) * 1000 if len(valid_times) > 0 else 0,
            'max_time_ms': np.max(valid_times) * 1000 if len(valid_times) > 0 else 0
        }

    def reinitialize_imu(self, imu_num):
        """
        Attempt to reinitialize a failed IMU
        Useful if an IMU stops responding
        """
        if imu_num not in self.imus:
            return False

        try:
            channel = self.imus[imu_num]['channel']
            channel_bus = self.mux[channel]
            imu = ISM330DHCX(channel_bus)

            # Set the configuration using stored settings
            # imu.accelerometer_data_rate = self.accel_rate
            # imu.gyro_data_rate = self.gyro_rate
            # imu.accelerometer_range = self.accel_range
            # imu.gyro_range = self.gyro_range

            # Test read
            accel = imu.acceleration
            gyro = imu.gyro

            # Update the IMU object
            self.imus[imu_num]['imu'] = imu
            print(f"Reinitialized IMU_{imu_num} on channel {channel}")
            return True
        except Exception as e:
            print(f"Failed to reinitialize IMU_{imu_num}: {e}")
            return False

    @staticmethod
    def quaternion_to_euler(q):
        """
        Utility method to convert quaternion to Euler angles if needed for debugging
        Do not use this in your main control loop - stick with quaternions for performance

        Args:
            q: quaternion [w, x, y, z]

        Returns:
            Numpy array with [roll, pitch, yaw] in radians
        """
        # Extract quaternion components
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        # Check for gimbal lock
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if in gimbal lock
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])