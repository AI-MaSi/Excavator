#!/usr/bin/env python3
"""
Hardware Interface

...

Usage:
    ...
"""



import time
import numpy as np
from typing import List, Optional, Dict, Any
import threading

from .PCA9685_controller import PWMController
from .usb_serial_reader import USBSerialReader
from .ADC import SimpleADC
from .quaternion_math import quat_from_axis_angle, wrap_to_pi


class EncoderTracker:
    def __init__(self, gear_ratio, min_voltage=0.5, max_voltage=4.5):
        """
        Initialize encoder tracker with simple continuous angle tracking.

        :param gear_ratio: Gear ratio (encoder_rotations / actual_rotations)
        :param min_voltage: Minimum voltage of encoder range
        :param max_voltage: Maximum voltage of encoder range
        """
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.voltage_range = max_voltage - min_voltage
        self.gear_ratio = gear_ratio
        self.z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-axis for rotation

        # Simple wrapping tracking
        self.last_wrapped_angle = None
        self.total_rotations = 0.0
        self.zero_offset = None  # Will be set from first reading

    def update(self, raw_voltage: float) -> dict:
        """
        Update encoder tracking with simple wrap detection and zero calibration.

        :param raw_voltage: Raw voltage from encoder
        :return: Dictionary with tracking information
        """
        # Convert voltage to angle in [0, 2π] range first
        # Clamp voltage to valid range and normalize
        clamped_voltage = max(self.min_voltage, min(self.max_voltage, raw_voltage))
        voltage_fraction = (clamped_voltage - self.min_voltage) / self.voltage_range
        raw_angle = voltage_fraction * 2 * np.pi  # Map [0,1] to [0,2π]

        # Convert to [-π, π] range
        wrapped_angle = wrap_to_pi(np.array([raw_angle]))[0]

        # Set zero offset from first reading
        if self.zero_offset is None:
            self.zero_offset = wrapped_angle

        # Handle wrap-around detection
        if self.last_wrapped_angle is not None:
            angle_diff = wrapped_angle - self.last_wrapped_angle

            # Detect wrap-around (crossing ±π boundary)
            if angle_diff > np.pi:
                # Wrapped from +π to -π (backward rotation)
                self.total_rotations -= 1.0
            elif angle_diff < -np.pi:
                # Wrapped from -π to +π (forward rotation)
                self.total_rotations += 1.0

        # Calculate continuous angle
        continuous_angle = wrapped_angle + self.total_rotations * 2 * np.pi

        # Apply zero offset (subtract to make first reading = 0)
        zero_referenced_angle = continuous_angle - self.zero_offset

        # Adjust for gear ratio
        actual_angle = zero_referenced_angle / self.gear_ratio

        # Store for next iteration
        self.last_wrapped_angle = wrapped_angle

        # Convert to quaternion
        quaternion = quat_from_axis_angle(self.z_axis, actual_angle)

        return {
            'angle_radians': actual_angle,
            'quaternion': quaternion
        }


class HardwareInterface:
    """Hardware interface for excavator control system.

    Manages PWM control, IMU data, and encoder readings with background threads.
    """
    
    def __init__(self, 
                 config_file: str = "configuration_files/linear_config.yaml",
                 pump_variable: bool = False,
                 toggle_channels: bool = False, # basically tracks disabled (no IK for them)
                 # Defaults to disabled input gate checking for IK usage! Remember to enable if internal safety stop is desired.
                 input_rate_threshold: int = 0,
                 default_unset_to_zero: bool = True):
        """
        Initialize real hardware interface.
        
        Args:
            config_file: Path to PWM controller configuration
            pump_variable: Whether to use variable/static pump speed
            toggle_channels: Whether to allow usage of "toggleable" channels (i.e. tracks + center rotation)
            input_rate_threshold: Input rate threshold for PWM controller
        """
        self.config_file = config_file
        
        # Initialize PWM controller
        if PWMController is not None:
            try:
                self.pwm_controller = PWMController(
                    config_file=config_file,
                    pump_variable=pump_variable,
                    toggle_channels=toggle_channels,
                    input_rate_threshold=input_rate_threshold,
                    default_unset_to_zero=default_unset_to_zero
                )
                self.pwm_ready = True
                print("PWM controller initialized")
            except Exception as e:
                print(f"PWM controller initialization failed: {e}")
                self.pwm_controller = None
                self.pwm_ready = False
        else:
            print("PWM controller not available")
            self.pwm_controller = None
            self.pwm_ready = False
        
        # Initialize IMU reader
        self.imu_ready = False
        self.latest_imu_data = None
        self._imu_lock = threading.Lock()
        self._start_imu_reader()

        # Initialize ADC and encoder
        self.adc_ready = False
        self.latest_slew_angle = 0.0
        self.latest_slew_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._adc_lock = threading.Lock()
        self._start_adc_reader()
    
    def _check_imu_streaming(self, timeout: float = 2.0) -> bool:
        """Check if IMUs are already streaming valid data."""
        if self.usb_reader is None:
            return False
            
        start_time = time.time()
        valid_count = 0
        
        while (time.time() - start_time) < timeout:
            imu_data = self.usb_reader.read_imus()
            if imu_data is not None and len(imu_data) >= 3:
                valid_count += 1
                if valid_count >= 3:  # Got multiple valid readings
                    return True
            time.sleep(0.1)
        
        return False
    
    def _start_imu_reader(self) -> None:
        """Initialize IMU reader and start background thread."""
        if USBSerialReader is not None:
            try:
                self.usb_reader = USBSerialReader()
                
                # Check if IMUs are already streaming data
                already_streaming = self._check_imu_streaming()

                if not already_streaming:
                    # Send handshake configuration to start IMU data streaming
                    self.usb_reader.send_handshake_config()

                # Start background thread for IMU reading
                self.imu_thread = threading.Thread(
                    target=self._imu_reader_thread,
                    daemon=True
                )
                self.imu_thread.start()
                
            except Exception as e:
                print(f"IMU initialization failed: {e}")
                self.usb_reader = None
        else:
            print("IMU reader not available")
            self.usb_reader = None
            
    def _imu_reader_thread(self) -> None:
        """Background thread for continuous IMU reading."""
        next_run_time = time.perf_counter()
        read_period = 1.0 / 240.0  # 4.16ms = 240Hz

        while True:
            try:
                quaternions = self.usb_reader.read_imus()
                if quaternions is not None and len(quaternions) >= 3:
                    # Convert to numpy arrays
                    quaternion_arrays = [np.array(q, dtype=np.float32) for q in quaternions]

                    # Validate quaternion magnitudes (should be ~1.0 for unit quaternions)
                    valid_data = True
                    for q in quaternion_arrays[:3]:
                        mag = np.linalg.norm(q)
                        if mag < 0.95 or mag > 1.05:
                            valid_data = False
                            break

                    if valid_data:
                        # Atomically update latest data
                        with self._imu_lock:
                            self.latest_imu_data = quaternion_arrays[:3]  # Take first 3
                            if not self.imu_ready:
                                self.imu_ready = True
                                print("IMU data streaming")

                # Accurate timing
                next_run_time += read_period
                sleep_time = next_run_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_run_time = time.perf_counter()

            except Exception as e:
                print(f"IMU reader thread error: {e}")
                with self._imu_lock:
                    self.imu_ready = False
                time.sleep(0.1)  # Longer sleep on error
                next_run_time = time.perf_counter() + 0.1

    def _start_adc_reader(self) -> None:
        """Initialize ADC reader and start background thread."""
        try:
            self.adc = SimpleADC()
            self.encoder_tracker = EncoderTracker(gear_ratio=2.60, min_voltage=0.5, max_voltage=4.5)

            # Start background thread for ADC reading
            self.adc_thread = threading.Thread(
                target=self._adc_reader_thread,
                daemon=True
            )
            self.adc_thread.start()

        except Exception as e:
            print(f"ADC initialization failed: {e}")
            self.adc = None

    def _adc_reader_thread(self) -> None:
        """Background thread for continuous ADC reading."""
        next_run_time = time.perf_counter()
        read_period = 1.0 / 120.0  # 8.33ms = 120Hz

        while True:
            try:
                # Read only the slew encoder channel directly
                slew_voltage = self.adc.read_raw_channel("b1", 8)
                slew_data = self.encoder_tracker.update(slew_voltage)

                # Atomically update latest data
                with self._adc_lock:
                    self.latest_slew_angle = slew_data['angle_radians']
                    self.latest_slew_quat = slew_data['quaternion']
                    if not self.adc_ready:
                        self.adc_ready = True
                        print("ADC and encoder ready")

                # Accurate timing
                next_run_time += read_period
                sleep_time = next_run_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_run_time = time.perf_counter()

            except Exception as e:
                print(f"ADC reader thread error: {e}")
                with self._adc_lock:
                    self.adc_ready = False
                time.sleep(0.1)  # Longer sleep on error
                next_run_time = time.perf_counter() + 0.1

    def read_imu_data(self) -> Optional[List[np.ndarray]]:
        """Read latest IMU data with yaw calibration applied."""
        # Get latest data atomically - check ready flag inside lock
        try:
            with self._imu_lock:
                if not self.imu_ready:
                    return None

                if self.latest_imu_data is not None:
                    # Return copy to avoid race conditions
                    raw_data = [q.copy() for q in self.latest_imu_data]
                    return raw_data
                else:
                    return None

        except Exception as e:
            print(f"IMU read error: {e}")
            return None

    def read_slew_voltage(self) -> float:
        """
        Read slew encoder voltage directly from ADC.

        Returns:
            Raw voltage reading from slew encoder, or 0.0 if not ready.
        """
        try:
            if not self.adc_ready or self.adc is None:
                return 0.0
            return self.adc.read_raw_channel("b1", 8)
        except Exception as e:
            print(f"ADC read error: {e}")
            return 0.0

    def read_slew_angle(self) -> float:
        """Read latest slew angle in radians."""
        try:
            with self._adc_lock:
                if not self.adc_ready:
                    return 0.0
                return self.latest_slew_angle
        except Exception as e:
            print(f"Slew angle read error: {e}")
            return 0.0

    def read_slew_quaternion(self) -> np.ndarray:
        """Read latest slew quaternion [w, x, y, z]."""
        try:
            with self._adc_lock:
                if not self.adc_ready:
                    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return self.latest_slew_quat.copy()
        except Exception as e:
            print(f"Slew quaternion read error: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def send_pwm_commands(self, commands: Dict[str, float]) -> bool:
        """Send PWM commands by name (compat wrapper)."""
        return self.send_named_pwm_commands(commands)

    def reset(self, reset_pump: bool = False) -> None:
        """Reset hardware to safe state."""
        if self.pwm_controller is not None:
            try:
                self.pwm_controller.reset(reset_pump=reset_pump)
            except Exception as e:
                print(f"Hardware reset error: {e}")

    def send_named_pwm_commands(self, commands: Dict[str, float], *,
                                unset_to_zero: Optional[bool] = None,
                                one_shot_pump_override: bool = True) -> bool:
        """Convenience method: send name-based PWM commands.

        Args:
            commands: Mapping from channel name to command value [-1, 1]. Unknown names ignored.
            unset_to_zero: If None, use controller default; otherwise override per call.
            one_shot_pump_override: If True, a provided 'pump' value only applies for this update.
        """
        if not self.pwm_ready or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.update_named(commands,
                                             unset_to_zero=unset_to_zero,
                                             one_shot_pump_override=one_shot_pump_override)
            return True
        except Exception as e:
            print(f"PWM named command error: {e}")
            return False

    def get_pwm_channel_names(self, include_pump: bool = True) -> List[str]:
        """Expose configured PWM channel names for user UIs/hints."""
        if self.pwm_controller is None:
            return []
        try:
            return self.pwm_controller.get_channel_names(include_pump=include_pump)
        except Exception:
            return []
    
    def is_hardware_ready(self) -> bool:
        """Check if hardware is ready."""
        return self.pwm_ready and self.imu_ready and self.adc_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get hardware status for monitoring."""
        status = {
            'pwm_ready': self.pwm_ready,
            'imu_ready': self.imu_ready,
            'adc_ready': self.adc_ready,
            'latest_imu_timestamp': time.time() if self.latest_imu_data is not None else None,
            'slew_angle': self.latest_slew_angle
        }

        return status

    def reload_config(self) -> bool:
        """Reload PWM controller configuration from config file."""
        if self.pwm_controller is None:
            print("Cannot reload config: PWM controller not initialized")
            return False

        try:
            success = self.pwm_controller.reload_config(self.config_file)
            if success:
                print(f"Configuration reloaded from {self.config_file}")
            return success
        except Exception as e:
            print(f"Error reloading configuration: {e}")
            return False
