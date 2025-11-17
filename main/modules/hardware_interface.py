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
from pathlib import Path


import yaml

from .PCA9685_controller import PWMController
from .usb_serial_reader import USBSerialReader
from .ADC import SimpleADC
from .quaternion_math import quat_from_axis_angle, wrap_to_pi


def _safe_hardware_operation(func):
    """Decorator that ensures hardware is safely reset (pump stopped, PWM zeroed) on any exception.

    Critical for safety: If any hardware read fails, we must stop the machine before crashing.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            # CRITICAL SAFETY: Stop pump and reset all PWM channels before propagating error
            print(f"[SAFETY] Error in {func.__name__}: {e}")
            print(f"[SAFETY] Stopping pump and resetting all PWM channels...")
            try:
                self.reset(reset_pump=True)
            except Exception as reset_error:
                print(f"[SAFETY] Warning: Reset failed: {reset_error}")
            raise
    return wrapper


class EncoderTracker:
    def __init__(self, gear_ratio, min_voltage=0.5, max_voltage=4.5, flip_direction=False):
        """
        Initialize encoder tracker with simple continuous angle tracking.

        :param gear_ratio: Gear ratio (encoder_rotations / actual_rotations)
        :param min_voltage: Minimum voltage of encoder range
        :param max_voltage: Maximum voltage of encoder range
        :param flip_direction: If True, flip encoder direction (needed when gearing causes mechanical inversion)
        """
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.voltage_range = max_voltage - min_voltage
        self.gear_ratio = gear_ratio
        self.flip_direction = flip_direction
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

        # Adjust for gear ratio and flip direction if needed
        # Flip is required when encoder gearing causes mechanical inversion:
        # physical slew rotates CW → gears cause encoder to rotate CCW
        actual_angle = zero_referenced_angle / self.gear_ratio
        if self.flip_direction:
            actual_angle = -actual_angle

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
                 config_file: str = "configuration_files/servo_config.yaml",
                 pump_variable: bool = False,
                 toggle_channels: bool = False, # basically tracks disabled (no IK for them)
                 # Defaults to disabled input gate checking for IK usage! Remember to enable if internal safety stop is desired.
                 input_rate_threshold: int = 0,
                 default_unset_to_zero: bool = True,
                 perf_enabled: bool = False,
                 imu_expected_hz: Optional[float] = None,
                 general_config_path: str = "configuration_files/general_config.yaml"):
        """
        Initialize real hardware interface.
        
        Args:
            config_file: Path to PWM controller configuration
            pump_variable: Whether to use variable/static pump speed
            toggle_channels: Whether to allow usage of "toggleable" channels (i.e. tracks + center rotation)
            input_rate_threshold: Input rate threshold for PWM controller
        """
        self.config_file = config_file
        self._general_config_path = general_config_path
        self._general_config = self._load_general_config(general_config_path)
        
        # Initialize PWM controller
        if PWMController is not None:
            try:
                self.pwm_controller = PWMController(
                    config_file=config_file,
                    pump_variable=self._g('pwm.pump_variable', pump_variable),
                    toggle_channels=self._g('pwm.toggle_channels', toggle_channels),
                    input_rate_threshold=self._g('pwm.input_rate_threshold', input_rate_threshold),
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
        
        # Stop/coordination event for background threads (IMU/ADC)
        self._stop_event = threading.Event()

        # Perf tracking (opt-in and very low overhead when disabled)
        # Initialize BEFORE starting any reader threads to avoid races
        self._perf_enabled = bool(perf_enabled)
        self._perf_lock = threading.Lock()
        # IMU perf
        self._imu_rate_count = 0
        self._imu_rate_window_start = time.perf_counter()
        self._imu_hz = 0.0
        self._imu_n = 0  # wallclock interval samples
        self._imu_mean = 0.0
        self._imu_m2 = 0.0
        self._imu_min = float('inf')
        self._imu_max = 0.0
        self._imu_last_wall = None
        # Device timestamp derived intervals (if available)
        self._imu_dev_n = 0
        self._imu_dev_mean = 0.0
        self._imu_dev_m2 = 0.0
        self._imu_dev_min = float('inf')
        self._imu_dev_max = 0.0
        self._imu_last_dev_ts = None
        # ADC perf
        self._adc_rate_count = 0
        self._adc_rate_window_start = time.perf_counter()
        self._adc_hz = 0.0
        self._adc_n = 0
        self._adc_mean = 0.0
        self._adc_m2 = 0.0
        self._adc_min = float('inf')
        self._adc_max = 0.0
        self._adc_last_wall = None

        # Initialize IMU reader
        self.imu_ready = False
        self.latest_imu_data = None
        self.latest_imu_pitch = None  # radians, if available from stream
        self._imu_lock = threading.Lock()
        # IMU target SR from config unless explicitly provided
        cfg_imu_hz = self._g('rates.imu_hz')
        if cfg_imu_hz is None:
            raise ValueError("rates.imu_hz must be specified in general config file")
        self._imu_expected_hz = float(imu_expected_hz) if imu_expected_hz else float(cfg_imu_hz)
        # Debug telemetry (gated): when enabled, keep latest IMU gyro data for logging
        self._debug_telemetry_enabled = False
        self.latest_imu_gyro = None  # list of [gx, gy, gz] per IMU
        self._imu_last_device_ts = None
        self.imu_thread = None
        self._start_imu_reader()

        # Initialize ADC and encoder
        self.adc_ready = False
        self.latest_slew_angle = 0.0
        self.latest_slew_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._adc_lock = threading.Lock()
        # ADC/encoder expected rate from config
        cfg_adc_hz = self._g('rates.adc_hz')
        if cfg_adc_hz is None:
            raise ValueError("rates.adc_hz must be specified in general config file")
        self._adc_expected_hz = float(cfg_adc_hz)
        self.adc_thread = None
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
                # Initialize USBSerialReader with basic parameters
                # Data format: CSV with [w,x,y,z,gx,gy,gz] per IMU
                self.usb_reader = USBSerialReader(baud_rate=115200, timeout=1.0, simulation_mode=False)

                # Send handshake to configure the device
                # Request expected sample rate explicitly (firmware may clamp/ignore)
                # Also pass optional LPF and QMODE from config if provided
                hs_kwargs = {
                    'sample_rate': int(self._imu_expected_hz)
                }
                lpf_enabled = self._g('imu.lpf_enabled', None)
                lpf_alpha = self._g('imu.lpf_alpha', None)
                qmode = self._g('imu.qmode', None)
                if lpf_enabled is not None:
                    hs_kwargs['lpf_enabled'] = int(bool(lpf_enabled))
                if lpf_alpha is not None:
                    hs_kwargs['lpf_alpha'] = float(lpf_alpha)
                if qmode is not None:
                    hs_kwargs['qmode'] = str(qmode)
                self.usb_reader.send_handshake_config(**hs_kwargs)

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
        # Aim to match the device sample rate to avoid unnecessary polling
        next_run_time = time.perf_counter()
        read_period = 1.0 / max(1.0, self._imu_expected_hz)
        while not self._stop_event.is_set():
            try:
                # Read IMU data - returns list of [w,x,y,z,gx,gy,gz] arrays
                imu_packets = self.usb_reader.read_imus()
                if imu_packets is not None and len(imu_packets) >= 3:
                    # Extract only the quaternion portion [w,x,y,z] from each IMU packet
                    # Data format is [w, x, y, z, gx, gy, gz] (7 values per IMU)
                    quat_only = [np.array(pkt[:4], dtype=np.float32) for pkt in imu_packets]
                    if self._debug_telemetry_enabled:
                        gyro_only = [np.array(pkt[4:7], dtype=np.float32) for pkt in imu_packets]
                        with self._imu_lock:
                            self.latest_imu_gyro = [g.copy() for g in gyro_only]
                            self._imu_last_device_ts = getattr(self.usb_reader, 'last_timestamp_us', None)

                    # Validate quaternion magnitudes (should be ~1.0 for unit quaternions)
                    valid_data = True
                    for q in quat_only[:3]:
                        mag = np.linalg.norm(q)
                        if mag < 0.95 or mag > 1.05:
                            valid_data = False
                            break

                    if valid_data:
                        # Atomically update latest data
                        with self._imu_lock:
                            self.latest_imu_data = quat_only[:3]  # Take first 3 IMUs
                            # Note: pitch can be computed from quaternion if needed
                            # pitch = arcsin(2*(w*y - z*x))
                            if not self.imu_ready:
                                self.imu_ready = True
                                print("IMU data streaming (CSV)")

                        # Perf: count sample and interval stats (minimal, gated)
                        if self._perf_enabled:
                            now = time.perf_counter()
                            # Wallclock intervals
                            if self._imu_last_wall is not None:
                                interval = max(0.0, now - self._imu_last_wall)
                                with self._perf_lock:
                                    self._imu_n += 1
                                    # Welford update
                                    delta = interval - self._imu_mean
                                    self._imu_mean += delta / self._imu_n
                                    self._imu_m2 += delta * (interval - self._imu_mean)
                                    self._imu_min = interval if interval < self._imu_min else self._imu_min
                                    self._imu_max = interval if interval > self._imu_max else self._imu_max
                            self._imu_last_wall = now

                            # Device timestamp derived intervals (if present)
                            dev_ts = getattr(self.usb_reader, 'last_timestamp_us', None)
                            if isinstance(dev_ts, (int, float)):
                                if self._imu_last_dev_ts is not None and dev_ts > self._imu_last_dev_ts:
                                    dt_ms = (dev_ts - self._imu_last_dev_ts) / 1000.0
                                    with self._perf_lock:
                                        self._imu_dev_n += 1
                                        ddelta = dt_ms - self._imu_dev_mean
                                        self._imu_dev_mean += ddelta / self._imu_dev_n
                                        self._imu_dev_m2 += ddelta * (dt_ms - self._imu_dev_mean)
                                        self._imu_dev_min = dt_ms if dt_ms < self._imu_dev_min else self._imu_dev_min
                                        self._imu_dev_max = dt_ms if dt_ms > self._imu_dev_max else self._imu_dev_max
                                self._imu_last_dev_ts = dev_ts

                            # Rate over a short window
                            with self._perf_lock:
                                self._imu_rate_count += 1
                                elapsed = now - self._imu_rate_window_start
                                if elapsed >= 0.5:
                                    self._imu_hz = self._imu_rate_count / elapsed
                                    self._imu_rate_count = 0
                                    self._imu_rate_window_start = now

                # Accurate timing to match expected device SR
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
                time.sleep(0.1)  # Back off on error

    def _start_adc_reader(self) -> None:
        """Initialize ADC reader and start background thread."""
        try:
            self.adc = SimpleADC()
            # flip_direction=True because slew encoder gearing inverts rotation direction
            self.encoder_tracker = EncoderTracker(gear_ratio=2.60, min_voltage=0.5, max_voltage=4.5, flip_direction=True)

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
        read_period = 1.0 / max(1.0, self._adc_expected_hz)

        while not self._stop_event.is_set():
            try:
                # Read only the slew encoder channel directly (with EMA filtering)
                slew_voltage = self.adc.read_channel("b1", 8)
                slew_data = self.encoder_tracker.update(slew_voltage)

                # Atomically update latest data
                with self._adc_lock:
                    self.latest_slew_angle = slew_data['angle_radians']
                    self.latest_slew_quat = slew_data['quaternion']
                    if not self.adc_ready:
                        self.adc_ready = True
                        print("ADC and encoder ready")

                # Perf: intervals and rate (gated)
                if self._perf_enabled:
                    now = time.perf_counter()
                    if self._adc_last_wall is not None:
                        interval = max(0.0, now - self._adc_last_wall)
                        with self._perf_lock:
                            self._adc_n += 1
                            delta = interval - self._adc_mean
                            self._adc_mean += delta / self._adc_n
                            self._adc_m2 += delta * (interval - self._adc_mean)
                            self._adc_min = interval if interval < self._adc_min else self._adc_min
                            self._adc_max = interval if interval > self._adc_max else self._adc_max
                    self._adc_last_wall = now

                    with self._perf_lock:
                        self._adc_rate_count += 1
                        elapsed = now - self._adc_rate_window_start
                        if elapsed >= 0.5:
                            self._adc_hz = self._adc_rate_count / elapsed
                            self._adc_rate_count = 0
                            self._adc_rate_window_start = now

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

    @_safe_hardware_operation
    def read_imu_data(self) -> Optional[List[np.ndarray]]:
        """Read latest IMU data with yaw calibration applied.

        Raises on error instead of returning None (SAFETY: PWM reset + pump stopped before raising)
        """
        # Get latest data atomically - check ready flag inside lock
        with self._imu_lock:
            if not self.imu_ready:
                raise RuntimeError("IMU not ready - cannot read IMU data")

            if self.latest_imu_data is not None:
                # Return copy to avoid race conditions
                raw_data = [q.copy() for q in self.latest_imu_data]
                return raw_data
            else:
                raise RuntimeError("IMU data is None despite ready flag being set")

    @_safe_hardware_operation
    def read_imu_gyro(self) -> Optional[List[np.ndarray]]:
        """Read latest IMU gyro data [gx, gy, gz] per IMU (deg/s).

        Requires debug telemetry to be enabled.
        """
        with self._imu_lock:
            if not self.imu_ready:
                raise RuntimeError("IMU not ready - cannot read gyro data")
            if not self._debug_telemetry_enabled:
                raise RuntimeError("Debug telemetry disabled - gyro data not captured")
            if self.latest_imu_gyro is None:
                raise RuntimeError("IMU gyro data is None")
            return [g.copy() for g in self.latest_imu_gyro]

    @_safe_hardware_operation
    def read_imu_pitch(self) -> Optional[List[float]]:
        """Read latest IMU pitch angles (radians) if provided by the serial stream.

        Raises on error instead of returning None (SAFETY: PWM reset + pump stopped before raising)
        """
        with self._imu_lock:
            if not self.imu_ready:
                raise RuntimeError("IMU not ready - cannot read pitch data")
            if self.latest_imu_pitch is None:
                raise RuntimeError("IMU pitch data is None (may not be streamed by device)")
            return list(self.latest_imu_pitch)

    @_safe_hardware_operation
    def read_slew_voltage(self) -> float:
        """
        Read slew encoder voltage from ADC (with EMA filtering).

        Returns:
            Filtered voltage reading from slew encoder

        Raises:
            RuntimeError: If ADC is not ready or read fails (SAFETY: PWM reset + pump stopped before raising)
        """
        if not self.adc_ready or self.adc is None:
            raise RuntimeError("ADC not ready - cannot read slew voltage")
        return self.adc.read_channel("b1", 8)

    @_safe_hardware_operation
    def read_slew_angle(self) -> float:
        """Read latest slew angle in radians.

        Raises:
            RuntimeError: If ADC is not ready (SAFETY: PWM reset + pump stopped before raising)
        """
        with self._adc_lock:
            if not self.adc_ready:
                raise RuntimeError("ADC not ready - cannot read slew angle")
            return self.latest_slew_angle

    @_safe_hardware_operation
    def read_slew_quaternion(self) -> np.ndarray:
        """Read latest slew quaternion [w, x, y, z].

        Raises:
            RuntimeError: If ADC is not ready (SAFETY: PWM reset + pump stopped before raising)
        """
        with self._adc_lock:
            if not self.adc_ready:
                raise RuntimeError("ADC not ready - cannot read slew quaternion")
            return self.latest_slew_quat.copy()

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

    def shutdown(self) -> None:
        """Gracefully stop background threads and close I/O resources."""
        try:
            self._stop_event.set()
        except Exception:
            pass

        # Join IMU/ADC threads if they were started
        try:
            if getattr(self, 'imu_thread', None) is not None:
                self.imu_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if getattr(self, 'adc_thread', None) is not None:
                self.adc_thread.join(timeout=1.0)
        except Exception:
            pass

        # Close serial connection if present
        try:
            if hasattr(self, 'usb_reader') and self.usb_reader:
                self.usb_reader.close()
        except Exception:
            pass

        # Mark sensors not ready
        with self._imu_lock:
            self.imu_ready = False
        with self._adc_lock:
            self.adc_ready = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

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
    
    def set_debug_telemetry_enabled(self, enabled: bool) -> None:
        """Enable or disable debug telemetry capture (e.g., IMU gyro)."""
        self._debug_telemetry_enabled = bool(enabled)

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

    def reload_general_config(self) -> bool:
        """Reload general YAML configuration and apply rate settings to readers."""
        try:
            cfg = self._load_general_config(self._general_config_path)
            self._general_config = cfg
            # Apply rates
            self._imu_expected_hz = float(self._g('rates.imu_hz', self._imu_expected_hz))
            self._adc_expected_hz = float(self._g('rates.adc_hz', self._adc_expected_hz))
            # Update handshake for IMU if possible
            try:
                if self.usb_reader is not None:
                    self.usb_reader.send_handshake_config(sample_rate=int(self._imu_expected_hz))
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"Error reloading general config: {e}")
            return False

    def _load_general_config(self, path: str) -> Dict[str, Any]:
        """Load general configuration YAML file if available; return dict."""
        try:
            p = Path(path)
            if not p.exists():
                return {}
            with p.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    return {}
                return data
        except Exception:
            return {}

    def _g(self, dotted: str, default=None):
        """Get nested config value from general config using dotted path."""
        cur = self._general_config
        try:
            for part in dotted.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur
        except Exception:
            return default

    # ------------------------
    # Perf helpers (opt-in)
    # ------------------------
    def set_perf_enabled(self, enabled: bool) -> None:
        self._perf_enabled = bool(enabled)

    def reset_perf_stats(self) -> None:
        with self._perf_lock:
            # IMU
            self._imu_rate_count = 0
            self._imu_rate_window_start = time.perf_counter()
            self._imu_hz = 0.0
            self._imu_n = 0
            self._imu_mean = 0.0
            self._imu_m2 = 0.0
            self._imu_min = float('inf')
            self._imu_max = 0.0
            self._imu_last_wall = None
            self._imu_dev_n = 0
            self._imu_dev_mean = 0.0
            self._imu_dev_m2 = 0.0
            self._imu_dev_min = float('inf')
            self._imu_dev_max = 0.0
            self._imu_last_dev_ts = None
            # ADC
            self._adc_rate_count = 0
            self._adc_rate_window_start = time.perf_counter()
            self._adc_hz = 0.0
            self._adc_n = 0
            self._adc_mean = 0.0
            self._adc_m2 = 0.0
            self._adc_min = float('inf')
            self._adc_max = 0.0
            self._adc_last_wall = None

    def get_perf_stats(self) -> Dict[str, Any]:
        if not self._perf_enabled:
            return {}
        with self._perf_lock:
            def pack_wall(n, mean_s, m2, min_s, max_s):
                if n > 1:
                    var = m2 / (n - 1)
                    std_ms = (var ** 0.5) * 1000.0
                else:
                    std_ms = 0.0
                return {
                    'avg_interval_ms': float(mean_s * 1000.0),
                    'std_interval_ms': float(std_ms),
                    'min_interval_ms': float(min_s * 1000.0 if min_s != float('inf') else 0.0),
                    'max_interval_ms': float(max_s * 1000.0),
                }

            imu_wall = pack_wall(self._imu_n, self._imu_mean, self._imu_m2, self._imu_min, self._imu_max)
            if self._imu_dev_n > 1:
                dev_var = self._imu_dev_m2 / (self._imu_dev_n - 1)
                dev_std = dev_var ** 0.5
            else:
                dev_std = 0.0
            imu_dev = {
                'dev_avg_interval_ms': float(self._imu_dev_mean),
                'dev_std_interval_ms': float(dev_std),
                'dev_min_interval_ms': float(self._imu_dev_min if self._imu_dev_min != float('inf') else 0.0),
                'dev_max_interval_ms': float(self._imu_dev_max),
            }

            adc_wall = pack_wall(self._adc_n, self._adc_mean, self._adc_m2, self._adc_min, self._adc_max)

            return {
                'imu': {
                    'hz': float(self._imu_hz),
                    **imu_wall,
                    **imu_dev,
                    'samples': int(self._imu_n),
                },
                'adc': {
                    'hz': float(self._adc_hz),
                    **adc_wall,
                    'samples': int(self._adc_n),
                }
            }
