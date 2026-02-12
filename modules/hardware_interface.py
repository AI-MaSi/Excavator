#!/usr/bin/env python3
"""
Hardware Interface

...

Usage:
    ...
"""



import os
import time
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading
import yaml

from .PCA9685_controller import PWMController
from .usb_serial_reader import USBSerialReader
from .ADC import SimpleADC
from .quaternion_math import quat_from_axis_angle, wrap_to_pi
from .perf_tracker import IntervalTracker
from .rt_utils import apply_rt_to_thread, SCHED_FIFO


def _load_control_config(path: str = "configuration_files/control_config.yaml") -> Dict[str, Any]:
    """Load control configuration YAML file."""
    try:
        p = Path(path)
        if not p.exists():
            # Try relative to this module's parent directory
            p = Path(__file__).parent.parent / path
        if not p.exists():
            return {}
        with p.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class ReadyState:
    """Hardware subsystem readiness states."""
    PENDING = "pending"  # Still initializing, may become ready
    READY = "ready"      # Working normally
    FAULT = "fault"      # Failed permanently, unrecoverable


class HardwareFaultError(Exception):
    """Raised when a hardware subsystem has faulted."""
    def __init__(self, subsystem: str, reason: str):
        self.subsystem = subsystem
        self.reason = reason
        super().__init__(f"{subsystem} fault: {reason}")


def _safe_hardware_operation(func):
    """Decorator that ensures hardware is safely reset (pump stopped, PWM zeroed) on any exception.

    Critical for safety: If any hardware read fails, we must stop the machine before crashing.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            # CRITICAL SAFETY: Stop pump and reset all PWM channels before propagating error
            self.logger.critical(f"Error in {func.__name__}: {e}")
            self.logger.critical("Stopping pump and resetting all PWM channels...")
            try:
                self.reset(reset_pump=True)
            except Exception as reset_error:
                self.logger.error(f"Reset failed: {reset_error}")
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
                 config_file: str = "configuration_files/servo_config_200.yaml",
                 pump_variable: bool = False,
                 toggle_channels: bool = False, # basically tracks disabled (no IK for them)
                 # Defaults to disabled input gate checking for IK usage! Remember to enable if internal safety stop is desired.
                 input_rate_threshold: int = 0,
                 stale_timeout_s: float = 0.0,
                 default_unset_to_zero: bool = True,
                 watchdog_channel: Optional[int] = None,
                 watchdog_toggle_hz: float = 0.0,
                 cleanup_disable_osc: bool = True,
                 perf_enabled: bool = False,
                 imu_expected_hz: Optional[float] = None,
                 imu_gyro_dps: Optional[int] = None,
                 imu_ahrs_gain: Optional[float] = None,
                 imu_accel_rejection: Optional[float] = None,
                 imu_recovery_s: Optional[float] = None,
                 imu_offset_s: Optional[float] = None,
                 adc_sample_hz: float = 200.0,
                 adc_channels: Optional[List[Any]] = None,
                 log_level: str = "INFO",
                 enable_pwm: bool = True,
                 enable_imu: bool = True,
                 enable_adc: bool = True,
                 start_imu_reader: bool = True,
                 start_adc_reader: bool = True,
                 imu_rt_priority: int = 0,
                 adc_rt_priority: int = 0,
                 imu_cpu_core: Optional[int] = None,
                 adc_cpu_core: Optional[int] = None,
                 pwm_frequency: Optional[int] = None):
        """
        Initialize real hardware interface.

        Args:
            config_file: Path to PWM controller configuration
            pump_variable: Whether to use variable/static pump speed
            toggle_channels: Whether to allow usage of "toggleable" channels (i.e. tracks + center rotation)
            input_rate_threshold: Input rate threshold for PWM controller safety monitoring.
                                  If > 0, enables rate checking and resets PWM if rate drops below threshold.
            stale_timeout_s: If > 0, reject commands older than this (requires command_ts in send_named_pwm_commands).
                             Also triggers PWM reset if no commands received within this timeout.
            watchdog_channel: PWM channel (0-15) to use for external hardware watchdog signal.
                              If set, this channel toggles at watchdog_toggle_hz during normal operation.
                              External watchdog relay detects missing pulses and cuts power - hardware-level failsafe.
            watchdog_toggle_hz: Frequency (Hz) to toggle the watchdog channel. Typical: 5-20 Hz.
                                Must be set along with watchdog_channel for watchdog to function.
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            enable_pwm: If False, skip PWM controller initialization (sensor-only mode)
            enable_imu: If False, skip IMU initialization and threads
            enable_adc: If False, skip ADC/encoder initialization and threads
            start_imu_reader: Auto-start IMU background thread when IMU is enabled
            start_adc_reader: Auto-start ADC background thread when ADC is enabled
            imu_rt_priority: RT priority for IMU thread (0 = normal, 1-89 = SCHED_FIFO).
                             Use lower than control loop priority (e.g., 50 if control is 70).
            adc_rt_priority: RT priority for ADC thread (0 = normal, 1-89 = SCHED_FIFO).
            imu_cpu_core: CPU core to pin IMU thread to (None = no pinning, 0-3 on Pi 5).
            adc_cpu_core: CPU core to pin ADC thread to (None = no pinning, 0-3 on Pi 5).
            adc_sample_hz: Target ADC sampling frequency for the background thread
            adc_channels: List of ADC channels to sample in the background thread.
                           Accepts sensor names from ADCConfig or (board, channel) tuples.
            pwm_frequency: PCA9685 PWM signal frequency in Hz (None = use default 50Hz).
        """
        # Setup logger first
        self.logger = logging.getLogger(f"{__name__}.HardwareInterface")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config_file = config_file
        self._enable_pwm = bool(enable_pwm)
        self._enable_imu = bool(enable_imu)
        self._enable_adc = bool(enable_adc)
        self._auto_start_imu = bool(start_imu_reader)
        self._auto_start_adc = bool(start_adc_reader)
        self._imu_rt_priority = int(imu_rt_priority)
        self._adc_rt_priority = int(adc_rt_priority)
        self._imu_cpu_core = imu_cpu_core
        self._adc_cpu_core = adc_cpu_core
        self._adc_channel_requests = list(adc_channels) if adc_channels is not None else None
        self._adc_channel_plan = None  # Resolved plan of dicts {board, channel, name, is_slew}

        # IMU AHRS settings - load from config file, with constructor overrides
        _cfg = _load_control_config()
        _imu_cfg = _cfg.get('imu', {})
        self._imu_gyro_dps = int(imu_gyro_dps if imu_gyro_dps is not None else _imu_cfg.get('gyro_dps', 500))
        self._imu_ahrs_gain = float(imu_ahrs_gain if imu_ahrs_gain is not None else _imu_cfg.get('ahrs_gain', 0.5))
        self._imu_accel_rejection = float(imu_accel_rejection if imu_accel_rejection is not None else _imu_cfg.get('accel_rejection', 10.0))
        self._imu_recovery_s = float(imu_recovery_s if imu_recovery_s is not None else _imu_cfg.get('recovery_s', 1.0))
        self._imu_offset_s = float(imu_offset_s if imu_offset_s is not None else _imu_cfg.get('offset_s', 1.0))

        # Initialize PWM controller
        # Use tri-state: PENDING -> READY or FAULT
        self._pwm_state = ReadyState.PENDING
        self._pwm_fault_reason: Optional[str] = None

        if PWMController is not None and self._enable_pwm:
            try:
                self.pwm_controller = PWMController(
                    config_file=config_file,
                    pump_variable=pump_variable,
                    toggle_channels=toggle_channels,
                    input_rate_threshold=input_rate_threshold,
                    stale_timeout_s=stale_timeout_s,
                    default_unset_to_zero=default_unset_to_zero,
                    watchdog_channel=watchdog_channel,
                    watchdog_toggle_hz=watchdog_toggle_hz,
                    cleanup_disable_osc=cleanup_disable_osc,
                    perf_enabled=perf_enabled,
                    pwm_frequency=pwm_frequency,
                )
                self._pwm_state = ReadyState.READY
                self.logger.info("PWM controller initialized")
            except Exception as e:
                self.logger.error(f"PWM controller initialization failed: {e}")
                self.pwm_controller = None
                self._pwm_state = ReadyState.FAULT
                self._pwm_fault_reason = str(e)
        elif not self._enable_pwm:
            self.logger.info("PWM controller disabled by configuration")
            self.pwm_controller = None
            self._pwm_state = ReadyState.READY  # Disabled counts as "ready"
        else:
            self.logger.warning("PWM controller not available")
            self.pwm_controller = None
            self._pwm_state = ReadyState.FAULT
            self._pwm_fault_reason = "PWMController module not available"
        
        # Stop/coordination event for background threads (IMU/ADC)
        self._stop_event = threading.Event()

        # Perf tracking (opt-in and very low overhead when disabled)
        # Initialize BEFORE starting any reader threads to avoid races
        self._perf_enabled = bool(perf_enabled)
        # IMU interval tracker (wallclock-based)
        self._imu_tracker = IntervalTracker(enabled=perf_enabled)
        # IMU device timestamp tracker (separate, for hardware-reported intervals)
        self._imu_dev_tracker = IntervalTracker(enabled=perf_enabled)
        self._imu_last_dev_ts = None  # Track last device timestamp for delta
        # ADC interval tracker
        self._adc_tracker = IntervalTracker(enabled=perf_enabled)

        # Initialize IMU reader
        # Use tri-state: PENDING -> READY or FAULT
        # If disabled, mark as READY (not needed = OK)
        self._imu_state = ReadyState.PENDING if self._enable_imu else ReadyState.READY
        self._imu_fault_reason: Optional[str] = None
        self.latest_imu_data = None
        self.latest_imu_pitch = None  # radians, if available from stream
        self._imu_lock = threading.Lock()
        self._imu_expected_hz = None
        self.usb_reader = None
        # IMU target sample rate (from config or constructor override)
        if self._enable_imu:
            _cfg = _load_control_config()
            _imu_cfg = _cfg.get('imu', {})
            self._imu_expected_hz = float(imu_expected_hz if imu_expected_hz is not None else _imu_cfg.get('sample_rate', 200))
        # Debug telemetry (gated): when enabled, keep latest IMU gyro data for logging
        self._debug_telemetry_enabled = False
        self.latest_imu_gyro = None  # list of [gx, gy, gz] per IMU
        self._imu_last_device_ts = None
        self.imu_thread = None
        if self._enable_imu and self._auto_start_imu:
            self._start_imu_reader()

        # Initialize ADC and encoder
        # Use tri-state: PENDING -> READY or FAULT
        # If disabled, mark as READY (not needed = OK)
        self._adc_state = ReadyState.PENDING if self._enable_adc else ReadyState.READY
        self._adc_fault_reason: Optional[str] = None
        self.latest_slew_angle = 0.0
        self.latest_slew_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._adc_lock = threading.Lock()
        self._adc_expected_hz = None
        self._latest_adc_readings = {}
        self._latest_adc_timestamp = None
        self.adc_thread = None
        self.adc = None
        self.encoder_tracker = None
        # ADC/encoder expected rate (constructor arg, no config fallback)
        if self._enable_adc:
            self._adc_expected_hz = float(adc_sample_hz)
            # Initialize ADC hardware even if the thread is not started so callers can read synchronously.
            if self._ensure_adc_initialized():
                if self._auto_start_adc:
                    self._start_adc_reader()
                else:
                    self._adc_state = ReadyState.READY
    
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

    def _check_faults(self) -> None:
        """Check for hardware faults and raise HardwareFaultError if any subsystem has faulted.

        Call this before waiting on hardware to avoid infinite waits on unrecoverable failures.
        """
        if self._pwm_state == ReadyState.FAULT:
            raise HardwareFaultError("PWM", self._pwm_fault_reason or "Unknown error")
        if self._imu_state == ReadyState.FAULT:
            raise HardwareFaultError("IMU", self._imu_fault_reason or "Unknown error")
        if self._adc_state == ReadyState.FAULT:
            raise HardwareFaultError("ADC", self._adc_fault_reason or "Unknown error")
    
    def _start_imu_reader(self) -> None:
        """Initialize IMU reader and start background thread."""
        if not self._enable_imu:
            self.logger.debug("IMU disabled; _start_imu_reader skipped")
            return
        if self.imu_thread is not None and self.imu_thread.is_alive():
            return
        if USBSerialReader is not None:
            try:
                # Initialize USBSerialReader with basic parameters
                # Data format: CSV with [w,x,y,z,gx,gy,gz] per IMU
                self.usb_reader = USBSerialReader(baud_rate=115200, timeout=1.0, simulation_mode=False)

                # Send config and wait for acknowledgment
                self.usb_reader.send_config(
                    sample_rate=int(self._imu_expected_hz),
                    gyro_dps=self._imu_gyro_dps,
                    gain=self._imu_ahrs_gain,
                    accel_rejection=self._imu_accel_rejection,
                    recovery_s=self._imu_recovery_s,
                    offset_s=self._imu_offset_s,
                )
                if not self.usb_reader.wait_for_cfg_ok(timeout_s=5.0):
                    self.logger.warning("IMU config acknowledgment not received")

                # Start background thread for IMU reading
                self.imu_thread = threading.Thread(
                    target=self._imu_reader_thread,
                    daemon=True
                )
                self.imu_thread.start()

            except Exception as e:
                self.logger.error(f"IMU initialization failed: {e}")
                self.usb_reader = None
        else:
            self.logger.warning("IMU reader not available")
            self.usb_reader = None

    def start_imu_streaming(self) -> bool:
        """Public helper to start IMU thread when auto-start was disabled."""
        try:
            self._start_imu_reader()
            return self.imu_thread is not None and self.imu_thread.is_alive()
        except Exception as e:
            self.logger.error(f"Failed to start IMU streaming: {e}")
            return False
            
    def _imu_reader_thread(self) -> None:
        """Background thread for continuous IMU reading."""
        # Apply CPU core pinning if configured (must be done from within the thread)
        if self._imu_cpu_core is not None:
            try:
                os.sched_setaffinity(0, {self._imu_cpu_core})
                self.logger.info(f"IMU thread: Pinned to Core {self._imu_cpu_core}")
            except Exception as e:
                self.logger.warning(f"IMU thread: Failed to pin to Core {self._imu_cpu_core}: {e}")

        # Apply RT priority if configured (must be done from within the thread)
        if self._imu_rt_priority > 0:
            success = apply_rt_to_thread(
                priority=self._imu_rt_priority,
                policy=SCHED_FIFO,
                quiet=False
            )
            if success:
                self.logger.info(f"IMU thread: RT priority SCHED_FIFO-{self._imu_rt_priority} applied")
            else:
                self.logger.warning(f"IMU thread: Failed to apply RT priority {self._imu_rt_priority}")

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
                            if self._imu_state != ReadyState.READY:
                                self._imu_state = ReadyState.READY
                                self.logger.info("IMU data streaming (CSV)")

                        # Perf: record sample for interval/rate tracking (uses IntervalTracker)
                        self._imu_tracker.record_sample()

                        # Device timestamp derived intervals (if present)
                        dev_ts = getattr(self.usb_reader, 'last_timestamp_us', None)
                        if isinstance(dev_ts, (int, float)):
                            if self._imu_last_dev_ts is not None and dev_ts > self._imu_last_dev_ts:
                                # Convert us delta to seconds for tracker
                                dt_s = (dev_ts - self._imu_last_dev_ts) / 1_000_000.0
                                self._imu_dev_tracker.record_interval(dt_s)
                            self._imu_last_dev_ts = dev_ts

                # Accurate timing to match expected device SR
                next_run_time += read_period
                sleep_time = next_run_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_run_time = time.perf_counter()

            except Exception as e:
                self.logger.error(f"IMU reader thread error: {e}")
                with self._imu_lock:
                    self._imu_state = ReadyState.PENDING
                time.sleep(0.1)  # Back off on error

    def _ensure_adc_initialized(self) -> bool:
        """Create ADC and encoder helper if not already initialized."""
        if self.adc is None:
            try:
                self.adc = SimpleADC()
            except Exception as e:
                self.logger.error(f"ADC initialization failed: {e}")
                self.adc = None
                return False
        if self.encoder_tracker is None:
            # flip_direction=True because slew encoder gearing inverts rotation direction
            self.encoder_tracker = EncoderTracker(gear_ratio=2.60, min_voltage=0.5, max_voltage=4.5, flip_direction=True)
        if self._adc_channel_plan is None:
            self._adc_channel_plan = self._build_adc_channel_plan()
        return True

    def _build_adc_channel_plan(self) -> List[Dict[str, Any]]:
        """Resolve requested ADC channels into a concrete sampling plan."""
        if self.adc is None:
            raise RuntimeError("ADC not initialized")

        requests = self._adc_channel_requests if self._adc_channel_requests is not None else ["Slew encoder rot"]
        if not requests:
            raise ValueError("adc_channels must contain at least one channel")

        plan: List[Dict[str, Any]] = []
        sensors_cfg = getattr(self.adc, "config", None)
        for item in requests:
            if isinstance(item, str):
                if not hasattr(sensors_cfg, "sensors") or item not in sensors_cfg.sensors:
                    raise ValueError(f"Unknown ADC sensor name '{item}'")
                board_name, channel = sensors_cfg.sensors[item]['input']
                name = item
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                board_name, channel = item
                name_lookup = sensors_cfg.get_sensor_by_input(board_name, int(channel)) if hasattr(sensors_cfg, "get_sensor_by_input") else None
                name = name_lookup if name_lookup else f"{board_name}:{channel}"
            else:
                raise ValueError(f"Invalid adc_channels entry: {item}")

            is_slew = (name == "Slew encoder rot") or (board_name == "b1" and int(channel) == 8)
            plan.append({
                'board': board_name,
                'channel': int(channel),
                'name': name,
                'is_slew': is_slew,
            })

        return plan

    def _start_adc_reader(self) -> None:
        """Initialize ADC reader and start background thread."""
        if not self._enable_adc:
            self.logger.debug("ADC disabled; _start_adc_reader skipped")
            return
        if self.adc_thread is not None and self.adc_thread.is_alive():
            return
        if not self._ensure_adc_initialized():
            return
        if self._adc_expected_hz is None:
            raise ValueError("ADC expected Hz is not configured")

        # Start background thread for ADC reading
        self.adc_thread = threading.Thread(
            target=self._adc_reader_thread,
            daemon=True
        )
        self.adc_thread.start()
        # Mark not ready until first successful read
        self._adc_state = ReadyState.PENDING

    def start_adc_streaming(self) -> bool:
        """Public helper to start ADC thread when auto-start was disabled."""
        try:
            self._start_adc_reader()
            return self.adc_thread is not None and self.adc_thread.is_alive()
        except Exception as e:
            self.logger.error(f"Failed to start ADC streaming: {e}")
            return False

    def _ensure_slew_channel_configured(self) -> None:
        """Raise if slew encoder is not part of the configured ADC sampling plan."""
        if not self._enable_adc:
            raise RuntimeError("ADC disabled - slew encoder not available")
        if self._adc_channel_plan is None:
            if not self._ensure_adc_initialized():
                raise RuntimeError("ADC not initialized - cannot check slew channel")
        has_slew = any(ch.get('is_slew') for ch in (self._adc_channel_plan or []))
        if not has_slew:
            raise RuntimeError("Slew encoder channel not configured in adc_channels")

    def _adc_reader_thread(self) -> None:
        """Background thread for continuous ADC reading."""
        # Apply CPU core pinning if configured (must be done from within the thread)
        if self._adc_cpu_core is not None:
            try:
                os.sched_setaffinity(0, {self._adc_cpu_core})
                self.logger.info(f"ADC thread: Pinned to Core {self._adc_cpu_core}")
            except Exception as e:
                self.logger.warning(f"ADC thread: Failed to pin to Core {self._adc_cpu_core}: {e}")

        # Apply RT priority if configured (must be done from within the thread)
        if self._adc_rt_priority > 0:
            success = apply_rt_to_thread(
                priority=self._adc_rt_priority,
                policy=SCHED_FIFO,
                quiet=False
            )
            if success:
                self.logger.info(f"ADC thread: RT priority SCHED_FIFO-{self._adc_rt_priority} applied")
            else:
                self.logger.warning(f"ADC thread: Failed to apply RT priority {self._adc_rt_priority}")

        next_run_time = time.perf_counter()
        read_period = 1.0 / max(1.0, self._adc_expected_hz)

        while not self._stop_event.is_set():
            try:
                # Sample configured channels
                for ch in self._adc_channel_plan:
                    if ch['is_slew']:
                        # Slew encoder: 1x oversample for 200Hz support (4.16ms vs 8.32ms)
                        voltage = self.adc.read_channel_fast(ch['board'], ch['channel'], oversample=1)
                    else:
                        voltage = self.adc.read_channel(ch['board'], ch['channel'])
                    if ch['is_slew']:
                        slew_data = self.encoder_tracker.update(voltage)
                        with self._adc_lock:
                            self.latest_slew_angle = slew_data['angle_radians']
                            self.latest_slew_quat = slew_data['quaternion']
                    with self._adc_lock:
                        self._latest_adc_readings[ch['name']] = voltage
                        self._latest_adc_timestamp = time.time()
                        if self._adc_state != ReadyState.READY:
                            self._adc_state = ReadyState.READY
                            self.logger.info("ADC sampling ready")

                # Perf: record sample for interval/rate tracking (uses IntervalTracker)
                self._adc_tracker.record_sample()

                # Accurate timing
                next_run_time += read_period
                sleep_time = next_run_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_run_time = time.perf_counter()

            except Exception as e:
                self.logger.error(f"ADC reader thread error: {e}")
                with self._adc_lock:
                    self._adc_state = ReadyState.PENDING
                    self._latest_adc_readings = {}
                    self._latest_adc_timestamp = None
                time.sleep(0.1)  # Longer sleep on error
                next_run_time = time.perf_counter() + 0.1

    @_safe_hardware_operation
    def read_imu_data(self) -> Optional[List[np.ndarray]]:
        """Read latest IMU data with yaw calibration applied.

        Raises on error instead of returning None (SAFETY: PWM reset + pump stopped before raising)
        """
        # Get latest data atomically - check ready flag inside lock
        with self._imu_lock:
            if self._imu_state != ReadyState.READY:
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
            if self._imu_state != ReadyState.READY:
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
            if self._imu_state != ReadyState.READY:
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
        self._ensure_slew_channel_configured()
        with self._adc_lock:
            if self._adc_state != ReadyState.READY:
                raise RuntimeError("ADC not ready - cannot read slew voltage")
            for ch in self._adc_channel_plan:
                if ch['is_slew']:
                    name = ch['name']
                    break
            else:
                raise RuntimeError("Slew channel not in ADC plan")
            if name not in self._latest_adc_readings:
                raise RuntimeError("Slew voltage not yet sampled")
            return float(self._latest_adc_readings[name])

    @_safe_hardware_operation
    def read_slew_angle(self) -> float:
        """Read latest slew angle in radians.

        Raises:
            RuntimeError: If ADC is not ready (SAFETY: PWM reset + pump stopped before raising)
        """
        self._ensure_slew_channel_configured()
        with self._adc_lock:
            if self._adc_state != ReadyState.READY:
                raise RuntimeError("ADC not ready - cannot read slew angle")
            return self.latest_slew_angle

    @_safe_hardware_operation
    def read_slew_quaternion(self) -> np.ndarray:
        """Read latest slew quaternion [w, x, y, z].

        Raises:
            RuntimeError: If ADC is not ready (SAFETY: PWM reset + pump stopped before raising)
        """
        self._ensure_slew_channel_configured()
        with self._adc_lock:
            if self._adc_state != ReadyState.READY:
                raise RuntimeError("ADC not ready - cannot read slew quaternion")
            return self.latest_slew_quat.copy()

    def get_latest_adc_readings(self) -> Dict[str, float]:
        """Get a copy of the latest ADC readings sampled by the background thread."""
        if not self._enable_adc:
            return {}
        with self._adc_lock:
            if self._adc_state != ReadyState.READY:
                raise RuntimeError("ADC not ready - cannot get ADC readings")
            return dict(self._latest_adc_readings)

    def get_latest_adc_snapshot(self) -> Dict[str, Any]:
        """Get latest ADC readings plus their wallclock timestamp."""
        if not self._enable_adc:
            return {'readings': {}, 'timestamp': None}
        with self._adc_lock:
            if self._adc_state != ReadyState.READY:
                raise RuntimeError("ADC not ready - cannot get ADC readings")
            return {
                'readings': dict(self._latest_adc_readings),
                'timestamp': self._latest_adc_timestamp
            }

    def send_pwm_commands(self, commands: Dict[str, float]) -> bool:
        """Send PWM commands by name (compat wrapper)."""
        return self.send_named_pwm_commands(commands)

    def reset(self, reset_pump: bool = False) -> None:
        """Reset hardware to safe state."""
        if self.pwm_controller is not None:
            try:
                self.pwm_controller.reset(reset_pump=reset_pump)
            except Exception as e:
                self.logger.error(f"Hardware reset error: {e}")

    def shutdown(self) -> None:
        """Gracefully stop background threads and close I/O resources."""
        # SAFETY: Reset PWM to safe state before shutting down.
        # PWMController has its own atexit handler that should do this internally,
        # but better to call it twice than risk leaving actuators in unknown state.
        try:
            self.reset(reset_pump=True)
        except Exception:
            pass

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
            self._imu_state = ReadyState.PENDING
        with self._adc_lock:
            self._adc_state = ReadyState.PENDING

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def send_named_pwm_commands(self, commands: Dict[str, float], *,
                                unset_to_zero: Optional[bool] = None,
                                one_shot_pump_override: bool = True,
                                command_ts: Optional[float] = None) -> bool:
        """Convenience method: send name-based PWM commands.

        Args:
            commands: Mapping from channel name to command value [-1, 1]. Unknown names ignored.
            unset_to_zero: If None, use controller default; otherwise override per call.
            one_shot_pump_override: If True, a provided 'pump' value only applies for this update.
            command_ts: Optional monotonic timestamp of when command was generated (e.g., UDP receive time).
                        If provided and PWM controller has stale_timeout_s configured, commands older
                        than the timeout will be rejected for safety.
        """
        if self._pwm_state != ReadyState.READY or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.update_named(commands,
                                             unset_to_zero=unset_to_zero,
                                             one_shot_pump_override=one_shot_pump_override,
                                             command_ts=command_ts)
            return True
        except Exception as e:
            self.logger.error(f"PWM named command error: {e}")
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
        """Check if hardware is ready. Raises HardwareFaultError if any subsystem has faulted."""
        # Check for faults first - raises if any subsystem failed permanently
        self._check_faults()

        pwm_ok = (not self._enable_pwm) or (self._pwm_state == ReadyState.READY)
        imu_ok = (not self._enable_imu) or (self._imu_state == ReadyState.READY)
        adc_ok = (not self._enable_adc) or (self._adc_state == ReadyState.READY)
        return pwm_ok and imu_ok and adc_ok
    
    def get_status(self) -> Dict[str, Any]:
        """Get hardware status for monitoring."""
        status = {
            'pwm_state': self._pwm_state,
            'imu_state': self._imu_state,
            'adc_state': self._adc_state,
            'pwm_fault': self._pwm_fault_reason,
            'imu_fault': self._imu_fault_reason,
            'adc_fault': self._adc_fault_reason,
            'latest_imu_timestamp': time.time() if self.latest_imu_data is not None else None,
            'slew_angle': self.latest_slew_angle
        }

        return status

    def reload_config(self) -> bool:
        """Reload PWM controller configuration from config file."""
        if self.pwm_controller is None:
            self.logger.warning("Cannot reload config: PWM controller not initialized")
            return False

        try:
            success = self.pwm_controller.reload_config(self.config_file)
            if success:
                self.logger.info(f"Configuration reloaded from {self.config_file}")
            return success
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
            return False

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.info(f"Hardware interface log level changed to {level.upper()}")

        # Propagate to sub-modules that support logging
        if hasattr(self, 'usb_reader') and self.usb_reader and hasattr(self.usb_reader, 'set_log_level'):
            self.usb_reader.set_log_level(level)
        if hasattr(self, 'pwm_controller') and self.pwm_controller and hasattr(self.pwm_controller, 'set_log_level'):
            self.pwm_controller.set_log_level(level)
        if hasattr(self, 'adc') and self.adc and hasattr(self.adc, 'set_log_level'):
            self.adc.set_log_level(level)

    # ------------------------
    # Perf helpers (opt-in)
    # ------------------------
    def set_perf_enabled(self, enabled: bool) -> None:
        """Enable or disable performance tracking for all subsystems."""
        self._perf_enabled = bool(enabled)
        self._imu_tracker.set_enabled(enabled)
        self._imu_dev_tracker.set_enabled(enabled)
        self._adc_tracker.set_enabled(enabled)
        if self.pwm_controller is not None:
            self.pwm_controller.set_perf_enabled(enabled)

    def reset_perf_stats(self) -> None:
        """Reset all performance statistics for all subsystems."""
        self._imu_tracker.reset()
        self._imu_dev_tracker.reset()
        self._imu_last_dev_ts = None
        self._adc_tracker.reset()
        if self.pwm_controller is not None:
            self.pwm_controller.reset_perf_stats()

    def get_perf_stats(self) -> Dict[str, Any]:
        """Get performance statistics for IMU, ADC, and PWM.

        Returns:
            Dictionary with 'imu', 'adc', and 'pwm' sub-dicts containing:
            - hz: Measured sampling/update rate
            - avg_interval_ms, std_interval_ms, min_interval_ms, max_interval_ms
            - samples: Number of samples collected
            For IMU, also includes dev_* fields for device timestamp intervals.
            For PWM, includes proc_* fields for processing time stats.
        """
        if not self._perf_enabled:
            return {}

        imu_stats = self._imu_tracker.get_stats()
        imu_dev_stats = self._imu_dev_tracker.get_stats()
        adc_stats = self._adc_tracker.get_stats()

        result = {
            'imu': {
                'hz': imu_stats.get('hz', 0.0),
                'avg_interval_ms': imu_stats.get('avg_interval_ms', 0.0),
                'std_interval_ms': imu_stats.get('std_interval_ms', 0.0),
                'min_interval_ms': imu_stats.get('min_interval_ms', 0.0),
                'max_interval_ms': imu_stats.get('max_interval_ms', 0.0),
                # Device timestamp derived intervals
                'dev_avg_interval_ms': imu_dev_stats.get('avg_interval_ms', 0.0),
                'dev_std_interval_ms': imu_dev_stats.get('std_interval_ms', 0.0),
                'dev_min_interval_ms': imu_dev_stats.get('min_interval_ms', 0.0),
                'dev_max_interval_ms': imu_dev_stats.get('max_interval_ms', 0.0),
                'samples': imu_stats.get('samples', 0),
            },
            'adc': {
                'hz': adc_stats.get('hz', 0.0),
                'avg_interval_ms': adc_stats.get('avg_interval_ms', 0.0),
                'std_interval_ms': adc_stats.get('std_interval_ms', 0.0),
                'min_interval_ms': adc_stats.get('min_interval_ms', 0.0),
                'max_interval_ms': adc_stats.get('max_interval_ms', 0.0),
                'samples': adc_stats.get('samples', 0),
            }
        }

        # Include PWM stats if controller is available and has tracking enabled
        if self.pwm_controller is not None:
            pwm_stats = self.pwm_controller.get_perf_stats()
            if pwm_stats:
                result['pwm'] = pwm_stats

        return result
