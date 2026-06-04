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
from .quaternion_math import (
    quat_normalize,
    quat_multiply,
    quat_conjugate,
    quat_rotate_vector,
)
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


def _parse_offset_quaternion(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a list of 4 numbers")
    quat = quat_normalize(np.asarray(value, dtype=np.float32))
    return quat


class ReadyState:
    """Hardware subsystem readiness states."""
    PENDING = "pending"  # Still initializing, may become ready
    READY = "ready"      # Working normally
    FAULT = "fault"      # Failed permanently, unrecoverable


class _ImuSnapshot:
    """Immutable IMU data bundle published atomically by the IMU thread.

    The control-loop hot path reads this with a single attribute load (no lock).
    All fields are set once at construction; the snapshot is never mutated.
    CPython guarantees that a single STORE_ATTR / LOAD_ATTR is atomic under the
    GIL, so replacing self._imu_snapshot with a new instance is safe without an
    explicit lock on the read side.
    """
    __slots__ = (
        'imu_data', 'imu_by_role', 'base_imu_quat', 'base_imu_gyro',
        'imu_gyro', 'raw_quat', 'corrected_quat', 'device_ts',
    )

    def __init__(self, *, imu_data, imu_by_role, base_imu_quat, base_imu_gyro,
                 imu_gyro, raw_quat, corrected_quat, device_ts):
        self.imu_data = imu_data
        self.imu_by_role = imu_by_role
        self.base_imu_quat = base_imu_quat
        self.base_imu_gyro = base_imu_gyro
        self.imu_gyro = imu_gyro
        self.raw_quat = raw_quat
        self.corrected_quat = corrected_quat
        self.device_ts = device_ts


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


class HardwareInterface:
    """Hardware interface for excavator control system.

    Manages PWM control, IMU data, and optional pressure-sensor ADC readings.
    """
    
    def __init__(self,
                 config_file: str = "configuration_files/servo_config_200.yaml",
                 control_config_file: str = "configuration_files/control_config.yaml",
                 pump_auto_mode: bool = False,
                 toggle_channels: bool = False, # basically tracks disabled (no IK for them)
                 input_rate_threshold: int = 20,
                 stale_timeout_s: float = 0.5,
                 default_unset_to_zero: bool = True,
                 cleanup_disable_osc: bool = True,
                 perf_enabled: bool = False,
                 adc_sample_hz: float = 20.0,
                 adc_channels: Optional[List[Any]] = None,
                 log_level: str = "INFO",
                 enable_pwm: bool = True,
                 enable_imu: bool = True,
                 enable_adc: bool = False,
                 start_imu_reader: bool = True,
                 start_adc_reader: bool = True,
                 rt_lock_memory: bool = False,
                 usb_rt_priority: int = 0,
                 imu_rt_priority: int = 0,
                 adc_rt_priority: int = 0,
                 usb_cpu_core: Optional[int] = None,
                 imu_cpu_core: Optional[int] = None,
                 adc_cpu_core: Optional[int] = None,
                 pwm_frequency: Optional[int] = None):
        """
        Initialize real hardware interface.

        Args:
            config_file: Path to PWM controller configuration
            control_config_file: Path to controller/IMU configuration
            pump_auto_mode: Whether to use valve-activity-based auto pump speed (True) or static speed (False)
            toggle_channels: Whether to allow usage of "toggleable" channels (i.e. tracks + center rotation)
            input_rate_threshold: Minimum required PWM command rate (Hz). Resets valves to center
                                  if the control loop drops below this rate. Default 20 Hz.
            stale_timeout_s: Centers all valves if no PWM command arrives within this window (last-resort watchdog).
                             Default 0.5 s — fires if the control thread dies without calling stop().
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            enable_pwm: If False, skip PWM controller initialization (sensor-only mode)
            enable_imu: If False, skip IMU initialization and threads
            enable_adc: If False, skip pressure ADC initialization and threads
            start_imu_reader: Auto-start IMU background thread when IMU is enabled
            start_adc_reader: Auto-start ADC background thread when ADC is enabled
            rt_lock_memory: Whether to call mlockall() for RT worker threads.
            usb_rt_priority: RT priority for USB serial background reader thread (0 = normal).
            imu_rt_priority: RT priority for IMU thread (0 = normal, 1-89 = SCHED_FIFO).
            adc_rt_priority: RT priority for ADC thread (0 = normal, 1-89 = SCHED_FIFO).
            usb_cpu_core: CPU core to pin USB serial background reader thread to.
            imu_cpu_core: CPU core to pin IMU thread to (None = no pinning, 0-3 on Pi 5).
            adc_cpu_core: CPU core to pin ADC thread to (None = no pinning, 0-3 on Pi 5).
            adc_sample_hz: Target pressure ADC sampling frequency for the background thread
            adc_channels: List of ADC channels to sample in the background thread.
                           Accepts sensor names from ADCConfig or (board, channel) tuples.
            pwm_frequency: PCA9685 PWM signal frequency in Hz (None = use default 50Hz).
        """
        # Setup logger first
        self.logger = logging.getLogger(f"{__name__}.HardwareInterface")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Only add a fallback handler when no root handler exists (standalone use).
        # When run under run_hw_v2 / basicConfig, propagation to root is sufficient.
        if not logging.root.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config_file = config_file
        self.control_config_file = control_config_file
        self._enable_pwm = bool(enable_pwm)
        self._enable_imu = bool(enable_imu)
        self._enable_adc = bool(enable_adc)
        self._auto_start_imu = bool(start_imu_reader)
        self._auto_start_adc = bool(start_adc_reader)
        self._rt_lock_memory = bool(rt_lock_memory)
        self._usb_rt_priority = int(usb_rt_priority)
        self._imu_rt_priority = int(imu_rt_priority)
        self._adc_rt_priority = int(adc_rt_priority)
        self._usb_cpu_core = usb_cpu_core
        self._imu_cpu_core = imu_cpu_core
        self._adc_cpu_core = adc_cpu_core
        self._adc_channel_requests = list(adc_channels) if adc_channels is not None else None
        self._adc_channel_plan = None  # Resolved plan of dicts {board, channel, name}

        if self._enable_imu:
            self._init_imu_config()
        else:
            self._init_disabled_imu_config()

        # Initialize PWM controller
        # Use tri-state: PENDING -> READY or FAULT
        self._pwm_state = ReadyState.PENDING
        self._pwm_fault_reason: Optional[str] = None

        if PWMController is not None and self._enable_pwm:
            try:
                self.pwm_controller = PWMController(
                    config_file=config_file,
                    pump_auto_mode=pump_auto_mode,
                    toggle_channels=toggle_channels,
                    input_rate_threshold=input_rate_threshold,
                    stale_timeout_s=stale_timeout_s,
                    default_unset_to_zero=default_unset_to_zero,
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
        self.latest_imu_by_role = None
        self.latest_imu_raw_quat = None
        self.latest_imu_corrected_quat = None
        self.latest_base_imu_quat = None   # Base/slew IMU quaternion used by IK yaw extraction
        self.latest_base_imu_gyro = None   # Base IMU gyro [gx, gy, gz] deg/s
        self._imu_lock = threading.Lock()
        # Lock-free snapshot for control-loop hot path (read_imu_data / read_base_imu).
        # Written by the IMU thread via a single atomic assignment; read without any lock.
        self._imu_snapshot: Optional[_ImuSnapshot] = None
        self.usb_reader = None
        # IMU streams at a fixed 200 Hz from firmware; no rate negotiation needed.
        # Corrected gyro is always retained because it arrives in every IMU packet and
        # is used by controllers/loggers. Only the heavier raw debug quaternion copies
        # are gated by _debug_telemetry_enabled.
        self._debug_telemetry_enabled = False
        self.latest_imu_gyro = None  # list of [gx, gy, gz] per IMU
        self._imu_last_device_ts = None
        self.imu_thread = None
        if self._enable_imu and self._auto_start_imu:
            self._start_imu_reader()

        # Initialize pressure ADC
        # Use tri-state: PENDING -> READY or FAULT
        # If disabled, mark as READY (not needed = OK)
        self._adc_state = ReadyState.PENDING if self._enable_adc else ReadyState.READY
        self._adc_fault_reason: Optional[str] = None
        self._adc_lock = threading.Lock()
        self._adc_expected_hz = None
        self._latest_adc_readings = {}
        self._latest_adc_timestamp = None
        self.adc_thread = None
        self.adc = None
        # Pressure ADC expected rate (constructor arg, no config fallback)
        if self._enable_adc:
            self._adc_expected_hz = float(adc_sample_hz)
            # Initialize ADC hardware even if the thread is not started so callers can read synchronously.
            if self._ensure_adc_initialized():
                if self._auto_start_adc:
                    self._start_adc_reader()
                else:
                    self._adc_state = ReadyState.READY

    def _init_disabled_imu_config(self) -> None:
        """Install neutral IMU metadata when IMU is a hard-disabled subsystem."""
        self._imu_mapping = {}
        self._imu_chain = []
        self._imu_all_indices = []
        self._expected_imu_count = 0
        self._imu_sensor_roles = []
        self._imu_joint_roles = []
        self._imu_joint_indices = []
        self._base_imu_index = None
        self._imu_offset_inv_by_index = []
        self._base_yaw_offset_deg = 0.0
        self._base_yaw_offset_inv_quat = None

    def _init_imu_config(self) -> None:
        """Load and validate IMU mapping only when IMU support is enabled."""
        # IMU mapping and chain — AHRS params are hardcoded in Pico firmware.
        _cfg = _load_control_config(self.control_config_file)
        _imu_cfg = _cfg.get('imu', {})
        # Named IMU mapping: logical role -> physical sensor index
        _imu_mapping = _imu_cfg.get('imu_mapping')
        if _imu_mapping is None or not isinstance(_imu_mapping, dict):
            raise ValueError("imu.imu_mapping is required in control_config.yaml (dict of role -> sensor index)")
        self._imu_mapping = dict(_imu_mapping)
        _imu_chain = _imu_cfg.get('chain')
        if not isinstance(_imu_chain, list):
            _imu_chain = [
                {'joint': 'slew', 'output_index': 0, 'source': 'all', 'axis': 'z', 'extraction': 'average_z_yaw'},
                {'joint': 'lift', 'role': 'boom', 'parent_role': 'base', 'output_index': 1, 'axis': 'y', 'extraction': 'gravity_pitch_delta'},
                {'joint': 'arm', 'role': 'arm', 'parent_role': 'boom', 'output_index': 2, 'axis': 'y', 'extraction': 'gravity_pitch_delta'},
                {'joint': 'bucket', 'role': 'bucket', 'parent_role': 'arm', 'output_index': 3, 'axis': 'y', 'extraction': 'gravity_pitch_delta'},
            ]
        _mounting_offsets_cfg = _imu_cfg.get('mounting_offsets_quat', {})
        if _mounting_offsets_cfg is None:
            _mounting_offsets_cfg = {}
        if not isinstance(_mounting_offsets_cfg, dict):
            raise ValueError("imu.mounting_offsets_quat must be a dict of role -> quaternion")
        self._imu_chain = _imu_chain
        # Sensor roles in the order the controller's canonical extraction expects.
        _sensor_roles = []

        def _add_sensor_role(role):
            if role and role != 'all' and role in _imu_mapping and role not in _sensor_roles:
                _sensor_roles.append(role)

        for item in _imu_chain:
            if not isinstance(item, dict):
                continue
            _add_sensor_role(item.get('parent_role'))
            _add_sensor_role(item.get('role'))

        _joint_roles = []
        for item in _imu_chain:
            if not isinstance(item, dict) or not item.get('role'):
                continue
            role = item['role']
            if role not in _imu_mapping:
                if bool(item.get('optional', False)):
                    continue
                raise ValueError(f"imu.imu_mapping missing required joint role '{role}'")
            if role not in _joint_roles:
                _joint_roles.append(role)
        # Validate: all indices distinct and within [0, expected_count)
        all_indices = list(_imu_mapping.values())
        self._imu_all_indices = all_indices
        if not all_indices:
            raise ValueError("imu.imu_mapping must contain at least one mapped sensor role")
        if len(set(all_indices)) != len(all_indices):
            raise ValueError(f"imu.imu_mapping has duplicate indices: {_imu_mapping}")
        for role, idx in _imu_mapping.items():
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"imu.imu_mapping['{role}'] = {idx} must be a non-negative integer")
        self._expected_imu_count = max(all_indices) + 1
        for role, idx in _imu_mapping.items():
            if idx >= self._expected_imu_count:
                raise ValueError(f"imu.imu_mapping['{role}'] = {idx} out of range [0, {self._expected_imu_count})")
        self._imu_sensor_roles = _sensor_roles
        self._imu_joint_roles = _joint_roles
        self._imu_joint_indices = [_imu_mapping[r] for r in _joint_roles]
        self._base_imu_index = _imu_mapping.get('base')
        self._imu_offset_inv_by_index = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                                         for _ in range(self._expected_imu_count)]
        for role, idx in _imu_mapping.items():
            quat = _parse_offset_quaternion(
                _mounting_offsets_cfg.get(role, [1.0, 0.0, 0.0, 0.0]),
                f"imu.mounting_offsets_quat.{role}",
            )
            self._imu_offset_inv_by_index[idx] = quat_conjugate(quat)

        # Runtime base-yaw zeroing. Applied left-multiply to the base IMU quat
        # inside the IMU thread so every downstream consumer (imu_by_role,
        # snapshot.base_imu_quat, read_all_imu_quaternions) sees the corrected
        # value. Set via set_base_yaw_offset_deg(); see HallSlewHoming.
        self._base_yaw_offset_deg: float = 0.0
        self._base_yaw_offset_inv_quat: Optional[np.ndarray] = None  # None == identity

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

    def _correct_imu_quaternion(self, quat: np.ndarray, sensor_index: int) -> np.ndarray:
        """Map a raw IMU quaternion into the configured joint/base frame."""
        normalized = quat_normalize(np.asarray(quat, dtype=np.float32))
        offset_inv = self._imu_offset_inv_by_index[sensor_index]
        return quat_normalize(quat_multiply(normalized, offset_inv))

    def _correct_imu_gyro(self, gyro: np.ndarray, sensor_index: int) -> np.ndarray:
        """Map a raw IMU gyro vector into the configured joint/base frame."""
        return quat_rotate_vector(self._imu_offset_inv_by_index[sensor_index], np.asarray(gyro, dtype=np.float32))
    
    def _start_imu_reader(self) -> None:
        """Initialize IMU reader and start background thread."""
        if not self._enable_imu:
            self.logger.debug("IMU disabled; _start_imu_reader skipped")
            return
        if self.imu_thread is not None and self.imu_thread.is_alive():
            return
        if USBSerialReader is not None:
            try:
                # Data format: binary frames with [w,x,y,z,gx,gy,gz] per IMU
                self.usb_reader = USBSerialReader(
                    baud_rate=115200,
                    timeout=1.0,
                    rt_priority=self._usb_rt_priority,
                    rt_lock_memory=self._rt_lock_memory,
                    rt_cpu_core=self._usb_cpu_core,
                )
                # Firmware self-calibrates on power-on (~30 s stationary), then streams
                # at 200 Hz autonomously. No config handshake needed.
                self.logger.info("IMU connected; Pico firmware will self-calibrate (~30 s) then stream at 200 Hz")
                self.usb_reader.start_background_reader()

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
        if self._imu_rt_priority > 0 or self._rt_lock_memory or self._imu_cpu_core is not None:
            cpu_affinity = None if self._imu_cpu_core is None else {int(self._imu_cpu_core)}
            success = apply_rt_to_thread(
                priority=self._imu_rt_priority,
                policy=SCHED_FIFO,
                lock_memory=self._rt_lock_memory,
                cpu_affinity=cpu_affinity,
                quiet=False
            )
            if success:
                details = []
                if self._imu_rt_priority > 0:
                    details.append(f"SCHED_FIFO-{self._imu_rt_priority}")
                if self._rt_lock_memory:
                    details.append("mlockall")
                if self._imu_cpu_core is not None:
                    details.append(f"core {self._imu_cpu_core}")
                self.logger.info("IMU thread: applied %s", ", ".join(details) if details else "RT settings")
            else:
                self.logger.warning("IMU thread: Failed to apply requested RT settings")

        # Pico streams at 200 Hz; declare data stale if nothing arrives within this window.
        _IMU_STALE_TIMEOUT_S = 0.5
        _last_valid_packet_t = time.monotonic()

        while not self._stop_event.is_set():
            try:
                # Read IMU data - returns list of [w,x,y,z,gx,gy,gz] packets
                imu_packets = self.usb_reader.get_latest_imus(only_new=True)

                # Silent-disconnect guard: if no packets arrive for too long, mark PENDING.
                if imu_packets is None or len(imu_packets) < self._expected_imu_count:
                    if time.monotonic() - _last_valid_packet_t > _IMU_STALE_TIMEOUT_S:
                        with self._imu_lock:
                            if self._imu_state == ReadyState.READY:
                                self._imu_state = ReadyState.PENDING
                                self.logger.warning(
                                    "IMU data stale (no packets for >%.1f s); marking PENDING",
                                    _IMU_STALE_TIMEOUT_S,
                                )

                if imu_packets is not None and len(imu_packets) >= self._expected_imu_count:
                    # Extract only the quaternion portion [w,x,y,z] from each IMU packet
                    # Data format is [w, x, y, z, gx, gy, gz] (7 values per IMU)
                    raw_quat_only = [
                        quat_normalize(np.array(pkt[:4], dtype=np.float32))
                        for pkt in imu_packets[:self._expected_imu_count]
                    ]
                    quat_only = [
                        self._correct_imu_quaternion(raw_quat_only[i], i)
                        for i in range(self._expected_imu_count)
                    ]
                    # Apply runtime base-yaw zeroing if configured. Atomic read
                    # under the GIL — no lock needed for the inverse-quat ref.
                    if self._base_imu_index is not None:
                        yaw_off_inv = self._base_yaw_offset_inv_quat
                        if yaw_off_inv is not None:
                            quat_only[self._base_imu_index] = quat_normalize(
                                quat_multiply(yaw_off_inv, quat_only[self._base_imu_index])
                            )
                    capture_debug = self._debug_telemetry_enabled
                    gyro_only = [
                        self._correct_imu_gyro(np.array(pkt[4:7], dtype=np.float32), i)
                        for i, pkt in enumerate(imu_packets[:self._expected_imu_count])
                    ]
                    # Build all derived values before acquiring the lock so the
                    # lock hold time is just a burst of reference assignments.
                    new_imu_gyro = [gyro_only[i] for i in self._imu_joint_indices] if gyro_only is not None else None
                    new_device_ts = getattr(self.usb_reader, 'last_timestamp_us', None)
                    new_base_imu_gyro = (
                        gyro_only[self._base_imu_index].copy()
                        if gyro_only is not None and self._base_imu_index is not None else None
                    )

                    # Validate quaternion magnitudes for all configured IMUs (should be ~1.0).
                    # During the ~33 s power-on calibration phase (3 s settle + 30 s sampling)
                    # the Pico sends MSG_TYPE_CAL_WAIT every 200 ms and emits no data frames,
                    # so _imu_state stays PENDING and this block is never reached. Magnitude
                    # validation only runs once streaming begins post-calibration.
                    valid_data = True
                    for i in self._imu_all_indices:
                        mag = np.linalg.norm(quat_only[i])
                        if mag < 0.95 or mag > 1.05:
                            valid_data = False
                            break

                    new_imu_by_role = None
                    new_raw_quat = None
                    new_corrected_quat = None
                    new_imu_data = None
                    new_base_imu_quat = None
                    became_ready = False
                    if valid_data:
                        new_imu_by_role = {
                            role: quat_only[idx].copy()
                            for role, idx in self._imu_mapping.items()
                        }
                        if capture_debug:
                            new_raw_quat = [q.copy() for q in raw_quat_only]
                            new_corrected_quat = [q.copy() for q in quat_only]
                        new_imu_data = [quat_only[i] for i in self._imu_joint_indices]
                        if self._base_imu_index is not None:
                            new_base_imu_quat = quat_only[self._base_imu_index].copy()

                    if valid_data:
                        _last_valid_packet_t = time.monotonic()

                    # Publish lock-free snapshot for hot-path readers first.
                    # Single STORE_ATTR is atomic under the GIL — no lock needed.
                    if valid_data:
                        self._imu_snapshot = _ImuSnapshot(
                            imu_data=new_imu_data,
                            imu_by_role=new_imu_by_role,
                            base_imu_quat=new_base_imu_quat,
                            base_imu_gyro=new_base_imu_gyro,
                            imu_gyro=new_imu_gyro,
                            raw_quat=new_raw_quat if capture_debug else None,
                            corrected_quat=new_corrected_quat if capture_debug else None,
                            device_ts=new_device_ts,
                        )

                    # Lock section: keep legacy fields alive for non-hot-path callers
                    # and manage _imu_state transitions.
                    with self._imu_lock:
                        self.latest_imu_gyro = new_imu_gyro
                        self._imu_last_device_ts = new_device_ts
                        if new_base_imu_gyro is not None:
                            self.latest_base_imu_gyro = new_base_imu_gyro
                        else:
                            self.latest_base_imu_gyro = None
                        if valid_data:
                            self.latest_imu_by_role = new_imu_by_role
                            self.latest_imu_raw_quat = new_raw_quat if capture_debug else None
                            self.latest_imu_corrected_quat = new_corrected_quat if capture_debug else None
                            self.latest_imu_data = new_imu_data
                            if new_base_imu_quat is not None:
                                self.latest_base_imu_quat = new_base_imu_quat
                            if self._imu_state != ReadyState.READY:
                                self._imu_state = ReadyState.READY
                                became_ready = True
                    if became_ready:
                        self.logger.info("IMU data streaming")

                    if valid_data:
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

                if imu_packets is None:
                    time.sleep(0.0005)

            except Exception as e:
                self.logger.error(f"IMU reader thread error: {e}")
                with self._imu_lock:
                    self._imu_state = ReadyState.PENDING
                time.sleep(0.1)  # Back off on error

    def _ensure_adc_initialized(self) -> bool:
        """Create ADC helper if not already initialized."""
        if self.adc is None:
            try:
                from .ADC import SimpleADC
                self.adc = SimpleADC()
            except Exception as e:
                self.logger.error(f"ADC initialization failed: {e}")
                self.adc = None
                return False
        if self._adc_channel_plan is None:
            self._adc_channel_plan = self._build_adc_channel_plan()
        return True

    def _build_adc_channel_plan(self) -> List[Dict[str, Any]]:
        """Resolve requested ADC channels into a concrete sampling plan."""
        if self.adc is None:
            raise RuntimeError("ADC not initialized")

        default_pressure_channels = [
            "LiftBoom retract ps",
            "LiftBoom extend ps",
            "TiltBoom retract ps",
            "TiltBoom extend ps",
            "Scoop extend ps",
            "Scoop retract ps",
            "Pump ps",
        ]
        requests = self._adc_channel_requests if self._adc_channel_requests is not None else default_pressure_channels
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

            plan.append({
                'board': board_name,
                'channel': int(channel),
                'name': name,
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

    def _adc_reader_thread(self) -> None:
        """Background thread for continuous ADC reading."""
        if self._adc_rt_priority > 0 or self._rt_lock_memory or self._adc_cpu_core is not None:
            cpu_affinity = None if self._adc_cpu_core is None else {int(self._adc_cpu_core)}
            success = apply_rt_to_thread(
                priority=self._adc_rt_priority,
                policy=SCHED_FIFO,
                lock_memory=self._rt_lock_memory,
                cpu_affinity=cpu_affinity,
                quiet=False
            )
            if success:
                details = []
                if self._adc_rt_priority > 0:
                    details.append(f"SCHED_FIFO-{self._adc_rt_priority}")
                if self._rt_lock_memory:
                    details.append("mlockall")
                if self._adc_cpu_core is not None:
                    details.append(f"core {self._adc_cpu_core}")
                self.logger.info("ADC thread: applied %s", ", ".join(details) if details else "RT settings")
            else:
                self.logger.warning("ADC thread: Failed to apply requested RT settings")

        next_run_time = time.perf_counter()
        read_period = 1.0 / max(1.0, self._adc_expected_hz)

        while not self._stop_event.is_set():
            try:
                # Sample configured channels
                new_readings = {}
                for ch in self._adc_channel_plan:
                    voltage = self.adc.read_channel(ch['board'], ch['channel'])
                    new_readings[ch['name']] = voltage

                sample_timestamp = time.time()
                with self._adc_lock:
                    self._latest_adc_readings = new_readings
                    self._latest_adc_timestamp = sample_timestamp
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
        """Read latest corrected IMU quaternions.

        Mounting offsets are already removed in the hardware layer.

        Raises on error instead of returning None (SAFETY: PWM reset + pump stopped before raising)

        Hot path — lock-free. Reads _imu_snapshot via a single atomic attribute
        load; _imu_state is checked bare (single-word read, safe under CPython GIL).
        """
        if self._imu_state != ReadyState.READY:
            raise RuntimeError("IMU not ready - cannot read IMU data")
        snapshot = self._imu_snapshot
        if snapshot is None or snapshot.imu_data is None:
            raise RuntimeError("IMU data unavailable")
        return [q.copy() for q in snapshot.imu_data]

    @_safe_hardware_operation
    def read_imu_gyro(self) -> Optional[List[np.ndarray]]:
        """Read latest corrected IMU gyro data [gx, gy, gz] per IMU (deg/s).
        """
        with self._imu_lock:
            if self._imu_state != ReadyState.READY:
                raise RuntimeError("IMU not ready - cannot read gyro data")
            if self.latest_imu_gyro is None:
                raise RuntimeError("IMU gyro data is None")
            return [g.copy() for g in self.latest_imu_gyro]

    def try_read_imu_gyro(self) -> Optional[Dict[str, Any]]:
        """Best-effort gyro read for optional control features.

        Unlike read_imu_gyro(), this method never raises and never triggers a safety reset.
        Returns None when gyro telemetry is unavailable.
        """
        with self._imu_lock:
            if self._imu_state != ReadyState.READY:
                return None
            if self.latest_imu_gyro is None:
                return None
            return {
                'gyro': [g.copy() for g in self.latest_imu_gyro],
                'device_timestamp_us': self._imu_last_device_ts,
            }

    def read_base_imu(self) -> Optional[Dict[str, Any]]:
        """Read latest corrected base/slew IMU data.

        Returns {'quat': ndarray[4], 'gyro': ndarray[3]} or None if base IMU
        is not configured or no data has arrived yet.

        Hot path — lock-free via _imu_snapshot.
        """
        if self._base_imu_index is None:
            return None
        if self._imu_state != ReadyState.READY:
            return None
        snapshot = self._imu_snapshot
        if snapshot is None or snapshot.base_imu_quat is None:
            return None
        result = {'quat': snapshot.base_imu_quat.copy()}
        if snapshot.base_imu_gyro is not None:
            result['gyro'] = snapshot.base_imu_gyro.copy()
        return result

    def set_base_yaw_offset_deg(self, yaw_deg: float) -> None:
        """Subtract ``yaw_deg`` from the base IMU yaw on every IMU sample.

        Used by Hall-based homing to re-zero slew yaw at a known mechanical
        reference. The offset is applied left-multiply in the world frame so
        the corrected base quat has yaw=0 when the raw base IMU reads
        ``yaw_deg``. Pass ``0.0`` to clear.
        """
        yaw_rad = float(np.radians(float(yaw_deg)))
        if abs(yaw_rad) < 1e-9:
            self._base_yaw_offset_deg = 0.0
            self._base_yaw_offset_inv_quat = None
            return
        # Inverse Z-rotation: pre-multiplied to cancel measured yaw.
        half = -yaw_rad / 2.0
        self._base_yaw_offset_inv_quat = np.array(
            [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32
        )
        self._base_yaw_offset_deg = float(yaw_deg)
        self.logger.info("Base yaw offset set: subtracting %.3f deg from IMU yaw", float(yaw_deg))

    def get_base_yaw_offset_deg(self) -> float:
        """Current base-yaw offset (deg) being subtracted from IMU readings."""
        return float(self._base_yaw_offset_deg)

    def clear_base_yaw_offset(self) -> None:
        """Remove the base-yaw offset (back to raw mounting-corrected yaw)."""
        self.set_base_yaw_offset_deg(0.0)

    @_safe_hardware_operation
    def read_all_imu_quaternions(self) -> Optional[List[np.ndarray]]:
        """Read corrected IMU quaternions in configured canonical role order."""
        with self._imu_lock:
            if self._imu_state != ReadyState.READY:
                raise RuntimeError("IMU not ready - cannot read IMU data")
            if self.latest_imu_by_role is None:
                raise RuntimeError("IMU role data is missing")
            missing = [role for role in self._imu_sensor_roles if role not in self.latest_imu_by_role]
            if missing:
                raise RuntimeError(f"IMU role data is missing for {missing}")
            return [self.latest_imu_by_role[role].copy() for role in self._imu_sensor_roles]

    def read_imu_debug_quaternions(self) -> Optional[Dict[str, Any]]:
        """Read raw and mounting-corrected IMU quaternions by physical sensor index.

        This is for bench mapping/debug display only. It deliberately does not
        project values through the controller's kinematic model.
        """
        with self._imu_lock:
            if self.latest_imu_raw_quat is None or self.latest_imu_corrected_quat is None:
                return None
            role_by_index = {idx: role for role, idx in self._imu_mapping.items()}
            descriptors = []
            if self.usb_reader is not None:
                try:
                    descriptors = self.usb_reader.imu_descriptors
                except Exception:
                    descriptors = []
            return {
                'raw_quats': [q.copy() for q in self.latest_imu_raw_quat],
                'corrected_quats': [q.copy() for q in self.latest_imu_corrected_quat],
                'role_by_index': dict(role_by_index),
                'descriptors': list(descriptors),
                'device_timestamp_us': self._imu_last_device_ts,
            }

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

        try:
            if getattr(self, 'usb_reader', None) is not None:
                self.usb_reader.stop_background_reader()
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
            self.latest_imu_data = None
            self.latest_imu_by_role = None
            self.latest_imu_raw_quat = None
            self.latest_imu_corrected_quat = None
            self.latest_base_imu_quat = None
            self.latest_base_imu_gyro = None
            self.latest_imu_gyro = None
            self._imu_last_device_ts = None
            self._imu_snapshot = None
        with self._adc_lock:
            self._adc_state = ReadyState.PENDING
            self._latest_adc_readings = {}
            self._latest_adc_timestamp = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def send_named_pwm_commands(self, commands: Dict[str, float], *,
                                unset_to_zero: Optional[bool] = None,
                                command_ts: Optional[float] = None) -> bool:
        """Convenience method: send name-based PWM commands.

        Args:
            commands: Mapping from channel name to command value [-1, 1]. Unknown names ignored.
            unset_to_zero: If None, use controller default; otherwise override per call.
            command_ts: Optional monotonic timestamp of when command was generated (e.g., UDP receive time).
                        If provided and PWM controller has stale_timeout_s configured, commands older
                        than the timeout will be rejected for safety.
        """
        if self._pwm_state != ReadyState.READY or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.update_named(commands,
                                             unset_to_zero=unset_to_zero,
                                             command_ts=command_ts)
            return True
        except Exception as e:
            self.logger.error(f"PWM named command error: {e}")
            return False

    def set_pump_enabled(self, enabled: bool, *, flush: bool = True) -> bool:
        """Enable or disable the hydraulic pump immediately."""
        if self._pwm_state != ReadyState.READY or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.set_pump_enabled(enabled, flush=flush)
            return True
        except Exception as e:
            self.logger.error(f"Pump toggle error: {e}")
            return False

    def set_pump_speed_us(self, pulse_us, *, flush: bool = True) -> bool:
        """Set pump speed directly in microseconds, bypassing auto/static logic.

        Clamped to [pulse_min, pulse_max] from config.
        Pass None to release direct control and return to auto/static mode.
        """
        if self._pwm_state != ReadyState.READY or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.set_pump_speed_us(pulse_us, flush=flush)
            return True
        except Exception as e:
            self.logger.error(f"Pump direct speed error: {e}")
            return False

    def set_pump_auto(self, auto: bool) -> bool:
        """Enable (True) or disable (False) valve-activity-based auto pump speed scaling."""
        if self._pwm_state != ReadyState.READY or self.pwm_controller is None:
            return False
        try:
            self.pwm_controller.set_pump_auto(auto)
            return True
        except Exception as e:
            self.logger.error(f"Pump auto mode error: {e}")
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
        """Compatibility flag for optional extra telemetry paths."""
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
        }
        if self.usb_reader is not None:
            try:
                imu_status = self.usb_reader.status()
                status['imu_startup_phase'] = imu_status.get('startup_phase')
                status['imu_calibration_wait_s'] = imu_status.get('calibration_wait_s', 0.0)
                status['imu_calibration_report'] = imu_status.get('calibration_report')
            except Exception:
                pass

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
