#!/usr/bin/env python3
"""
Excavator Controller API with Background Control Loop

This provides a simple interface:
- give_pose(position, rotation_y_deg): Set target pose
- get_pose(): Get current pose  
- start()/stop(): Control background loop
- pause()/resume(): Pause/resume control loop (hardware stays active)

The controller runs autonomously in a background thread, constantly working
towards the last given target pose. Use pause()/resume() for safe A* planning.
"""

import time
import threading
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import sys
import os
import yaml

# Import project modules
from . import diff_ik_V2 as diff_ik
from .pid import PIDController
from .quaternion_math import quat_from_axis_angle
from .perf_tracker import ControlLoopPerfTracker

# Load settings
_here = os.path.dirname(os.path.abspath(__file__))
_cfg_dir = os.path.abspath(os.path.join(_here, os.pardir, 'configuration_files'))
_cfg_file = os.path.join(_cfg_dir, 'pathing_config.py')
if _cfg_dir not in sys.path:
    sys.path.append(_cfg_dir)
if not os.path.isfile(_cfg_file):
    raise ImportError("configuration_files/pathing_config.py not found (copy required for IRL)")
from pathing_config import DEFAULT_CONFIG as _DEFAULT_CONFIG  # type: ignore


@dataclass
class ControllerConfig:
    """Configuration for the excavator controller.

    Note: All parameters are now loaded from configuration_files/control_config.yaml
    This class is populated dynamically at runtime.
    """
    output_limits: Tuple[float, float]
    control_frequency: float  # Hz


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


@dataclass
class MotionState:
    """Internal state for jerk-limited motion smoothing."""

    # Linear motion state (Cartesian position)
    current_position: np.ndarray  # [x, y, z] in meters
    current_velocity: float       # Linear velocity magnitude in m/s
    current_acceleration: float   # Linear acceleration magnitude in m/s^2

    # Rotational motion state (around Y axis for excavator)
    current_rotation_deg: float    # Current rotation in degrees
    current_rot_velocity_dps: float  # Rotational velocity in deg/s
    current_rot_acceleration_dps2: float  # Rotational acceleration in deg/s^2

    # Timing
    last_update_time: float  # Timestamp of last update (perf_counter)
    is_initialized: bool     # Whether state has been initialized


@dataclass
class _SCurveParams:
    max_velocity: float
    max_acceleration: float
    max_deceleration: float
    max_jerk: float


class _SCurveGenerator:
    """Jerk-limited velocity profile (inlined from the old scurve_profile)."""

    def __init__(self, params: _SCurveParams):
        self.params = params
        self.reset()

    def reset(self):
        self.current_velocity = 0.0
        self.current_acceleration = 0.0

    def compute_next_velocity(
        self,
        current_velocity: float,
        current_acceleration: float,
        target_velocity: float,
        distance_remaining: float,
        dt: float
    ):
        max_jerk = self.params.max_jerk
        max_accel = self.params.max_acceleration
        max_decel = self.params.max_deceleration
        max_vel = min(self.params.max_velocity, target_velocity)

        t_jerk = max_decel / max_jerk if max_jerk > 0 else 0.0

        d_jerk_up = current_velocity * t_jerk + (1 / 6) * max_jerk * t_jerk**3
        v_after_ramp = current_velocity + 0.5 * max_decel * t_jerk

        t_const_decel = 0.0
        if v_after_ramp > 0 and max_decel > 0:
            t_const_decel = max(0, (v_after_ramp - 0.5 * max_decel * t_jerk) / max_decel)

        d_const = v_after_ramp * t_const_decel - 0.5 * max_decel * t_const_decel**2

        v_before_final = v_after_ramp - max_decel * t_const_decel
        d_jerk_down = v_before_final * t_jerk - 0.5 * max_decel * t_jerk**2 + (1 / 6) * (-max_jerk) * t_jerk**3

        stopping_distance = d_jerk_up + max(0, d_const) + max(0, d_jerk_down)
        need_decelerate = distance_remaining <= stopping_distance * 1.2

        if need_decelerate:
            target_accel = -max_decel
            if current_acceleration > target_accel:
                jerk = -max_jerk
            else:
                jerk = 0
                current_acceleration = target_accel
        else:
            if current_velocity < max_vel:
                target_accel = max_accel
                if current_acceleration < target_accel:
                    jerk = max_jerk
                else:
                    jerk = 0
                    current_acceleration = target_accel
            else:
                if abs(current_acceleration) > 1e-6:
                    jerk = -np.sign(current_acceleration) * max_jerk
                else:
                    jerk = 0
                    current_acceleration = 0

        next_acceleration = current_acceleration + jerk * dt
        next_acceleration = np.clip(next_acceleration, -max_decel, max_accel)

        next_velocity = current_velocity + next_acceleration * dt
        next_velocity = np.clip(next_velocity, 0.0, max_vel)

        if distance_remaining < 0.001 and next_velocity < 0.01:
            next_velocity = 0.0
            next_acceleration = 0.0

        return next_velocity, next_acceleration


def _create_scurve_generator(
    max_velocity: float,
    max_acceleration: float,
    max_deceleration: float,
    max_jerk: float
) -> _SCurveGenerator:
    params = _SCurveParams(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_deceleration=max_deceleration,
        max_jerk=max_jerk
    )
    return _SCurveGenerator(params)


class MotionProcessor:
    """
    Jerk-limited smoothing layer for pose commands.

    Kept lightweight: operates on EE position + yaw, uses jerk-limited S-curve
    profiles, and accepts measured feedback to stay aligned with hardware state.
    """

    def __init__(
        self,
        max_velocity: float = 0.02,
        max_acceleration: float = 0.5,
        max_deceleration: float = 0.5,
        max_jerk: float = 2.0,
        max_rot_velocity_dps: float = 45.0,
        max_rot_acceleration_dps2: float = 180.0,
        max_rot_jerk_dps3: float = 720.0,
        enable_smoothing: bool = True
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_jerk = max_jerk

        self.max_rot_velocity_dps = max_rot_velocity_dps
        self.max_rot_acceleration_dps2 = max_rot_acceleration_dps2
        self.max_rot_jerk_dps3 = max_rot_jerk_dps3

        self.enable_smoothing = enable_smoothing

        self.linear_scurve = _create_scurve_generator(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
            max_jerk=max_jerk
        )

        self.rotation_scurve = _create_scurve_generator(
            max_velocity=max_rot_velocity_dps,
            max_acceleration=max_rot_acceleration_dps2,
            max_deceleration=max_rot_acceleration_dps2,
            max_jerk=max_rot_jerk_dps3
        )

        self.state = MotionState(
            current_position=np.zeros(3, dtype=np.float32),
            current_velocity=0.0,
            current_acceleration=0.0,
            current_rotation_deg=0.0,
            current_rot_velocity_dps=0.0,
            current_rot_acceleration_dps2=0.0,
            last_update_time=0.0,
            is_initialized=False
        )

    def reset(self, position: Optional[np.ndarray] = None, rotation_deg: float = 0.0):
        """Reset smoothing state; keeps last position if none provided."""
        if position is not None:
            self.state.current_position = np.array(position, dtype=np.float32)
        self.state.current_velocity = 0.0
        self.state.current_acceleration = 0.0
        self.state.current_rotation_deg = float(rotation_deg)
        self.state.current_rot_velocity_dps = 0.0
        self.state.current_rot_acceleration_dps2 = 0.0
        self.state.last_update_time = time.perf_counter()
        self.state.is_initialized = True
        self.linear_scurve.reset()
        self.rotation_scurve.reset()

    def sync_feedback(
        self,
        position: np.ndarray,
        rotation_deg: float,
        linear_velocity: Optional[float] = None,
        rot_velocity_dps: Optional[float] = None
    ):
        """Align internal state with measured pose/velocity (cheap re-seed)."""
        self.state.current_position = np.array(position, dtype=np.float32)
        self.state.current_rotation_deg = float(rotation_deg)
        if linear_velocity is not None:
            self.state.current_velocity = float(abs(linear_velocity))
        if rot_velocity_dps is not None:
            self.state.current_rot_velocity_dps = float(rot_velocity_dps)
        self.state.last_update_time = time.perf_counter()
        self.state.is_initialized = True

    def process_target(
        self,
        target_pos: np.ndarray,
        target_rot_deg: float,
        dt: float
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Smooth a single target pose given an explicit dt.

        Returns:
            smoothed_position, smoothed_rotation_deg, linear_velocity_vector (m/s), rotational_velocity_dps
        """
        target_pos = np.array(target_pos, dtype=np.float32)

        if not self.state.is_initialized:
            self.reset(position=target_pos, rotation_deg=target_rot_deg)
            return target_pos.copy(), float(target_rot_deg), np.zeros(3, dtype=np.float32), 0.0

        dt = max(0.001, min(float(dt), 0.1))

        if not self.enable_smoothing:
            self.state.current_position = target_pos.copy()
            self.state.current_rotation_deg = float(target_rot_deg)
            return target_pos.copy(), float(target_rot_deg), np.zeros(3, dtype=np.float32), 0.0

        prev_pos = self.state.current_position.copy()
        smoothed_pos = self._process_linear_motion(target_pos, dt)
        lin_vel_vec = (smoothed_pos - prev_pos) / dt

        prev_rot = self.state.current_rotation_deg
        smoothed_rot = self._process_rotational_motion(target_rot_deg, dt)
        rot_vel_dps = (smoothed_rot - prev_rot) / dt if dt > 0 else 0.0

        # Use the internally tracked rotational velocity when smoothing is enabled
        # (captures the signed direction from the S-curve generator).
        rot_vel_dps = self.state.current_rot_velocity_dps if self.enable_smoothing else rot_vel_dps
        return smoothed_pos, smoothed_rot, lin_vel_vec.astype(np.float32), float(rot_vel_dps)

    def _process_linear_motion(self, target_pos: np.ndarray, dt: float) -> np.ndarray:
        delta = target_pos - self.state.current_position
        distance = float(np.linalg.norm(delta))

        if distance < 1e-6:
            self.state.current_velocity = 0.0
            self.state.current_acceleration = 0.0
            return self.state.current_position.copy()

        direction = delta / distance
        next_velocity, next_acceleration = self.linear_scurve.compute_next_velocity(
            current_velocity=self.state.current_velocity,
            current_acceleration=self.state.current_acceleration,
            target_velocity=self.max_velocity,
            distance_remaining=distance,
            dt=dt
        )

        step_distance = min(distance, next_velocity * dt)
        next_position = self.state.current_position + direction * step_distance

        self.state.current_position = next_position
        self.state.current_velocity = next_velocity
        self.state.current_acceleration = next_acceleration
        return next_position.copy()

    def _process_rotational_motion(self, target_rot_deg: float, dt: float) -> float:
        def wrap_deg(angle: float) -> float:
            return (angle + 180.0) % 360.0 - 180.0

        rot_error = wrap_deg(target_rot_deg - self.state.current_rotation_deg)
        abs_rot_error = abs(rot_error)

        if abs_rot_error < 1e-3:
            self.state.current_rot_velocity_dps = 0.0
            self.state.current_rot_acceleration_dps2 = 0.0
            return self.state.current_rotation_deg

        direction = np.sign(rot_error)
        next_velocity, next_acceleration = self.rotation_scurve.compute_next_velocity(
            current_velocity=abs(self.state.current_rot_velocity_dps),
            current_acceleration=abs(self.state.current_rot_acceleration_dps2),
            target_velocity=self.max_rot_velocity_dps,
            distance_remaining=abs_rot_error,
            dt=dt
        )

        next_velocity_signed = next_velocity * direction
        step_angle = min(abs_rot_error, next_velocity * dt) * direction
        next_rotation = self.state.current_rotation_deg + step_angle

        self.state.current_rotation_deg = next_rotation
        self.state.current_rot_velocity_dps = next_velocity_signed
        self.state.current_rot_acceleration_dps2 = next_acceleration * direction
        return next_rotation


class ExcavatorController:
    def __init__(self, hardware_interface, config: Optional[ControllerConfig] = None,
                 enable_perf_tracking: bool = False, log_level: str = "INFO"):
        """Initialize the excavator controller.

        Args:
            hardware_interface: Hardware interface instance
            config: Optional controller configuration
            enable_perf_tracking: Enable performance statistics tracking
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR" (default: "INFO")
        """
        self.hardware = hardware_interface
        self._enable_perf_tracking = enable_perf_tracking

        # Setup logger for this controller instance
        self.logger = logging.getLogger(f"{__name__}.ExcavatorController")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize debug counters (for periodic logging)
        self._debug_counter = 0
        self._ik_debug_counter = 0

        # Cascade perf flag to hardware if supported (no-op otherwise)
        try:
            if hasattr(self.hardware, 'set_perf_enabled'):
                self.hardware.set_perf_enabled(bool(enable_perf_tracking))
        except Exception:
            pass

        # Load controller configuration from control_config.yaml if not provided
        if config is None:
            gc = _load_control_config()
            if not isinstance(gc, dict) or not gc:
                raise RuntimeError("Controller configuration requires control_config.yaml")

            # Load control frequency from rates section
            control_hz = gc.get('rates', {}).get('control_hz')
            if not isinstance(control_hz, (int, float)) or control_hz <= 0:
                raise RuntimeError("Missing or invalid 'rates.control_hz' in control_config.yaml")

            # Load controller parameters
            ctrl_cfg = gc.get('controller', {})
            if not ctrl_cfg:
                raise RuntimeError("Missing 'controller' section in control_config.yaml")

            try:
                self.config = ControllerConfig(
                    output_limits=(
                        float(ctrl_cfg['output_limits_min']),
                        float(ctrl_cfg['output_limits_max'])
                    ),
                    control_frequency=float(control_hz)
                )
            except KeyError as e:
                raise RuntimeError(f"Missing required controller parameter in control_config.yaml: {e}")
        else:
            self.config = config

        # Store control config reference for later use (PID, velocity limits, etc.)
        self._control_config = _load_control_config() if config is None else {}

        # Robot configuration
        self.robot_config = diff_ik.create_excavator_config()
        diff_ik.warmup_numba_functions()

        # IK controller setup (pull settings from shared config - fail hard if missing)
        if _DEFAULT_CONFIG is None:
            raise RuntimeError("ExcavatorController requires configuration_files/pathing_config.py")

        try:
            _ik_cmd_type = _DEFAULT_CONFIG.ik_command_type
            _ik_method = _DEFAULT_CONFIG.ik_method
            _ik_rel = bool(_DEFAULT_CONFIG.ik_use_relative_mode)
            _ik_params = _DEFAULT_CONFIG.ik_params
            _ignore_axes = _DEFAULT_CONFIG.ignore_axes
            # New settings for diff_ik_V2
            _use_reduced = getattr(_DEFAULT_CONFIG, 'use_reduced_jacobian', True)
            _joint_limits_rel = getattr(_DEFAULT_CONFIG, 'joint_limits_relative', None)
            _ik_velocity_mode = bool(getattr(_DEFAULT_CONFIG, 'ik_velocity_mode', False))
            _ik_velocity_error_gain = float(getattr(_DEFAULT_CONFIG, 'ik_velocity_error_gain', 1.0))
            _ik_use_rot_vel = bool(getattr(_DEFAULT_CONFIG, 'ik_use_rotational_velocity', True))
        except AttributeError as e:
            raise RuntimeError(f"Missing required IK configuration in pathing_config.py: {e}")

        # IK config uses ignore_axes for orientation filtering
        # Prepare optional IK V2 extras from control_config
        ctrl_cfg = self._control_config.get('controller', {}) if isinstance(self._control_config, dict) else {}

        # Velocity limiting settings
        enable_vel_limit = bool(ctrl_cfg.get('enable_velocity_limiting', True))
        per_joint_max = ctrl_cfg.get('per_joint_max_velocity', None)
        max_joint_velocities = None
        if isinstance(per_joint_max, (list, tuple)) and len(per_joint_max) == 4:
            max_joint_velocities = [float(v) for v in per_joint_max]

        # Joint limits
        joint_limits = None
        if isinstance(_joint_limits_rel, (list, tuple)) and len(_joint_limits_rel) == 4:
            # Accept degrees or radians; assume degrees if magnitudes > pi
            def _to_rad_pair(p):
                a, b = float(p[0]), float(p[1])
                if max(abs(a), abs(b)) > np.pi + 1e-6:
                    return (np.radians(a), np.radians(b))
                return (a, b)
            joint_limits = [_to_rad_pair(p) for p in _joint_limits_rel]

        self.ik_config = diff_ik.IKControllerConfig(
            command_type=_ik_cmd_type,
            ik_method=_ik_method,
            use_relative_mode=_ik_rel,
            ik_params=_ik_params,
            ignore_axes=_ignore_axes,
            # IK V2 extensions
            use_reduced_jacobian=bool(_use_reduced),
            enable_velocity_limiting=enable_vel_limit,
            max_joint_velocities=max_joint_velocities,
            joint_limits=joint_limits,
            velocity_mode=_ik_velocity_mode,
            velocity_error_gain=_ik_velocity_error_gain,
            use_rotational_velocity=_ik_use_rot_vel,
        )
        # Pass verbose flag if IK controller supports it, else use DEBUG level check
        ik_verbose = self.logger.level <= logging.DEBUG
        ik_default_dt = 1.0 / float(self.config.control_frequency)
        self.ik_controller = diff_ik.IKController(
            self.ik_config,
            self.robot_config,
            verbose=ik_verbose,
            default_dt=ik_default_dt
        )

        # Relative IK control parameters (pulling toward target)
        try:
            self._relative_pos_gain = float(_DEFAULT_CONFIG.relative_pos_gain)
            self._relative_rot_gain = float(_DEFAULT_CONFIG.relative_rot_gain)
        except AttributeError as e:
            raise RuntimeError(f"Missing required relative control gains in pathing_config.py: {e}")

        # Load PID gains from control_config.yaml
        if not isinstance(self._control_config, dict) or 'pid' not in self._control_config:
            raise RuntimeError("PID configuration missing from control_config.yaml")

        pid_cfg = self._control_config['pid']
        joint_configs = []
        for i, name in enumerate(["Slew", "Boom", "Arm", "Bucket"]):
            joint_key = f'joint{i}'
            if joint_key not in pid_cfg:
                raise RuntimeError(f"Missing PID config for {joint_key} in control_config.yaml")
            j = pid_cfg[joint_key]
            joint_configs.append({
                "name": name,
                "kp": float(j['kp']),
                "ki": float(j['ki']),
                "kd": float(j['kd'])
            })

        self.joint_pids = []
        for cfg in joint_configs:
            pid = PIDController(
                kp=cfg["kp"],
                ki=cfg["ki"],
                kd=cfg["kd"],
                min_output=self.config.output_limits[0],
                max_output=self.config.output_limits[1],
            )
            self.joint_pids.append(pid)
        # Keep original PID gains for scheduling (primarily slew)
        self._base_pid_gains = joint_configs

        # Motion processor for jerk-limited pose commands (aligned with IK loop)
        self.motion_processor = MotionProcessor(
            max_velocity=float(_DEFAULT_CONFIG.speed_mps),
            max_acceleration=float(_DEFAULT_CONFIG.max_accel_mps2),
            max_deceleration=float(_DEFAULT_CONFIG.max_decel_mps2),
            max_jerk=float(_DEFAULT_CONFIG.max_jerk_mps3),
            enable_smoothing=bool(getattr(_DEFAULT_CONFIG, 'enable_jerk', True)),
        )

        # Thread control
        self._control_thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_ack_event = threading.Event()
        self._lock = threading.Lock()

        # State variables
        self._raw_target_position = None  # Un-smoothed target input
        self._raw_target_rotation_deg = None
        self._target_position = None     # Smoothed target (fed to IK)
        self._target_orientation = None
        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_orientation_y_deg = 0.0
        self._current_projected_quats = None  # Cache processed quaternions
        self._current_linear_velocity = 0.0
        self._current_rot_velocity_degps = 0.0
        self._target_linear_velocity = np.zeros(3, dtype=np.float32)
        self._target_rot_velocity_dps = 0.0
        self._outputs_zeroed = False  # Track whether we've already sent a zero/neutral command
        # Debug telemetry (data capture for detailed logging/analysis)
        self._last_pi_outputs = None
        self._last_named_commands = None
        self._prev_joint_angles = None
        self._prev_joint_time = None
        self._last_joint_vel_radps = None
        self._prev_pose_time = None
        self._prev_pose = None
        self._prev_orientation_deg = None

        # Performance tracking (standardized tracker with stage timing and percentiles)
        self._perf_tracker = ControlLoopPerfTracker(
            enabled=enable_perf_tracking,
            target_hz=self.config.control_frequency,
            buffer_size=1000
        )

        self.logger.info("Controller initialized")

    def start(self) -> None:
        # TODO: Call self.hardware._check_faults() here to fail fast with clear error
        # if any hardware subsystem (PWM/IMU/ADC) is in FAULT state
        if self._control_thread is not None:
            self.logger.warning("Controller already running!")
            return

        self._stop_event.clear()
        self._pause_event.clear()  # Start unpaused
        self._pause_ack_event.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        self.logger.info("Control loop started")

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._control_thread is None:
            return

        self._stop_event.set()
        self._control_thread.join(timeout=timeout_s)
        if self._control_thread.is_alive():
            self.logger.warning("Control loop did not stop within timeout; forcing hardware reset")
        self._control_thread = None
        # Safe actuator state
        self.hardware.reset(reset_pump=True)
        # Ensure background hardware threads and serial are closed
        try:
            self.hardware.shutdown()
        except Exception:
            pass

    def pause(self, timeout_s: float = 0.5) -> None:
        """Pause the control loop and clear any pending IK target."""
        if self._control_thread is None or not self._control_thread.is_alive():
            self.logger.warning("Controller not running - cannot pause")
            return

        # Engage paused state immediately so the loop stops computing commands
        self._pause_event.set()

        # Wait briefly for loop to acknowledge pause (reduces racey state changes)
        if timeout_s and timeout_s > 0:
            if not self._pause_ack_event.wait(timeout=timeout_s):
                self.logger.warning("Pause requested but not acknowledged by loop yet")

        # Clear target and controller states to avoid stale jumps on resume
        self.clear_target()
        # Ensure hardware is commanded to safe (zero) outputs while paused
        # This guarantees actuators hold still during path planning. Reset pump too
        # so hydraulics are depressurized while idle.
        try:
            self.hardware.reset(reset_pump=True)
        except Exception:
            pass
        self._outputs_zeroed = True
        self.logger.info("Controller paused (target cleared)")

    def resume(self, timeout_s: float = 0.5) -> None:
        """Resume the control loop from paused state."""
        if self._control_thread is None or not self._control_thread.is_alive():
            self.logger.warning("Controller not running - cannot resume")
            return
        
        # Clear stale PID states to prevent old integral/derivative terms
        for pid in self.joint_pids:
            pid.reset()

        # Reset IK internal command buffers to avoid stale desired pose
        try:
            self.ik_controller.reset()
        except Exception:
            pass

        # IMPORTANT: Update current state before resuming
        # This prevents IK confusion from stale position data
        self._update_current_state()
        # Sync smoother to measured pose so next command starts from reality
        self.motion_processor.reset(
            position=self._current_position,
            rotation_deg=self._current_orientation_y_deg
        )

        self._pause_event.clear()
        if timeout_s and timeout_s > 0:
            # Wait for loop to clear pause acknowledgement
            start = time.perf_counter()
            while self._pause_ack_event.is_set() and (time.perf_counter() - start) < timeout_s:
                time.sleep(0.01)

    def give_pose(self, position, rotation_y_deg: float = 0.0) -> None:
        with self._lock:
            self._raw_target_position = np.array(position, dtype=np.float32)
            self._raw_target_rotation_deg = float(rotation_y_deg)
        # We're about to actively command again
        self._outputs_zeroed = False

    def set_relative_control(self, enabled: bool, pos_gain: Optional[float] = None, rot_gain: Optional[float] = None) -> None:
        """Enable/disable relative IK mode and optionally set gains.

        Args:
            enabled: Whether to use relative mode (delta pose commands)
            pos_gain: Fraction of position error to apply each step (optional)
            rot_gain: Fraction of orientation error (axis-angle) to apply each step (optional)
        """
        self.ik_config.use_relative_mode = bool(enabled)
        # Reset IK internal buffers when toggling modes
        try:
            self.ik_controller.reset()
        except Exception:
            pass
        if pos_gain is not None:
            self._relative_pos_gain = float(pos_gain)
        if rot_gain is not None:
            self._relative_rot_gain = float(rot_gain)

    def clear_target(self) -> None:
        """Clear the current IK target and reset controller state.

        After clearing, the control loop will hold zero PWM (via hardware.reset)
        until a new target is provided with give_pose().
        """
        with self._lock:
            self._raw_target_position = None
            self._raw_target_rotation_deg = None
            self._target_position = None
            self._target_orientation = None

        # Reset PIDs to avoid residual terms
        for pid in self.joint_pids:
            pid.reset()
        # Reset smoother to measured state to avoid jumps on next command
        self.motion_processor.reset(
            position=self._current_position,
            rotation_deg=self._current_orientation_y_deg
        )

    def get_pose(self) -> Tuple[np.ndarray, float]:
        with self._lock:
            return self._current_position.copy(), self._current_orientation_y_deg

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles in degrees (computed from quaternions)."""
        with self._lock:
            if self._current_projected_quats is None:
                return np.zeros(4, dtype=np.float32)

            joint_angles = np.array([
                diff_ik.extract_axis_rotation(q, axis)
                for q, axis in zip(self._current_projected_quats, self.robot_config.rotation_axes)
            ])
            return np.degrees(joint_angles)

    def get_hardware_status(self) -> dict:
        """Get hardware status including all ADC channels for logging."""
        return self.hardware.get_status()

    def get_performance_stats(self) -> dict:
        """Get controller loop performance statistics.

        Returns dict with:
        - Loop timing: avg/std/min/max/p95/p99 in ms
        - Compute timing: avg/std/min/max in ms
        - Headroom and CPU usage
        - Stage timings (sensor, ik_fk, pwm)
        - IK telemetry (velocity limits, joint velocities)
        - Hardware stats (IMU/ADC/PWM rates)
        """
        # Get stats from standardized tracker
        tracker_stats = self._perf_tracker.get_stats()

        if not tracker_stats or tracker_stats.get('samples', 0) == 0:
            # Return empty stats structure
            stats = {
                'avg_loop_time_ms': 0.0,
                'min_loop_time_ms': 0.0,
                'max_loop_time_ms': 0.0,
                'std_loop_time_ms': 0.0,
                'jitter_p95_ms': 0.0,
                'jitter_p99_ms': 0.0,
                'avg_compute_time_ms': 0.0,
                'min_compute_time_ms': 0.0,
                'max_compute_time_ms': 0.0,
                'std_compute_time_ms': 0.0,
                'avg_headroom_ms': 0.0,
                'cpu_usage_pct': 0.0,
                'actual_hz': 0.0,
                'violation_pct': 0.0,
                'sample_count': 0,
                # Stage timings
                'avg_sensor_ms': 0.0, 'min_sensor_ms': 0.0, 'max_sensor_ms': 0.0,
                'avg_ik_fk_ms': 0.0, 'min_ik_fk_ms': 0.0, 'max_ik_fk_ms': 0.0,
                'avg_pwm_ms': 0.0, 'min_pwm_ms': 0.0, 'max_pwm_ms': 0.0,
                # IK telemetry
                'ik_vel_lim_enabled': bool(getattr(self.ik_config, 'enable_velocity_limiting', False)),
                'last_joint_vel_degps': [] if self._last_joint_vel_radps is None else list(np.degrees(self._last_joint_vel_radps)),
                'effective_vel_cap_degps': [],
            }
        else:
            # Map tracker stats to expected output format
            stages = tracker_stats.get('stages', {})
            sensor_stats = stages.get('sensor', {})
            ik_fk_stats = stages.get('ik_fk', {})
            pwm_stats = stages.get('pwm', {})

            actual_hz = tracker_stats.get('hz', 0.0)

            stats = {
                'avg_loop_time_ms': tracker_stats.get('loop_avg_ms', 0.0),
                'min_loop_time_ms': tracker_stats.get('loop_min_ms', 0.0),
                'max_loop_time_ms': tracker_stats.get('loop_max_ms', 0.0),
                'std_loop_time_ms': tracker_stats.get('loop_std_ms', 0.0),
                'jitter_p95_ms': tracker_stats.get('loop_p95_ms', 0.0),
                'jitter_p99_ms': tracker_stats.get('loop_p99_ms', 0.0),
                'avg_compute_time_ms': tracker_stats.get('compute_avg_ms', 0.0),
                'min_compute_time_ms': tracker_stats.get('compute_min_ms', 0.0),
                'max_compute_time_ms': tracker_stats.get('compute_max_ms', 0.0),
                'std_compute_time_ms': tracker_stats.get('compute_std_ms', 0.0),
                'avg_headroom_ms': tracker_stats.get('headroom_avg_ms', 0.0),
                'cpu_usage_pct': tracker_stats.get('cpu_usage_pct', 0.0),
                'actual_hz': actual_hz,
                'violation_pct': tracker_stats.get('violation_pct', 0.0),
                'violation_count': tracker_stats.get('violation_count', 0),
                'sample_count': tracker_stats.get('samples', 0),
                # Stage timings
                'avg_sensor_ms': sensor_stats.get('avg_ms', 0.0),
                'min_sensor_ms': sensor_stats.get('min_ms', 0.0),
                'max_sensor_ms': sensor_stats.get('max_ms', 0.0),
                'avg_ik_fk_ms': ik_fk_stats.get('avg_ms', 0.0),
                'min_ik_fk_ms': ik_fk_stats.get('min_ms', 0.0),
                'max_ik_fk_ms': ik_fk_stats.get('max_ms', 0.0),
                'avg_pwm_ms': pwm_stats.get('avg_ms', 0.0),
                'min_pwm_ms': pwm_stats.get('min_ms', 0.0),
                'max_pwm_ms': pwm_stats.get('max_ms', 0.0),
            }

            # Add IK limiter telemetry in deg/s for easier interpretation
            ik_vel_lim_enabled = bool(getattr(self.ik_config, 'enable_velocity_limiting', False))
            stats['ik_vel_lim_enabled'] = ik_vel_lim_enabled
            # Last measured joint velocities (deg/s)
            if self._last_joint_vel_radps is not None:
                stats['last_joint_vel_degps'] = list(np.degrees(self._last_joint_vel_radps))
            else:
                stats['last_joint_vel_degps'] = []
            # Effective cap in deg/s, inferred from rad/iter caps and actual loop Hz
            if ik_vel_lim_enabled and hasattr(self.ik_controller, 'max_joint_velocities') and actual_hz > 0.0:
                cap_degps = np.degrees(self.ik_controller.max_joint_velocities) * actual_hz
                stats['effective_vel_cap_degps'] = list(cap_degps)
            else:
                stats['effective_vel_cap_degps'] = []

        # Merge hardware perf stats
        try:
            if self._enable_perf_tracking and hasattr(self.hardware, 'get_perf_stats'):
                hw = self.hardware.get_perf_stats() or {}
                # Provide convenient top-level Hz if available
                imu_hz = hw.get('imu', {}).get('hz')
                adc_hz = hw.get('adc', {}).get('hz')
                if imu_hz is not None:
                    stats['imu_hz'] = float(imu_hz)
                if adc_hz is not None:
                    stats['adc_hz'] = float(adc_hz)
                stats['hardware_stats'] = hw
        except Exception:
            pass

        return stats

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.info(f"Log level changed to {level.upper()}")

        # Update IK controller verbose flag if DEBUG level
        ik_verbose = self.logger.level <= logging.DEBUG
        if hasattr(self.ik_controller, 'verbose'):
            self.ik_controller.verbose = ik_verbose

    def get_last_pid_outputs(self):
        """Return last per-joint PID outputs [slew, boom, arm, bucket] or None if not captured."""
        return None if self._last_pi_outputs is None else list(self._last_pi_outputs)

    def get_last_pwm_commands(self):
        """Return last named PWM command dict or empty dict if none."""
        return {} if self._last_named_commands is None else dict(self._last_named_commands)

    def get_ik_debug_info(self) -> dict:
        """Return IK telemetry such as adaptive damping and condition number."""
        return {
            'adaptive_lambda': float(self.ik_controller.last_adaptive_lambda),
            'condition_number': float(self.ik_controller.last_condition_number),
        }

    def get_joint_velocities_degps(self):
        """Return last estimated joint velocities (deg/s) or None if not captured."""
        if self._last_joint_vel_radps is None:
            return None
        return list(np.degrees(self._last_joint_vel_radps))

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._perf_tracker.reset()
        try:
            if self._enable_perf_tracking and hasattr(self.hardware, 'reset_perf_stats'):
                self.hardware.reset_perf_stats()
        except Exception:
            pass

    def _get_raw_quaternions(self) -> Optional[np.ndarray]:
        """
        Read raw sensor data and combine into quaternion array.

        Returns:
            Raw quaternions [slew, boom, arm, bucket] or None if hardware not ready
        """
        try:
            # Read IMU data (3 IMUs: boom, arm, bucket)
            quaternions = self.hardware.read_imu_data()
            if quaternions is None or len(quaternions) != 3:
                return None

            # Read slew quaternion directly from encoder (already computed in hardware)
            slew_quat = self.hardware.read_slew_quaternion()

            # Combine: [slew] + [boom, arm, bucket]
            all_quaternions = np.array([slew_quat] + quaternions, dtype=np.float32)

            return all_quaternions

        except Exception as e:
            self.logger.error(f"Error reading quaternions: {e}")
            return None

    def _control_loop(self) -> None:
        loop_period = 1.0 / self.config.control_frequency
        next_run_time = time.perf_counter()
        last_loop_start = next_run_time

        while not self._stop_event.is_set():
            loop_start_time = time.perf_counter()
            loop_dt = loop_start_time - last_loop_start if last_loop_start is not None else loop_period
            loop_dt = max(0.001, min(loop_dt, 0.1))

            # Check if paused - if so, just sleep and continue
            if self._pause_event.is_set():
                if not self._pause_ack_event.is_set():
                    self._pause_ack_event.set()
                next_run_time = time.perf_counter() + 0.1  # Reset timing when paused
                last_loop_start = next_run_time
                time.sleep(0.1)  # Sleep while paused
                continue
            elif self._pause_ack_event.is_set():
                self._pause_ack_event.clear()

            # Performance tracking: mark loop start
            self._perf_tracker.loop_start()

            try:
                self._perf_tracker.stage_start('sensor')
                self._update_current_state()
                self._perf_tracker.stage_end('sensor')

                # Refresh smoothed target using latest feedback + loop timing
                self._refresh_smoothed_target(loop_dt)

                with self._lock:
                    has_target = self._target_position is not None
                if has_target:
                    self._compute_control_commands(loop_dt)
                else:
                    # Only send a neutral command once when there is no target
                    if not self._outputs_zeroed:
                        self.hardware.reset(reset_pump=False)
                        self._outputs_zeroed = True

            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                self.hardware.reset(reset_pump=True)
                break

            # Performance tracking: mark loop end (before sleep)
            self._perf_tracker.loop_end()

            last_loop_start = loop_start_time

            # Accurate timing: calculate next run time and sleep until then
            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind, reset timing to prevent drift
                next_run_time = time.perf_counter()

    def _update_current_state(self) -> None:
        """Update current robot state from sensors."""
        try:
            # Get raw quaternions from sensors
            raw_quats = self._get_raw_quaternions()
            if raw_quats is None:
                return

            # Compute forward kinematics - get_pose() handles full pipeline
            ee_pos, ee_quat = diff_ik.get_pose(raw_quats, self.robot_config)

            # Also compute processed quaternions for joint angle extraction in control commands
            corrected_quats = diff_ik.apply_imu_offsets(raw_quats, self.robot_config)
            projected_quats = diff_ik.project_to_rotation_axes(corrected_quats, self.robot_config.rotation_axes)
            propagated_quats = diff_ik.propagate_base_rotation(projected_quats, self.robot_config)

            # Extract Y-axis rotation for end-effector orientation
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ee_y_angle_rad = diff_ik.extract_axis_rotation(ee_quat, y_axis)
            ee_y_angle_deg = np.degrees(ee_y_angle_rad)

            # Update shared state
            with self._lock:
                self._current_position = ee_pos
                self._current_orientation_y_deg = ee_y_angle_deg
                self._current_projected_quats = propagated_quats  # Cache processed quats for control
                self._current_raw_quats = raw_quats
                self._current_ee_quat_full = ee_quat

            # Update simple EE velocity estimates for smoother seeding
            now_t = time.perf_counter()
            if self._prev_pose_time is not None:
                dt = max(1e-6, now_t - self._prev_pose_time)
                pos_delta = ee_pos - (self._prev_pose if self._prev_pose is not None else ee_pos)
                lin_vel = float(np.linalg.norm(pos_delta) / dt)
                rot_delta = ee_y_angle_deg - (self._prev_orientation_deg if self._prev_orientation_deg is not None else ee_y_angle_deg)
                rot_delta = (rot_delta + 180.0) % 360.0 - 180.0
                rot_vel = float(rot_delta / dt)
                self._current_linear_velocity = lin_vel
                self._current_rot_velocity_degps = rot_vel
            self._prev_pose_time = now_t
            self._prev_pose = ee_pos
            self._prev_orientation_deg = ee_y_angle_deg

        except Exception as e:
            self.logger.error(f"Error in state update: {e}")

    def _refresh_smoothed_target(self, loop_dt: float) -> None:
        """Update smoothed IK target using jerk-limited motion processor."""
        with self._lock:
            raw_pos = None if self._raw_target_position is None else self._raw_target_position.copy()
            raw_rot = self._raw_target_rotation_deg
            current_pos = self._current_position.copy()
            current_rot = float(self._current_orientation_y_deg)
            lin_vel = self._current_linear_velocity
            rot_vel = self._current_rot_velocity_degps

        if raw_pos is None or raw_rot is None:
            with self._lock:
                self._target_position = None
                self._target_orientation = None
            return

        # Keep smoother aligned with measured state each loop
        self.motion_processor.sync_feedback(
            position=current_pos,
            rotation_deg=current_rot,
            linear_velocity=lin_vel,
            rot_velocity_dps=rot_vel
        )

        smoothed_pos, smoothed_rot, lin_vel_vec, rot_vel_dps = self.motion_processor.process_target(
            target_pos=raw_pos,
            target_rot_deg=raw_rot,
            dt=loop_dt
        )

        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        smoothed_quat = quat_from_axis_angle(y_axis, np.radians(smoothed_rot))

        with self._lock:
            self._target_position = smoothed_pos.astype(np.float32)
            self._target_orientation = smoothed_quat
            self._target_linear_velocity = lin_vel_vec.astype(np.float32)
            self._target_rot_velocity_dps = float(rot_vel_dps)

    def _compute_control_commands(self, loop_dt: float) -> None:
        """Compute and send control commands based on current state and target."""
        # Track IK/FK timing via perf_tracker
        self._perf_tracker.stage_start('ik_fk')

        try:
            # Get cached state from _update_current_state()
            with self._lock:
                if self._current_projected_quats is None:
                    return

                target_pos = self._target_position.copy()
                target_quat = self._target_orientation.copy()
                current_pos = self._current_position.copy()
                raw_quats = self._current_raw_quats
                current_ee_quat_full = self._current_ee_quat_full
                target_lin_vel = self._target_linear_velocity.copy()
                target_rot_vel_dps = float(self._target_rot_velocity_dps)

            # Extract current joint angles using V2 helper (relative angles)
            if raw_quats is None:
                return
            current_joint_angles = diff_ik.compute_relative_joint_angles(raw_quats, self.robot_config)

            # Outer Loop: Task-space IK control
            # Note: Using current_ee_quat_full (including Z-rotation from slew) for IK.
            # The Jacobian's structure naturally constrains which rotations each joint can achieve
            # based on rotation_axes config (slew=Z, boom/arm/bucket=Y).
            if self.ik_config.use_relative_mode:
                # Compute error w.r.t target
                if self.ik_config.command_type == "position":
                    pos_err = (target_pos - current_pos)
                    delta_pos = self._relative_pos_gain * pos_err
                    self.ik_controller.set_command(delta_pos, ee_pos=current_pos, ee_quat=current_ee_quat_full)
                else:
                    # Pose: 6D delta [dx, dy, dz, rx, ry, rz]
                    # Orientation locks are now applied internally by IK controller
                    pos_err, axis_angle_err = diff_ik.compute_pose_error(
                        current_pos, current_ee_quat_full, target_pos, target_quat
                    )
                    delta_pos = self._relative_pos_gain * pos_err
                    delta_rot = self._relative_rot_gain * axis_angle_err
                    delta_pose = np.concatenate([delta_pos, delta_rot])
                    self.ik_controller.set_command(delta_pose, ee_pos=current_pos, ee_quat=current_ee_quat_full)
            else:
                if self.ik_config.command_type == "position":
                    # Position-only mode: command is just [x, y, z]
                    position_command = target_pos
                    self.ik_controller.set_command(position_command, ee_quat=current_ee_quat_full)
                else:
                    # Pose mode: command is [x, y, z, qw, qx, qy, qz]
                    # Orientation locks are now applied internally by IK controller
                    pose_command = np.concatenate([target_pos, target_quat])
                    self.ik_controller.set_command(pose_command)

            desired_vel = None
            if getattr(self.ik_config, "velocity_mode", False):
                if self.ik_config.command_type == "position":
                    desired_vel = target_lin_vel
                else:
                    if getattr(self.ik_config, "use_rotational_velocity", True):
                        rot_vel_rad_s = np.radians(target_rot_vel_dps)
                        desired_vel = np.concatenate([
                            target_lin_vel,
                            np.array([0.0, rot_vel_rad_s, 0.0], dtype=np.float32)
                        ])
                    else:
                        desired_vel = np.concatenate([target_lin_vel, np.zeros(3, dtype=np.float32)])

            target_joint_angles = self.ik_controller.compute(
                ee_pos=current_pos,
                ee_quat=current_ee_quat_full,  # Full orientation incl. slew yaw
                joint_angles=current_joint_angles,
                joint_quats=raw_quats,  # Pass raw IMU quats; V2 handles offsets/propagation
                desired_ee_velocity=desired_vel,
                dt=loop_dt,
            )

            # DEBUG: Periodic IK output check (every ~1 second at 50Hz)
            self._ik_debug_counter += 1
            if self.logger.level <= logging.DEBUG and self._ik_debug_counter % 50 == 0:
                if target_joint_angles is not None:
                    delta_angles = target_joint_angles - current_joint_angles
                    delta_deg = np.degrees(delta_angles)
                    max_delta = np.max(np.abs(delta_deg))
                    if max_delta > 0.1:
                        self.logger.debug(f"IK delta: {delta_deg} deg (max={max_delta:.2f})")
                else:
                    self.logger.warning("IK returned None!")

        except Exception as e:
            self.logger.error(f"Error in control computation: {e}")
            if self.logger.level <= logging.DEBUG:
                import traceback
                traceback.print_exc()
            return

        if target_joint_angles is None:
            # If IK failed, command neutral once (avoid constant zeroing)
            if not self._outputs_zeroed:
                self.logger.warning("IK returned None, sending neutral command")
                self.hardware.reset(reset_pump=False)
                self._outputs_zeroed = True
            return

        def angle_error(target, current):
            return np.arctan2(np.sin(target - current), np.cos(target - current))

        # Inner Loop: Joint-space PID control
        pi_outputs = []
        # TODO: integrate this properly
        # Adaptive slew gain scheduling based on FK X-distance (reach)
        slew_pid = self.joint_pids[0]
        base_slew = self._base_pid_gains[0]
        x_dist = abs(current_pos[0])
        sched_start = 0.35  # no change below this reach
        sched_end = 1.0     # hit min_scale by this reach (keeps user-facing simplicity)
        min_scale = 0.7
        if x_dist <= sched_start:
            scale = 1.0
        elif x_dist >= sched_end:
            scale = min_scale
        else:
            span = max(1e-6, sched_end - sched_start)
            frac = (x_dist - sched_start) / span
            scale = 1.0 - frac * (1.0 - min_scale)
        # Match damping: scale D by sqrt(scale) to roughly preserve damping ratio
        slew_pid.kp = base_slew["kp"] * scale
        slew_pid.kd = base_slew["kd"] * np.sqrt(scale)
        if self.logger.level <= logging.DEBUG and self._debug_counter % 50 == 0:
            print(f"slew gains kp={slew_pid.kp:.3f} kd={slew_pid.kd:.3f} scale={scale:.3f} x={x_dist:.3f}")

        for pid, target_angle, current_angle in zip(
                self.joint_pids, target_joint_angles, current_joint_angles
        ):
            error = angle_error(target_angle, current_angle)
            output = pid.compute(0.0, -error, dt=loop_dt)  # setpoint=0, measurement=-error
            pi_outputs.append(output)

        # Prefer name-based commands to avoid fragile indexing
        named_commands = {
            'scoop': pi_outputs[3],      # bucket
            'lift_boom': pi_outputs[1],  # boom
            'rotate': pi_outputs[0],     # slew
            'tilt_boom': pi_outputs[2],  # arm
        }

        # Always update joint velocity estimate (rad/s)
        now_t = time.perf_counter()
        if self._prev_joint_angles is not None and self._prev_joint_time is not None:
            dtj = max(1e-6, now_t - self._prev_joint_time)
            self._last_joint_vel_radps = (current_joint_angles - self._prev_joint_angles) / dtj
        self._prev_joint_angles = current_joint_angles.copy()
        self._prev_joint_time = now_t

        # Debug telemetry capture (always enabled for get_last_* methods)
        self._last_pi_outputs = list(pi_outputs)
        self._last_named_commands = dict(named_commands)

        # DEBUG: Print commands periodically (every ~1 second at 50Hz)
        self._debug_counter += 1
        if self.logger.level <= logging.DEBUG and self._debug_counter % 50 == 0:
            max_cmd = max(abs(x) for x in pi_outputs)
            if max_cmd > 0.01:  # Only print if non-trivial commands
                self.logger.debug(f"Control commands: slew={pi_outputs[0]:+.3f} boom={pi_outputs[1]:+.3f} arm={pi_outputs[2]:+.3f} bucket={pi_outputs[3]:+.3f}")

        # End IK/FK stage timing, start PWM stage
        self._perf_tracker.stage_end('ik_fk')
        self._perf_tracker.stage_start('pwm')

        self.hardware.send_named_pwm_commands(named_commands)
        # Mark that we are actively commanding
        self._outputs_zeroed = False

        # End PWM stage timing
        self._perf_tracker.stage_end('pwm')

    def __del__(self):
        # Be defensive: __init__ can fail before thread fields exist
        try:
            if getattr(self, "_control_thread", None) is not None:
                self.stop()
        except Exception:
            pass
