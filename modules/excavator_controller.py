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
from . import reachability as reachability_module
from .pid import PIDController
from .differential_ik_cfg import IKControllerConfig, load_excavator_robot_config
from .differential_ik import (
    IKController, extract_axis_rotation,
    joint_angles_to_absolute_quaternions, get_pose_from_joint_angles,
)
from .excavator_ik_utils import (
    canonical_joint_angles_from_imus, gravity_pitch_from_quat,
    compute_relative_joint_angles, warmup_numba_functions,
)
from .quaternion_math import (
    quat_from_axis_angle, quat_multiply, quat_conjugate, quat_normalize,
    compute_pose_error,
)
from .reachability import ReachabilityResult
from .perf_tracker import ControlLoopPerfTracker
from .rt_utils import apply_rt_to_thread, SCHED_FIFO

# Load settings
_here = os.path.dirname(os.path.abspath(__file__))
_cfg_dir = os.path.abspath(os.path.join(_here, os.pardir, 'configuration_files'))
if _cfg_dir not in sys.path:
    sys.path.append(_cfg_dir)

# Import pathing config for motion processing parameters only (speed, jerk, etc.)
_pathing_cfg_file = os.path.join(_cfg_dir, 'pathing_config.py')
if os.path.isfile(_pathing_cfg_file):
    from pathing_config import DEFAULT_CONFIG as _PATHING_CONFIG  # type: ignore
else:
    _PATHING_CONFIG = None

# Module-level constants and pure helpers used in the hot control loop.
# Defining them here avoids per-tick allocation / nested-function overhead.
_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _angle_error(target: float, current: float) -> float:
    """Wrap-aware angle difference in radians."""
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


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
                 enable_perf_tracking: bool = False, log_level: str = "INFO",
                 rt_priority: int = 0, rt_lock_memory: bool = False,
                 rt_cpu_core: Optional[int] = None):
        """Initialize the excavator controller.

        Args:
            hardware_interface: Hardware interface instance
            config: Optional controller configuration
            enable_perf_tracking: Enable performance statistics tracking
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR" (default: "INFO")
            rt_priority: RT priority for the controller loop thread (0 = normal).
            rt_lock_memory: Whether to call mlockall() from the controller thread.
            rt_cpu_core: Optional CPU core to pin the controller loop thread to.
        """
        self.hardware = hardware_interface
        self._enable_perf_tracking = enable_perf_tracking
        self._rt_priority = int(rt_priority)
        self._rt_lock_memory = bool(rt_lock_memory)
        self._rt_cpu_core = rt_cpu_core

        # Setup logger for this controller instance
        self.logger = logging.getLogger(f"{__name__}.ExcavatorController")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Only add a fallback handler when no root handler exists (standalone use).
        if not logging.root.handlers:
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

        # Store full control config reference for later use (PID, IK, velocity limits, etc.)
        self._control_config = _load_control_config()
        if not isinstance(self._control_config, dict) or not self._control_config:
            raise RuntimeError("Controller configuration requires control_config.yaml")

        # Robot configuration
        self.robot_config = load_excavator_robot_config()
        warmup_numba_functions()

        # IK controller setup (pull settings from control_config.yaml - fail hard if missing)
        ik_cfg = self._control_config.get('ik', {})
        if not ik_cfg:
            raise RuntimeError("Missing 'ik' section in control_config.yaml")

        try:
            _ik_cmd_type = ik_cfg['command_type']
            _ik_method = ik_cfg['method']
            _ik_rel = bool(ik_cfg.get('use_relative_mode', False))
            _ik_params = ik_cfg.get('params', {})
            _joint_limits_rel = ik_cfg.get('joint_limits_relative', None)
            _ik_velocity_mode = bool(ik_cfg.get('velocity_mode', False))
            _ik_velocity_error_gain = float(ik_cfg.get('velocity_error_gain', 1.0))
            _ik_use_rot_vel = bool(ik_cfg.get('use_rotational_velocity', True))
            # Adaptive damping settings (DLS method)
            _enable_adaptive_damping = bool(ik_cfg.get('enable_adaptive_damping', True))
            _adaptive_damping_max_mult = float(ik_cfg.get('adaptive_damping_max_multiplier', 2.0))
            _cond_threshold = float(ik_cfg.get('condition_number_threshold', 40.0))
        except KeyError as e:
            raise RuntimeError(f"Missing required IK configuration in control_config.yaml: {e}")

        # IK config: orientation is handled by target construction (pitch composed
        # with current slew quaternion in _refresh_smoothed_target). Uncontrollable
        # axes (roll for this robot) are auto-detected from joint rotation axes.
        # Prepare optional IK V2 extras from control_config
        ctrl_cfg = self._control_config.get('controller', {}) if isinstance(self._control_config, dict) else {}

        # Velocity limiting settings
        enable_vel_limit = bool(ctrl_cfg.get('enable_velocity_limiting', True))
        per_joint_max = ctrl_cfg.get('per_joint_max_velocity', None)
        max_joint_velocities = None
        if isinstance(per_joint_max, (list, tuple)) and len(per_joint_max) == 4:
            max_joint_velocities = [float(v) for v in per_joint_max]

        # Joint velocity estimation mode
        self._gyro_velocity_mode = str(ctrl_cfg.get('gyro_velocity_mode', 'fd_only')).strip().lower()
        if self._gyro_velocity_mode not in {'fd_only', 'gyro_only', 'fused'}:
            raise RuntimeError(
                "Invalid 'controller.gyro_velocity_mode' in control_config.yaml. "
                "Expected one of: fd_only, gyro_only, fused"
        )
        self._gyro_blend_alpha = float(np.clip(float(ctrl_cfg.get('gyro_blend_alpha', 0.30)), 0.0, 1.0))
        self._gyro_timeout_s = float(max(0.0, float(ctrl_cfg.get('gyro_timeout_s', 0.08))))
        self._gyro_max_abs_radps = float(np.radians(max(1e-3, float(ctrl_cfg.get('gyro_max_abs_degps', 180.0)))))

        bias_cfg = ctrl_cfg.get('gyro_bias_comp', {})
        self._gyro_bias_enabled = bool(bias_cfg.get('enabled', True))
        self._gyro_bias_stationarity_radps = float(
            np.radians(max(0.0, float(bias_cfg.get('stationarity_degps', 1.5))))
        )
        self._gyro_bias_adaptation_rate = float(
            np.clip(float(bias_cfg.get('adaptation_rate', 0.01)), 0.0, 1.0)
        )

        # Slew/yaw now comes from the configured IMU canonical state extractor.
        self._slew_fusion_enabled = False

        # Joint limits.  Per-joint null/[] is treated as "unbounded" — the
        # IK final clamp degenerates to a no-op (np.clip with ±inf) and the
        # joint-limit-avoidance loop skips inf-bounded joints because all of
        # its boundary comparisons go through NaN. Use this for slew when
        # continuous rotation past ±π is desired.
        joint_limits = None
        if isinstance(_joint_limits_rel, (list, tuple)) and len(_joint_limits_rel) == 4:
            # Accept degrees or radians; assume degrees if magnitudes > pi
            def _to_rad_pair(p):
                if p is None or (isinstance(p, (list, tuple)) and len(p) == 0):
                    return (float("-inf"), float("inf"))
                a, b = float(p[0]), float(p[1])
                if max(abs(a), abs(b)) > np.pi + 1e-6:
                    return (np.radians(a), np.radians(b))
                return (a, b)
            joint_limits = [_to_rad_pair(p) for p in _joint_limits_rel]

        self.ik_config = IKControllerConfig(
            command_type=_ik_cmd_type,
            ik_method=_ik_method,
            use_relative_mode=_ik_rel,
            ik_params=_ik_params,
            enable_velocity_limiting=enable_vel_limit,
            max_joint_velocities=max_joint_velocities,
            joint_limits=joint_limits,
            velocity_mode=_ik_velocity_mode,
            velocity_error_gain=_ik_velocity_error_gain,
            use_rotational_velocity=_ik_use_rot_vel,
            # Adaptive damping (DLS)
            enable_adaptive_damping=_enable_adaptive_damping,
            adaptive_damping_max_multiplier=_adaptive_damping_max_mult,
            condition_number_threshold=_cond_threshold,
        )
        # Pass verbose flag if IK controller supports it, else use DEBUG level check
        ik_verbose = self.logger.level <= logging.DEBUG
        ik_default_dt = 1.0 / float(self.config.control_frequency)
        self.ik_controller = IKController(
            self.ik_config,
            self.robot_config,
            verbose=ik_verbose,
            default_dt=ik_default_dt
        )

        # Cache IK config flags — fixed at init, avoid getattr() on every tick.
        self._ik_velocity_mode_enabled: bool = bool(getattr(self.ik_config, "velocity_mode", False))
        self._ik_use_rotational_velocity: bool = bool(getattr(self.ik_config, "use_rotational_velocity", True))

        # Condition number threshold gating — read from ik_config (shared with adaptive damping).
        self._cond_threshold = self.ik_config.condition_number_threshold
        self._cond_reject_count = 0

        # Pre-flight reachability check
        reach_cfg = self._control_config.get('reachability', {}) if isinstance(self._control_config, dict) else {}
        self._reach_enabled = bool(reach_cfg.get('enabled', True))
        self._reach_pos_tol = float(reach_cfg.get('pos_tol_m', 0.005))
        self._reach_max_iters = int(reach_cfg.get('max_iters', 80))
        self._reach_min_target_delta = float(reach_cfg.get('min_target_delta_m', 0.002))
        self._last_validated_target = None
        self._last_validated_rot_deg = None
        self._last_reachability_result: Optional[ReachabilityResult] = None

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

        # Motion processor for jerk-limited pose commands (aligned with IK loop)
        # Load motion parameters from pathing_config if available, otherwise use defaults
        if _PATHING_CONFIG is not None:
            _speed_mps = float(_PATHING_CONFIG.speed_mps)
            _max_accel = float(_PATHING_CONFIG.max_accel_mps2)
            _max_decel = float(_PATHING_CONFIG.max_decel_mps2)
            _max_jerk = float(_PATHING_CONFIG.max_jerk_mps3)
            _enable_jerk = bool(getattr(_PATHING_CONFIG, 'enable_jerk', True))
        else:
            # Sensible defaults if pathing_config not available
            _speed_mps = 0.02
            _max_accel = 0.5
            _max_decel = 0.5
            _max_jerk = 2.0
            _enable_jerk = True
        self.motion_processor = MotionProcessor(
            max_velocity=_speed_mps,
            max_acceleration=_max_accel,
            max_deceleration=_max_decel,
            max_jerk=_max_jerk,
            enable_smoothing=_enable_jerk,
        )

        # Cache hardware capability flags — hasattr() is non-trivial to call every tick.
        self._hw_has_try_read_imu_gyro: bool = hasattr(self.hardware, 'try_read_imu_gyro')

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
        self._current_fk_quats = None  # Canonical absolute link quats [slew, boom, arm, bucket]
        self._current_sensor_quats = None  # Corrected physical IMU quats in configured role order
        self._current_joint_angles = None
        self._current_ee_quat_full = None
        self._current_projected_quats = None  # Backward-compatible alias for canonical quats
        self._current_linear_velocity = 0.0
        self._current_rot_velocity_degps = 0.0
        self._target_linear_velocity = np.zeros(3, dtype=np.float32)
        self._target_rot_velocity_dps = 0.0
        self._outputs_zeroed = False  # Track whether we've already sent a zero/neutral command

        # Direct control mode (bypasses IK/PID, sends normalized commands straight to valves)
        self._direct_mode = False
        self._direct_commands = {}  # joint name -> float [-1, 1]
        self._direct_lock = threading.Lock()

        # Debug telemetry (data capture for detailed logging/analysis)
        self._last_pi_outputs = None
        self._last_named_commands = None
        self._prev_joint_angles = None
        self._prev_joint_time = None
        self._last_joint_vel_radps = None
        self._last_joint_vel_time: Optional[float] = None
        self._last_joint_vel_source = 'none'
        self._gyro_bias_radps = np.zeros(3, dtype=np.float32)
        # Pre-allocated buffers for hot-path — avoids per-tick GC pressure.
        self._gyro_joint_rates = np.zeros(3, dtype=np.float32)
        self._pi_outputs = np.zeros(4, dtype=np.float32)
        self._named_commands: Dict[str, float] = {
            'scoop': 0.0, 'lift_boom': 0.0, 'rotate': 0.0, 'tilt_boom': 0.0,
        }
        self._last_gyro_wall_t = None
        self._last_gyro_device_ts_us = None
        self._gyro_fallback_counter = 0
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
        self.hardware._check_faults()
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

    def pause(self, timeout_s: float = 0.5, reset_pump: bool = True) -> None:
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
            self.hardware.reset(reset_pump=reset_pump)
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

    def give_pose(self, position, rotation_y_deg: float = 0.0) -> Optional[ReachabilityResult]:
        target_pos = np.array(position, dtype=np.float32)
        rot_deg = float(rotation_y_deg)

        result: Optional[ReachabilityResult] = None
        if self._reach_enabled:
            result = self._evaluate_reachability(target_pos, rot_deg)
            if not result.reachable:
                self.logger.warning(
                    "Rejecting unreachable target %s rot=%.2f deg (closest=%s, "
                    "pos_err=%.4fm, cond=%.1f, iters=%d)",
                    np.round(target_pos, 4), rot_deg,
                    np.round(result.closest_position, 4),
                    result.pos_error_m, result.final_cond_number, result.iters,
                )
                return result

        with self._lock:
            self._raw_target_position = target_pos
            self._raw_target_rotation_deg = rot_deg
        # We're about to actively command again
        self._outputs_zeroed = False
        return result

    def _evaluate_reachability(
        self, target_pos: np.ndarray, rot_deg: float
    ) -> ReachabilityResult:
        """Run (or reuse a cached) reachability check for ``target_pos``."""
        if (
            self._last_reachability_result is not None
            and self._last_validated_target is not None
            and self._last_validated_rot_deg is not None
            and self._last_reachability_result.reachable
            and float(np.linalg.norm(target_pos - self._last_validated_target))
            <= self._reach_min_target_delta
            and abs(rot_deg - self._last_validated_rot_deg) <= 0.5
        ):
            return self._last_reachability_result

        with self._lock:
            cached_angles = getattr(self, '_current_joint_angles', None)
            cached_quats = getattr(self, '_current_fk_quats', None)
            current_angles = None if cached_angles is None else cached_angles.copy()
            fk_quats = None if cached_quats is None else cached_quats.copy()
        if current_angles is None and fk_quats is None:
            # No state yet — can't simulate; allow command through.
            self.logger.warning(
                "Reachability check skipped because current joint state is unavailable; allowing target %s rot=%.2f deg",
                np.round(target_pos, 4),
                rot_deg,
            )
            return ReachabilityResult(
                reachable=True,
                closest_position=target_pos.copy(),
                pos_error_m=0.0,
                iters=0,
                final_cond_number=0.0,
            )

        if current_angles is None:
            current_angles = compute_relative_joint_angles(fk_quats, self.robot_config)
        result = reachability_module.check_reachability(
            self.ik_controller,
            self.robot_config,
            current_joint_angles=current_angles,
            target_pos=target_pos,
            target_rot_y_deg=rot_deg,
            pos_tol=self._reach_pos_tol,
            max_iters=self._reach_max_iters,
            cond_threshold=self._cond_threshold,
            dt=1.0 / float(self.config.control_frequency),
        )
        if result.reachable:
            self._last_validated_target = target_pos.copy()
            self._last_validated_rot_deg = rot_deg
            self._last_reachability_result = result
        return result

    def get_last_reachability_result(self) -> Optional[ReachabilityResult]:
        return self._last_reachability_result

    def set_relative_control(self, enabled: bool) -> None:
        """Enable/disable relative IK mode.

        Args:
            enabled: Whether to use relative mode (delta pose commands)
        """
        self.ik_config.use_relative_mode = bool(enabled)
        # Reset IK internal buffers when toggling modes
        try:
            self.ik_controller.reset()
        except Exception:
            pass

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
        """Get current relative joint angles in degrees.

        Returns angles relative to each joint's parent link, matching what
        URDF viewers expect for ``setJointValue``.
        """
        with self._lock:
            if self._current_joint_angles is None:
                return np.zeros(4, dtype=np.float32)
            joint_angles = self._current_joint_angles.copy()

        return np.degrees(joint_angles)

    def get_absolute_link_angles(self) -> np.ndarray:
        """Per-sensor absolute pitch in degrees after mounting correction.

        This is the direct inclinometer-style readout for each configured IMU
        role: base/boom/arm/bucket pitch against gravity. It is not projected
        through the excavator FK model and is not relative to parent links.
        """
        with self._lock:
            if self._current_sensor_quats is None:
                return np.zeros(4, dtype=np.float32)
            quats = self._current_sensor_quats.copy()

        return np.degrees(np.array([
            gravity_pitch_from_quat(q) for q in quats
        ], dtype=np.float32))

    def get_fk_quaternions(self) -> Optional[np.ndarray]:
        """Get latest canonical absolute link quaternions [slew, boom, arm, bucket].

        These are rebuilt from the canonical joint state and are safe to
        feed into FK/debug visualization. They are not the raw physical IMU quats.
        """
        with self._lock:
            if self._current_fk_quats is None:
                return None
            return np.array(self._current_fk_quats, dtype=np.float32, copy=True)

    def get_condition_number(self) -> float:
        """Get the Jacobian condition number from the last IK solve."""
        return float(self.ik_controller.last_condition_number)

    def get_yoshikawa_index(self) -> float:
        """Yoshikawa manipulability index from the last IK solve."""
        return float(self.ik_controller.last_yoshikawa_index)

    def get_singular_values(self) -> np.ndarray:
        """Jacobian singular values (4-element float32) from the last IK solve."""
        return self.ik_controller.last_singular_values.copy()

    def get_hardware_status(self) -> dict:
        """Get hardware status including all ADC channels for logging."""
        return self.hardware.get_status()

    def get_performance_stats(self) -> dict:
        """Get controller loop performance statistics.

        Returns dict with:
        - Loop timing: avg/std/min/max/p95/p99 in ms
        - Compute timing: avg/std/min/max in ms
        - Headroom, loop utilization, and whole-process CPU usage
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
                'max_step_ms': 0.0,
                'avg_headroom_ms': 0.0,
                'cpu_usage_pct': 0.0,
                'loop_util_pct': 0.0,
                'process_cpu_pct': 0.0,
                'actual_hz': 0.0,
                'violation_pct': 0.0,
                'violation_count': 0,
                'compute_overrun_pct': 0.0,
                'compute_overrun_count': 0,
                'compute_overrun_count_recent': 0,
                'compute_overrun_pct_recent': 0.0,
                'deadline_miss_count': 0,
                'deadline_miss_pct': 0.0,
                'deadline_miss_count_recent': 0,
                'deadline_miss_pct_recent': 0.0,
                'deadline_miss_1pct_count': 0,
                'deadline_miss_1pct_pct': 0.0,
                'deadline_miss_1pct_count_recent': 0,
                'deadline_miss_1pct_pct_recent': 0.0,
                'deadline_window_sec': 0.0,
                'overrun_count_recent': 0,
                'overrun_pct_recent': 0.0,
                'overrun_window_sec': 0.0,
                'sample_count': 0,
                # Stage timings
                'avg_sensor_ms': 0.0, 'min_sensor_ms': 0.0, 'max_sensor_ms': 0.0,
                'avg_ik_fk_ms': 0.0, 'min_ik_fk_ms': 0.0, 'max_ik_fk_ms': 0.0,
                'avg_pwm_ms': 0.0, 'min_pwm_ms': 0.0, 'max_pwm_ms': 0.0,
                # IK telemetry
                'ik_vel_lim_enabled': bool(getattr(self.ik_config, 'enable_velocity_limiting', False)),
                'last_joint_vel_degps': [] if self._last_joint_vel_radps is None else list(np.degrees(self._last_joint_vel_radps)),
                'effective_vel_cap_degps': [],
                'gyro_velocity_mode': self._gyro_velocity_mode,
                'joint_velocity_source': self._last_joint_vel_source,
                'gyro_fallback_count': int(self._gyro_fallback_counter),
                'slew_fusion_enabled': False,
                'slew_fusion_active': False,
                'slew_fusion_gyro_z_degps': 0.0,
                'slew_fusion_alpha': 0.0,
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
                'max_step_ms': tracker_stats.get('max_step_ms', 0.0),
                'avg_compute_time_ms': tracker_stats.get('compute_avg_ms', 0.0),
                'min_compute_time_ms': tracker_stats.get('compute_min_ms', 0.0),
                'max_compute_time_ms': tracker_stats.get('compute_max_ms', 0.0),
                'std_compute_time_ms': tracker_stats.get('compute_std_ms', 0.0),
                'avg_headroom_ms': tracker_stats.get('headroom_avg_ms', 0.0),
                'cpu_usage_pct': tracker_stats.get('cpu_usage_pct', 0.0),
                'loop_util_pct': tracker_stats.get('loop_util_pct', tracker_stats.get('cpu_usage_pct', 0.0)),
                'process_cpu_pct': tracker_stats.get('process_cpu_pct', 0.0),
                'actual_hz': actual_hz,
                'violation_pct': tracker_stats.get('violation_pct', 0.0),
                'violation_count': tracker_stats.get('violation_count', 0),
                'compute_overrun_pct': tracker_stats.get('compute_overrun_pct', tracker_stats.get('violation_pct', 0.0)),
                'compute_overrun_count': tracker_stats.get('compute_overrun_count', tracker_stats.get('violation_count', 0)),
                'compute_overrun_count_recent': tracker_stats.get('compute_overrun_count_recent', 0),
                'compute_overrun_pct_recent': tracker_stats.get('compute_overrun_pct_recent', 0.0),
                'deadline_miss_count': tracker_stats.get('deadline_miss_count', 0),
                'deadline_miss_pct': tracker_stats.get('deadline_miss_pct', 0.0),
                'deadline_miss_count_recent': tracker_stats.get('deadline_miss_count_recent', tracker_stats.get('overrun_count_recent', 0)),
                'deadline_miss_pct_recent': tracker_stats.get('deadline_miss_pct_recent', tracker_stats.get('overrun_pct_recent', 0.0)),
                'deadline_miss_1pct_count': tracker_stats.get('deadline_miss_1pct_count', 0),
                'deadline_miss_1pct_pct': tracker_stats.get('deadline_miss_1pct_pct', 0.0),
                'deadline_miss_1pct_count_recent': tracker_stats.get('deadline_miss_1pct_count_recent', 0),
                'deadline_miss_1pct_pct_recent': tracker_stats.get('deadline_miss_1pct_pct_recent', 0.0),
                'deadline_window_sec': tracker_stats.get('deadline_window_sec', tracker_stats.get('overrun_window_sec', 0.0)),
                'overrun_count_recent': tracker_stats.get('overrun_count_recent', 0),
                'overrun_pct_recent': tracker_stats.get('overrun_pct_recent', 0.0),
                'overrun_window_sec': tracker_stats.get('overrun_window_sec', 0.0),
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
                'gyro_velocity_mode': self._gyro_velocity_mode,
                'joint_velocity_source': self._last_joint_vel_source,
                'gyro_fallback_count': int(self._gyro_fallback_counter),
                'slew_fusion_enabled': False,
                'slew_fusion_active': False,
                'slew_fusion_gyro_z_degps': 0.0,
                'slew_fusion_alpha': 0.0,
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

    def set_velocity_mode(self, mode: str) -> None:
        """Set joint velocity estimation mode: 'fd_only', 'gyro_only', or 'fused'."""
        valid = ('fd_only', 'gyro_only', 'fused')
        mode = mode.strip().lower()
        if mode not in valid:
            raise ValueError(f"velocity mode must be one of {valid}, got {mode!r}")
        if mode == self._gyro_velocity_mode:
            return
        self._gyro_velocity_mode = mode
        self._last_gyro_device_ts_us = None
        self._last_gyro_wall_t = None
        self._gyro_fallback_counter = 0
        self._gyro_bias_radps = np.zeros(3, dtype=np.float32)
        self.logger.info("Joint velocity mode set to %s", mode)

    def get_joint_velocities_with_age(self) -> tuple:
        """Return (velocities_degps, age_s). age=inf if never computed."""
        if self._last_joint_vel_radps is None or self._last_joint_vel_time is None:
            return None, float('inf')
        return list(np.degrees(self._last_joint_vel_radps)), time.perf_counter() - self._last_joint_vel_time

    def get_last_pid_outputs(self):
        """Return last per-joint PID outputs [slew, boom, arm, bucket] or None if not captured."""
        return None if self._last_pi_outputs is None else list(self._last_pi_outputs)

    def get_last_pwm_commands(self):
        """Return last named PWM command dict or empty dict if none."""
        return {} if self._last_named_commands is None else dict(self._last_named_commands)

    # -------------- Direct control mode (bypass IK/PID) --------------

    def enter_direct_mode(self) -> None:
        """Switch to direct valve control, bypassing IK and PID."""
        with self._direct_lock:
            self._direct_commands = {}
        for pid in self.joint_pids:
            pid.reset()
        with self._lock:
            self._direct_mode = True
        self._outputs_zeroed = False
        self.logger.info("Entered direct control mode")

    def exit_direct_mode(self) -> None:
        """Switch back to IK mode, syncing targets to current measured pose."""
        with self._direct_lock:
            self._direct_commands = {}

        # Reset PIDs to avoid residual terms
        for pid in self.joint_pids:
            pid.reset()

        # Reset IK internal command buffers
        try:
            self.ik_controller.reset()
        except Exception:
            pass

        # Sync smoother and IK target to current measured EE so there's no jump
        self._update_current_state()
        self.motion_processor.reset(
            position=self._current_position,
            rotation_deg=self._current_orientation_y_deg
        )
        # Set raw target to current pose so give_pose() starts from reality
        self.give_pose(self._current_position, self._current_orientation_y_deg)

        with self._lock:
            self._direct_mode = False
        self._outputs_zeroed = False
        self.logger.info("Exited direct control mode (synced to measured pose)")

    def give_direct_commands(self, commands: dict) -> None:
        """Set normalized [-1, 1] valve commands for direct mode.

        Args:
            commands: dict of joint name -> float, e.g.
                      {'rotate': 0.3, 'lift_boom': -0.5, 'tilt_boom': 0.0, 'scoop': 0.0}
        """
        with self._direct_lock:
            self._direct_commands = dict(commands)
        self._outputs_zeroed = False

    def _send_direct_commands(self) -> None:
        """Read current direct commands and send them to hardware."""
        with self._direct_lock:
            cmds = dict(self._direct_commands)

        if not cmds:
            if not self._outputs_zeroed:
                self.hardware.reset(reset_pump=False)
                self._outputs_zeroed = True
            return

        self.hardware.send_named_pwm_commands(cmds)
        self._outputs_zeroed = False

    def get_ik_debug_info(self) -> dict:
        """Return IK telemetry such as adaptive damping and condition number."""
        return {
            'adaptive_lambda': float(self.ik_controller.last_adaptive_lambda),
            'condition_number': float(self.ik_controller.last_condition_number),
            'cond_reject_count': self._cond_reject_count,
        }

    def get_joint_velocities_degps(self):
        """Return last estimated joint velocities (deg/s) or None if not captured."""
        if self._last_joint_vel_radps is None:
            return None
        return list(np.degrees(self._last_joint_vel_radps))

    def _compute_fd_joint_velocity(self, current_joint_angles: np.ndarray, now_t: float) -> Optional[np.ndarray]:
        """Estimate joint velocity from finite-difference angle derivative (rad/s)."""
        fd_joint_vel = None
        if self._prev_joint_angles is not None and self._prev_joint_time is not None:
            dtj = max(1e-6, now_t - self._prev_joint_time)
            fd_joint_vel = (current_joint_angles - self._prev_joint_angles) / dtj
        self._prev_joint_angles = current_joint_angles.copy()
        self._prev_joint_time = now_t
        return fd_joint_vel

    def _compute_gyro_joint_velocity(self, now_t: float) -> Optional[np.ndarray]:
        """Estimate joint velocity from IMU gyro and project to joint axes (rad/s)."""
        if self._gyro_velocity_mode == 'fd_only':
            return None

        payload = None
        try:
            if self._hw_has_try_read_imu_gyro:
                payload = self.hardware.try_read_imu_gyro()
            else:
                gyro = self.hardware.read_imu_gyro()
                if gyro is not None:
                    payload = {'gyro': gyro, 'device_timestamp_us': None}
        except Exception:
            payload = None

        if not payload:
            return None

        gyro_packets = payload.get('gyro')
        if gyro_packets is None or len(gyro_packets) < 3:
            return None

        dev_ts = payload.get('device_timestamp_us')
        if isinstance(dev_ts, (int, float)):
            if self._last_gyro_device_ts_us != dev_ts:
                self._last_gyro_device_ts_us = dev_ts
                self._last_gyro_wall_t = now_t
            elif self._last_gyro_wall_t is not None and self._gyro_timeout_s > 0.0:
                if (now_t - self._last_gyro_wall_t) > self._gyro_timeout_s:
                    return None

        joint_rates = self._gyro_joint_rates
        joint_rates[:] = 0.0
        for i in range(3):
            gyro_vec_dps = np.asarray(gyro_packets[i], dtype=np.float32)
            gyro_vec_rad = np.radians(gyro_vec_dps)

            axis = self.robot_config.rotation_axes[i + 1]
            axis_norm = axis / (np.linalg.norm(axis) + 1e-12)
            joint_rates[i] = float(np.dot(gyro_vec_rad, axis_norm))

        if self._gyro_bias_enabled:
            if np.max(np.abs(joint_rates)) <= self._gyro_bias_stationarity_radps:
                k = self._gyro_bias_adaptation_rate
                self._gyro_bias_radps = (1.0 - k) * self._gyro_bias_radps + k * joint_rates
            joint_rates = joint_rates - self._gyro_bias_radps

        joint_rates = np.clip(joint_rates, -self._gyro_max_abs_radps, self._gyro_max_abs_radps)

        out = np.zeros(4, dtype=np.float32)
        out[1:] = joint_rates
        return out

    def _select_joint_velocity(
        self,
        fd_joint_vel: Optional[np.ndarray],
        gyro_joint_vel: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Select/control fusion of joint velocity estimate by configured mode."""
        mode = self._gyro_velocity_mode

        if mode == 'fd_only':
            self._last_joint_vel_source = 'fd'
            return None if fd_joint_vel is None else fd_joint_vel.astype(np.float32)

        if gyro_joint_vel is None:
            self._gyro_fallback_counter += 1
            if self._gyro_fallback_counter % 200 == 1:
                self.logger.warning("Gyro velocity unavailable; falling back to finite-difference estimate")
            self._last_joint_vel_source = 'fd_fallback'
            return None if fd_joint_vel is None else fd_joint_vel.astype(np.float32)

        # Gyro provides boom/arm/bucket; slew remains finite-difference based.
        out = np.zeros(4, dtype=np.float32)
        if fd_joint_vel is not None:
            out[:] = fd_joint_vel
        elif self._last_joint_vel_radps is not None:
            out[:] = self._last_joint_vel_radps

        if mode == 'gyro_only':
            out[1:] = gyro_joint_vel[1:]
            self._last_joint_vel_source = 'gyro'
            return out

        alpha = self._gyro_blend_alpha
        if fd_joint_vel is None:
            out[1:] = gyro_joint_vel[1:]
            self._last_joint_vel_source = 'gyro_fallback'
            return out

        out[1:] = alpha * gyro_joint_vel[1:] + (1.0 - alpha) * fd_joint_vel[1:]
        self._last_joint_vel_source = 'fused'
        return out

    def _update_joint_velocity_estimate(self, current_joint_angles: np.ndarray, now_t: float) -> None:
        """Update joint velocity telemetry/control estimate from current state and IMU gyro."""
        fd_joint_vel = self._compute_fd_joint_velocity(current_joint_angles, now_t)
        gyro_joint_vel = self._compute_gyro_joint_velocity(now_t=now_t)
        selected_joint_vel = self._select_joint_velocity(fd_joint_vel, gyro_joint_vel)
        if selected_joint_vel is not None:
            self._last_joint_vel_radps = selected_joint_vel
            self._last_joint_vel_time = now_t

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._perf_tracker.reset()
        try:
            if self._enable_perf_tracking and hasattr(self.hardware, 'reset_perf_stats'):
                self.hardware.reset_perf_stats()
        except Exception:
            pass

    def _get_sensor_quaternions(self) -> Optional[np.ndarray]:
        """
        Read corrected configured IMU sensor data.

        Returns:
            Corrected physical IMU quaternions in configured role order, or None if hardware not ready
        """
        try:
            quaternions = self.hardware.read_all_imu_quaternions()
            if quaternions is None or len(quaternions) == 0:
                return None
            return np.array(quaternions, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error reading quaternions: {e}")
            return None

    def _control_loop(self) -> None:
        if self._rt_priority > 0 or self._rt_lock_memory or self._rt_cpu_core is not None:
            cpu_affinity = None if self._rt_cpu_core is None else {int(self._rt_cpu_core)}
            success = apply_rt_to_thread(
                priority=self._rt_priority,
                policy=SCHED_FIFO,
                lock_memory=self._rt_lock_memory,
                cpu_affinity=cpu_affinity,
                quiet=False,
            )
            if success:
                details = []
                if self._rt_priority > 0:
                    details.append(f"SCHED_FIFO-{self._rt_priority}")
                if self._rt_lock_memory:
                    details.append("mlockall")
                if self._rt_cpu_core is not None:
                    details.append(f"core {self._rt_cpu_core}")
                self.logger.info("Control thread: applied %s", ", ".join(details) if details else "RT settings")
            else:
                self.logger.warning("Control thread: Failed to apply requested RT settings")

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

                with self._lock:
                    direct_mode = self._direct_mode

                if direct_mode:
                    # Direct mode: bypass smoothing, IK, and PID
                    self._perf_tracker.stage_start('pwm')
                    self._send_direct_commands()
                    self._perf_tracker.stage_end('pwm')
                else:
                    # IK mode: smooth -> IK -> PID -> PWM
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
            # Get corrected physical IMU quaternions in configured role order.
            sensor_quats = self._get_sensor_quaternions()
            if sensor_quats is None:
                # Invalidate stale cached state so _compute_control_commands()
                # cannot proceed with old data while sensors are unavailable.
                with self._lock:
                    self._current_projected_quats = None
                    self._current_fk_quats = None
                    self._current_joint_angles = None
                return

            # Convert sensor quats once into canonical relative joint angles.
            joint_angles = canonical_joint_angles_from_imus(sensor_quats, self.robot_config)
            joint_quats = joint_angles_to_absolute_quaternions(joint_angles, self.robot_config)
            ee_pos, ee_quat = get_pose_from_joint_angles(joint_angles, self.robot_config)

            # Extract Y-axis rotation for end-effector orientation in body frame.
            # Remove slew (Z-axis) rotation so pitch is relative to the upper body,
            # not the world frame.  This keeps the reported pitch independent of
            # the excavator's heading.
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            slew_quat_inv = quat_conjugate(joint_quats[0])
            ee_quat_body = quat_normalize(quat_multiply(slew_quat_inv, ee_quat))
            ee_y_angle_rad = extract_axis_rotation(ee_quat_body, y_axis)
            ee_y_angle_deg = np.degrees(ee_y_angle_rad)

            # Update shared state
            with self._lock:
                self._current_position = ee_pos
                self._current_orientation_y_deg = ee_y_angle_deg
                self._current_projected_quats = joint_quats
                self._current_fk_quats = joint_quats
                self._current_sensor_quats = sensor_quats
                self._current_joint_angles = joint_angles
                self._current_ee_quat_full = ee_quat

            now_t = time.perf_counter()
            self._update_joint_velocity_estimate(joint_angles, now_t)

            # Update simple EE velocity estimates for smoother seeding
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
            fk_quats = self._current_fk_quats

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

        # Build target orientation: pitch is in body frame, so compose with
        # the current slew rotation to express it in world frame.
        # This ensures the target quaternion's yaw always matches the robot's
        # actual slew yaw — the IK sees zero yaw error by construction.
        # TODO: add orientation_mode here (pitch_follows_slew / full_pose) for tool orientation support.
        pitch_quat = quat_from_axis_angle(_Y_AXIS, np.radians(smoothed_rot))
        if fk_quats is not None:
            slew_quat = fk_quats[0]
            smoothed_quat = quat_normalize(quat_multiply(slew_quat, pitch_quat))
        else:
            smoothed_quat = pitch_quat

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
                joint_angles = None if self._current_joint_angles is None else self._current_joint_angles.copy()
                joint_quats = self._current_fk_quats
                current_ee_quat_full = self._current_ee_quat_full
                target_lin_vel = self._target_linear_velocity.copy()
                target_rot_vel_dps = float(self._target_rot_velocity_dps)

            if joint_angles is None or joint_quats is None:
                return
            current_joint_angles = joint_angles

            # Outer Loop: Task-space IK control
            # Note: Using current_ee_quat_full (including Z-rotation from slew) for IK.
            # The Jacobian's structure naturally constrains which rotations each joint can achieve
            # based on rotation_axes config (slew=Z, boom/arm/bucket=Y).
            if self.ik_config.use_relative_mode:
                # Compute error w.r.t target
                if self.ik_config.command_type == "position":
                    pos_err = target_pos - current_pos
                    self.ik_controller.set_command(pos_err, ee_pos=current_pos, ee_quat=current_ee_quat_full)
                else:
                    # Pose: 6D delta [dx, dy, dz, rx, ry, rz]
                    # Orientation locks are now applied internally by IK controller
                    pos_err, axis_angle_err = compute_pose_error(
                        current_pos, current_ee_quat_full, target_pos, target_quat
                    )
                    delta_pose = np.concatenate([pos_err, axis_angle_err])
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
            if self._ik_velocity_mode_enabled:
                if self.ik_config.command_type == "position":
                    desired_vel = target_lin_vel
                else:
                    if self._ik_use_rotational_velocity:
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
                joint_quats=joint_quats,
                desired_ee_velocity=desired_vel,
                dt=loop_dt,
                current_joint_velocities=self._last_joint_vel_radps,
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

        # Reject commands when Jacobian is near-singular (just skip, don't reset hardware)
        if self._cond_threshold > 0 and self.ik_controller.last_condition_number > self._cond_threshold:
            self._cond_reject_count += 1
            if self._cond_reject_count % 200 == 1:
                self.logger.warning(
                    f"Condition number {self.ik_controller.last_condition_number:.1f} "
                    f"exceeds threshold {self._cond_threshold:.1f}, skipping IK output"
                )
            return

        # Inner Loop: Joint-space PID — fills pre-allocated buffer in-place.
        pi_outputs = self._pi_outputs
        for i, (pid, target_angle, current_angle) in enumerate(
                zip(self.joint_pids, target_joint_angles, current_joint_angles)
        ):
            pi_outputs[i] = pid.compute(0.0, -_angle_error(target_angle, current_angle), dt=loop_dt)

        self._named_commands['scoop']     = float(pi_outputs[3])  # bucket
        self._named_commands['lift_boom'] = float(pi_outputs[1])  # boom
        self._named_commands['rotate']    = float(pi_outputs[0])  # slew
        self._named_commands['tilt_boom'] = float(pi_outputs[2])  # arm
        named_commands = self._named_commands

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
