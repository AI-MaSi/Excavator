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
from typing import Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Import project modules
from . import diff_ik_V2 as diff_ik
from .pid import PIDController
from .quaternion_math import quat_from_axis_angle

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

    Note: All parameters are now loaded from configuration_files/general_config.yaml
    This class is populated dynamically at runtime.
    """
    output_limits: Tuple[float, float]
    control_frequency: float  # Hz


class ExcavatorController:
    def __init__(self, hardware_interface, config: Optional[ControllerConfig] = None,
                 enable_perf_tracking: bool = False, verbose: bool = True):
        self.hardware = hardware_interface
        self._enable_perf_tracking = enable_perf_tracking
        self._verbose = verbose

        # Cascade perf flag to hardware if supported (no-op otherwise)
        try:
            if hasattr(self.hardware, 'set_perf_enabled'):
                self.hardware.set_perf_enabled(bool(enable_perf_tracking))
        except Exception:
            pass

        # Load controller configuration from general_config.yaml if not provided
        if config is None:
            gc = getattr(self.hardware, '_general_config', None)
            if not isinstance(gc, dict):
                raise RuntimeError("Controller configuration requires general_config.yaml")

            # Load control frequency from rates section
            control_hz = gc.get('rates', {}).get('control_hz')
            if not isinstance(control_hz, (int, float)) or control_hz <= 0:
                raise RuntimeError("Missing or invalid 'rates.control_hz' in general_config.yaml")

            # Load controller parameters
            ctrl_cfg = gc.get('controller', {})
            if not ctrl_cfg:
                raise RuntimeError("Missing 'controller' section in general_config.yaml")

            try:
                self.config = ControllerConfig(
                    output_limits=(
                        float(ctrl_cfg['output_limits_min']),
                        float(ctrl_cfg['output_limits_max'])
                    ),
                    control_frequency=float(control_hz)
                )
            except KeyError as e:
                raise RuntimeError(f"Missing required controller parameter in general_config.yaml: {e}")
        else:
            self.config = config

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
        except AttributeError as e:
            raise RuntimeError(f"Missing required IK configuration in pathing_config.py: {e}")

        # IK config uses ignore_axes for orientation filtering
        # Prepare optional IK V2 extras from general_config
        gc = getattr(self.hardware, '_general_config', {}) or {}
        ctrl_cfg = gc.get('controller', {}) if isinstance(gc, dict) else {}

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
            joint_limits=joint_limits
        )
        self.ik_controller = diff_ik.IKController(self.ik_config, self.robot_config, verbose=self._verbose)  # Advanced with weighting

        # Relative IK control parameters (pulling toward target)
        try:
            self._relative_pos_gain = float(_DEFAULT_CONFIG.relative_pos_gain)
            self._relative_rot_gain = float(_DEFAULT_CONFIG.relative_rot_gain)
        except AttributeError as e:
            raise RuntimeError(f"Missing required relative control gains in pathing_config.py: {e}")

        # Load PID gains from general_config.yaml
        gc = getattr(self.hardware, '_general_config', None)
        if not isinstance(gc, dict) or 'pid' not in gc:
            raise RuntimeError("PID configuration missing from general_config.yaml")

        pid_cfg = gc['pid']
        joint_configs = []
        for i, name in enumerate(["Slew", "Boom", "Arm", "Bucket"]):
            joint_key = f'joint{i}'
            if joint_key not in pid_cfg:
                raise RuntimeError(f"Missing PID config for {joint_key} in general_config.yaml")
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

        # Thread control
        self._control_thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # State variables
        self._target_position = None
        self._target_orientation = None
        self._current_position = np.zeros(3, dtype=np.float32)
        self._current_orientation_y_deg = 0.0
        self._current_projected_quats = None  # Cache processed quaternions
        self._outputs_zeroed = False  # Track whether we've already sent a zero/neutral command
        # Gated debug telemetry (disabled by default to avoid overhead)
        self._debug_telemetry_enabled = False
        self._last_pi_outputs = None
        self._last_named_commands = None
        self._prev_joint_angles = None
        self._prev_joint_time = None
        self._last_joint_vel_radps = None

        # Performance tracking
        self._loop_times = []
        self._compute_times = []  # Actual computation time (without sleep)
        self._timing_violations = 0
        self._loop_count = 0
        self._perf_lock = threading.Lock()

        # Detailed stage timing (sensors, IK/FK, PWM control)
        self._sensor_times = []
        self._ik_fk_times = []
        self._pwm_times = []

        print("Controller initialized")

    def start(self) -> None:
        if self._control_thread is not None:
            print("Controller already running!")
            return

        self._stop_event.clear()
        self._pause_event.clear()  # Start unpaused
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("Control loop started")

    def stop(self) -> None:
        if self._control_thread is None:
            return

        self._stop_event.set()
        self._control_thread.join(timeout=2.0)
        self._control_thread = None
        # Safe actuator state
        self.hardware.reset(reset_pump=True)
        # Ensure background hardware threads and serial are closed
        try:
            self.hardware.shutdown()
        except Exception:
            pass

    def pause(self) -> None:
        """Pause the control loop and clear any pending IK target."""
        if self._control_thread is None:
            print("Controller not running - cannot pause")
            return

        # Engage paused state immediately so the loop stops computing commands
        self._pause_event.set()

        # Clear target and controller states to avoid stale jumps on resume
        self.clear_target()
        # Ensure hardware is commanded to safe (zero) outputs while paused
        # This guarantees actuators hold still during path planning
        try:
            self.hardware.reset(reset_pump=False)
        except Exception:
            pass
        self._outputs_zeroed = True
        print("Controller paused (target cleared)")

    def resume(self) -> None:
        """Resume the control loop from paused state."""
        if self._control_thread is None:
            print("Controller not running - cannot resume")
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

        self._pause_event.clear()

    def give_pose(self, position, rotation_y_deg: float = 0.0) -> None:
        with self._lock:
            self._target_position = np.array(position, dtype=np.float32)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            y_rotation_rad = np.radians(rotation_y_deg)
            self._target_orientation = quat_from_axis_angle(y_axis, y_rotation_rad)
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
            self._target_position = None
            self._target_orientation = None

        # Reset PIDs to avoid residual terms
        for pid in self.joint_pids:
            pid.reset()

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
        """Get controller loop performance statistics."""
        with self._perf_lock:
            if not self._loop_times:
                return {
                    'avg_loop_time_ms': 0.0,
                    'min_loop_time_ms': 0.0,
                    'max_loop_time_ms': 0.0,
                    'std_loop_time_ms': 0.0,
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
                    'avg_sensor_ms': 0.0,
                    'min_sensor_ms': 0.0,
                    'max_sensor_ms': 0.0,
                    'avg_ik_fk_ms': 0.0,
                    'min_ik_fk_ms': 0.0,
                    'max_ik_fk_ms': 0.0,
                    'avg_pwm_ms': 0.0,
                    'min_pwm_ms': 0.0,
                    'max_pwm_ms': 0.0,
                    # Additional IK-related telemetry
                    'ik_vel_lim_enabled': bool(getattr(self.ik_config, 'enable_velocity_limiting', False)),
                    'last_joint_vel_degps': [] if self._last_joint_vel_radps is None else list(np.degrees(self._last_joint_vel_radps)),
                    'effective_vel_cap_degps': [],
                }

            loop_times_ms = np.array(self._loop_times) * 1000.0
            compute_times_ms = np.array(self._compute_times) * 1000.0
            sensor_times_ms = np.array(self._sensor_times) * 1000.0
            ik_fk_times_ms = np.array(self._ik_fk_times) * 1000.0
            pwm_times_ms = np.array(self._pwm_times) * 1000.0

            actual_hz = 1.0 / np.mean(self._loop_times)
            violation_pct = (self._timing_violations / self._loop_count * 100) if self._loop_count > 0 else 0.0

            # Calculate headroom (time available for sleep)
            target_period_ms = 1000.0 / self.config.control_frequency
            avg_headroom_ms = target_period_ms - np.mean(compute_times_ms)

            # Calculate CPU usage percentage (compute time / total time)
            cpu_usage_pct = (np.mean(compute_times_ms) / target_period_ms) * 100.0

            stats = {
                'avg_loop_time_ms': float(np.mean(loop_times_ms)),
                'min_loop_time_ms': float(np.min(loop_times_ms)),
                'max_loop_time_ms': float(np.max(loop_times_ms)),
                'std_loop_time_ms': float(np.std(loop_times_ms)),
                'avg_compute_time_ms': float(np.mean(compute_times_ms)),
                'min_compute_time_ms': float(np.min(compute_times_ms)),
                'max_compute_time_ms': float(np.max(compute_times_ms)),
                'std_compute_time_ms': float(np.std(compute_times_ms)),
                'avg_headroom_ms': float(avg_headroom_ms),
                'cpu_usage_pct': float(cpu_usage_pct),
                'actual_hz': float(actual_hz),
                'violation_pct': float(violation_pct),
                'sample_count': self._loop_count,
                # Stage timings
                'avg_sensor_ms': float(np.mean(sensor_times_ms)),
                'min_sensor_ms': float(np.min(sensor_times_ms)),
                'max_sensor_ms': float(np.max(sensor_times_ms)),
                'avg_ik_fk_ms': float(np.mean(ik_fk_times_ms)),
                'min_ik_fk_ms': float(np.min(ik_fk_times_ms)),
                'max_ik_fk_ms': float(np.max(ik_fk_times_ms)),
                'avg_pwm_ms': float(np.mean(pwm_times_ms)),
                'min_pwm_ms': float(np.min(pwm_times_ms)),
                'max_pwm_ms': float(np.max(pwm_times_ms)),
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
                # rad/iter * Hz -> rad/s, then convert to deg/s
                cap_degps = np.degrees(self.ik_controller.max_joint_velocities) * actual_hz
                stats['effective_vel_cap_degps'] = list(cap_degps)
            else:
                stats['effective_vel_cap_degps'] = []

        # Optionally merge hardware perf stats (kept outside perf_lock to avoid long holds)
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

    def set_debug_telemetry(self, enabled: bool) -> None:
        """Enable/disable capturing of extra telemetry for logging.

        When enabled, caches last PID outputs, PWM commands, and joint velocities.
        Also cascades to hardware to enable IMU gyro capture.
        """
        self._debug_telemetry_enabled = bool(enabled)
        if hasattr(self.hardware, 'set_debug_telemetry_enabled'):
            self.hardware.set_debug_telemetry_enabled(bool(enabled))

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
        with self._perf_lock:
            self._loop_times = []
            self._compute_times = []
            self._sensor_times = []
            self._ik_fk_times = []
            self._pwm_times = []
            self._timing_violations = 0
            self._loop_count = 0
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
            print(f"Error reading quaternions: {e}")
            return None

    def _control_loop(self) -> None:
        loop_period = 1.0 / self.config.control_frequency
        next_run_time = time.perf_counter()
        last_loop_start = next_run_time

        while not self._stop_event.is_set():
            loop_start_time = time.perf_counter()

            # Check if paused - if so, just sleep and continue
            if self._pause_event.is_set():
                next_run_time = time.perf_counter() + 0.1  # Reset timing when paused
                last_loop_start = next_run_time
                time.sleep(0.1)  # Sleep while paused
                continue

            # === Performance tracking (only if enabled) ===
            if self._enable_perf_tracking:
                compute_start_time = time.perf_counter()
                t_sensor_start = compute_start_time

            try:
                self._update_current_state()

                if self._enable_perf_tracking:
                    t_sensor_end = time.perf_counter()
                    t_ik_fk_start = t_sensor_end

                with self._lock:
                    has_target = self._target_position is not None
                if has_target:
                    self._compute_control_commands()
                else:
                    # Only send a neutral command once when there is no target
                    if not self._outputs_zeroed:
                        self.hardware.reset(reset_pump=False)
                        self._outputs_zeroed = True

                if self._enable_perf_tracking:
                    t_ik_fk_end = time.perf_counter()

            except Exception as e:
                print(f"Control loop error: {e}")
                self.hardware.reset(reset_pump=True)
                break

            # === Track performance metrics ===
            if self._enable_perf_tracking:
                compute_end_time = time.perf_counter()
                actual_compute_time = compute_end_time - compute_start_time
                actual_loop_period = loop_start_time - last_loop_start

                # Collect stage timings
                sensor_time = t_sensor_end - t_sensor_start
                ik_time = getattr(self, '_last_ik_time', 0.0)
                pwm_time = getattr(self, '_last_pwm_time', 0.0)

                if actual_loop_period > 0.001:  # Ignore first loop
                    with self._perf_lock:
                        self._loop_times.append(actual_loop_period)
                        self._compute_times.append(actual_compute_time)
                        self._sensor_times.append(sensor_time)
                        self._ik_fk_times.append(ik_time)
                        self._pwm_times.append(pwm_time)
                        self._loop_count += 1

                        # Check for timing violations
                        if actual_compute_time > loop_period:
                            self._timing_violations += 1

                        # Keep only last 1000 samples
                        if len(self._loop_times) > 1000:
                            self._loop_times.pop(0)
                            self._compute_times.pop(0)
                            self._sensor_times.pop(0)
                            self._ik_fk_times.pop(0)
                            self._pwm_times.pop(0)

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

        except Exception as e:
            print(f"Error in state update: {e}")

    def _compute_control_commands(self) -> None:
        """Compute and send control commands based on current state and target."""
        # Track IK/FK timing
        if self._enable_perf_tracking:
            t_ik_start = time.perf_counter()

        try:
            # Get cached state from _update_current_state()
            with self._lock:
                if self._current_projected_quats is None:
                    return

                target_pos = self._target_position.copy()
                target_quat = self._target_orientation.copy()
                current_pos = self._current_position.copy()
                projected_quats = self._current_projected_quats  # Use cached quaternions
                raw_quats = self._current_raw_quats
                current_ee_quat_full = self._current_ee_quat_full

            # Extract current joint angles using V2 helper (relative angles)
            if raw_quats is None:
                return
            current_joint_angles = diff_ik.compute_relative_joint_angles(raw_quats, self.robot_config)

            # End-effector orientation quaternion to use in IK
            # Use full orientation (including Z-rotation from slew)
            # The Jacobian's structure naturally constrains which rotations each joint can achieve
            # based on rotation_axes config (slew=Z, boom/arm/bucket=Y)
            current_ee_quat = current_ee_quat_full

            # Outer Loop: Task-space IK control
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

            target_joint_angles = self.ik_controller.compute(
                ee_pos=current_pos,
                ee_quat=current_ee_quat_full,  # Full orientation incl. slew yaw
                joint_angles=current_joint_angles,
                joint_quats=raw_quats  # Pass raw IMU quats; V2 handles offsets/propagation
            )

            # DEBUG: Check if IK produced output
            if hasattr(self, '_ik_debug_counter'):
                self._ik_debug_counter += 1
            else:
                self._ik_debug_counter = 0

            if self._verbose and self._ik_debug_counter % 50 == 0:
                if target_joint_angles is not None:
                    delta_angles = target_joint_angles - current_joint_angles
                    delta_deg = np.degrees(delta_angles)
                    max_delta = np.max(np.abs(delta_deg))
                    if max_delta > 0.1:
                        print(f"[IK] Delta: {delta_deg} deg (max={max_delta:.2f})")
                else:
                    print("[IK] WARNING: IK returned None!")

        except Exception as e:
            print(f"Error in control computation: {e}")
            import traceback
            traceback.print_exc()
            return

        if target_joint_angles is None:
            # If IK failed, command neutral once (avoid constant zeroing)
            if not self._outputs_zeroed:
                print("WARNING: IK returned None, sending neutral command")
                self.hardware.reset(reset_pump=False)
                self._outputs_zeroed = True
            return

        def angle_error(target, current):
            return np.arctan2(np.sin(target - current), np.cos(target - current))

        # Inner Loop: Joint-space PID control
        pi_outputs = []
        for pid, target_angle, current_angle in zip(
                self.joint_pids, target_joint_angles, current_joint_angles
        ):
            error = angle_error(target_angle, current_angle)
            output = pid.compute(0.0, -error)  # setpoint=0, measurement=-error
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

        # Gated debug telemetry capture (minimal overhead when disabled)
        if self._debug_telemetry_enabled:
            self._last_pi_outputs = list(pi_outputs)
            self._last_named_commands = dict(named_commands)

        # DEBUG: Print commands periodically (every ~50 loops = ~1 sec at 50Hz)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._verbose and self._debug_counter % 50 == 0:
            max_cmd = max(abs(x) for x in pi_outputs)
            if max_cmd > 0.01:  # Only print if non-trivial commands
                print(f"[CTRL] Commands: slew={pi_outputs[0]:+.3f} boom={pi_outputs[1]:+.3f} arm={pi_outputs[2]:+.3f} bucket={pi_outputs[3]:+.3f}")

        # Track IK/FK completion time before PWM
        if self._enable_perf_tracking:
            t_ik_end = time.perf_counter()
            t_pwm_start = t_ik_end

        self.hardware.send_named_pwm_commands(named_commands)
        # Mark that we are actively commanding
        self._outputs_zeroed = False

        # Track PWM completion time
        if self._enable_perf_tracking:
            t_pwm_end = time.perf_counter()
            # Store stage timings in instance variables for collection in main loop
            self._last_ik_time = t_ik_end - t_ik_start
            self._last_pwm_time = t_pwm_end - t_pwm_start

    def __del__(self):
        # Be defensive: __init__ can fail before thread fields exist
        try:
            if getattr(self, "_control_thread", None) is not None:
                self.stop()
        except Exception:
            pass
