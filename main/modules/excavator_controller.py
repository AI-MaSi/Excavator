#!/usr/bin/env python3
"""
Excavator Controller - Clean API with Background Control Loop

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
from . import diff_ik
from .pid import PIDController
from .quaternion_math import (
    quat_from_axis_angle,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
)

# Load IRL settings exclusively from configuration_files/pathing_config.py (fail hard if missing)
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
    """Configuration for the excavator controller."""
    kp0: float = 0.9 # slew
    ki0: float = 0.0
    kd0: float = 0.0

    kp1: float = 6.0 # lift
    ki1: float = 0.5
    kd1: float = 0.0

    kp2: float = 5.0 # tilt
    ki2: float = 0.25
    kd2: float = 0.0

    # pretty bad still
    kp3: float = 8.0 # scoop
    ki3: float = 1.0
    kd3: float = 0.0

    output_limits: Tuple[float, float] = (-1.0, 1.0)
    control_frequency: float = 100.0  # Hz
    enable_velocity_limiting: bool = True
    max_joint_velocity: float = 0.00090 #0.00070  # rad/iter when velocity limiting enabled


class ExcavatorController:
    def __init__(self, hardware_interface, config: Optional[ControllerConfig] = None,
                 enable_perf_tracking: bool = False):
        self.hardware = hardware_interface
        self.config = config or ControllerConfig()
        self._enable_perf_tracking = enable_perf_tracking
        # Cascade perf flag to hardware if supported (no-op otherwise)
        try:
            if hasattr(self.hardware, 'set_perf_enabled'):
                self.hardware.set_perf_enabled(bool(enable_perf_tracking))
        except Exception:
            pass

        # Adopt control loop rate from general config when no explicit config is provided
        try:
            if config is None and hasattr(self.hardware, 'get_status'):
                # Try to fetch general config if hardware exposes it
                gc = getattr(self.hardware, '_general_config', None)
                if isinstance(gc, dict):
                    control_hz = gc.get('rates', {}).get('control_hz')
                    if isinstance(control_hz, (int, float)) and control_hz > 0:
                        self.config.control_frequency = float(control_hz)
        except Exception:
            pass

        # Robot configuration
        self.robot_config = diff_ik.create_excavator_config()
        diff_ik.warmup_numba_functions()

        # IK controller setup (pull settings from shared config when available)
        if _DEFAULT_CONFIG is not None:
            _ik_cmd_type = getattr(_DEFAULT_CONFIG, 'ik_command_type', 'pose')
            _ik_method = getattr(_DEFAULT_CONFIG, 'ik_method', 'svd')
            _ik_rel = bool(getattr(_DEFAULT_CONFIG, 'ik_use_relative_mode', False))
            _ik_params = getattr(_DEFAULT_CONFIG, 'ik_params', None)
        else:
            raise RuntimeError("ExcavatorController requires configuration_files/pathing_config.py")

        self.ik_config = diff_ik.IKControllerConfig(
            command_type=_ik_cmd_type,
            ik_method=_ik_method,
            use_relative_mode=_ik_rel,
            ik_params=_ik_params,
        )
        self.ik_controller = diff_ik.IKController(self.ik_config, self.robot_config)  # Advanced with weighting

        # Relative IK control parameters (pulling toward target)
        if _DEFAULT_CONFIG is not None:
            self._relative_pos_gain = float(getattr(_DEFAULT_CONFIG, 'relative_pos_gain', 0.2))
            self._relative_rot_gain = float(getattr(_DEFAULT_CONFIG, 'relative_rot_gain', 0.4))
        else:
            self._relative_pos_gain = 0.2   # fraction of position error per control step
            self._relative_rot_gain = 0.4   # fraction of orientation error (axis-angle) per step

        # Orientation locks (hardware capability mapping) from shared config if available
        if _DEFAULT_CONFIG is not None:
            self._lock_roll = bool(getattr(_DEFAULT_CONFIG, 'lock_roll', True))
            self._lock_pitch = bool(getattr(_DEFAULT_CONFIG, 'lock_pitch', False))
            self._lock_yaw = bool(getattr(_DEFAULT_CONFIG, 'lock_yaw', True))
        else:
            # Safe defaults for excavator: lock roll & yaw, allow pitch
            self._lock_roll = True
            self._lock_pitch = False
            self._lock_yaw = True

        # Optionally adopt control loop rate from shared config when not explicitly provided
        try:
            if config is None and _DEFAULT_CONFIG is not None:
                hz = float(getattr(_DEFAULT_CONFIG, 'update_frequency', self.config.control_frequency))
                if hz > 0:
                    self.config.control_frequency = hz
        except Exception:
            pass

        # PID controllers for joints
        joint_configs = [
            {"name": "Slew", "kp": self.config.kp0, "ki": self.config.ki0, "kd": self.config.kd0},
            {"name": "Boom", "kp": self.config.kp1, "ki": self.config.ki1, "kd": self.config.kd1},
            {"name": "Arm", "kp": self.config.kp2, "ki": self.config.ki2, "kd": self.config.kd2},
            {"name": "Bucket", "kp": self.config.kp3, "ki": self.config.ki3, "kd": self.config.kd3},
        ]

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
        self._prev_target_angles = None  # For velocity limiting
        self._outputs_zeroed = False  # Track whether we've already sent a zero/neutral command

        # Performance tracking
        self._loop_times = []
        self._compute_times = []  # Actual computation time (without sleep)
        self._timing_violations = 0
        self._loop_count = 0
        self._perf_lock = threading.Lock()

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
        self.hardware.reset(reset_pump=True)

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

        # Reset velocity limiter to prevent jumps from stale data
        self._prev_target_angles = None

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

    def _apply_orientation_locks(self, current_ee_quat: np.ndarray, desired_quat: np.ndarray) -> np.ndarray:
        """Copy locked axes (roll=x, pitch=y, yaw=z) from current EE to desired.

        Uses Euler XYZ convention in base frame. Slew yaw propagation is preserved
        because current_ee_quat includes the high-accuracy slew yaw.
        """
        if not (self._lock_roll or self._lock_pitch or self._lock_yaw):
            return desired_quat

        # Convert to Euler
        r_cur, p_cur, y_cur = euler_xyz_from_quat(current_ee_quat)
        r_des, p_des, y_des = euler_xyz_from_quat(desired_quat)

        r = r_des
        p = p_des
        y = y_des
        if self._lock_roll:
            r = r_cur
        if self._lock_pitch:
            p = p_cur
        if self._lock_yaw:
            y = y_cur

        return quat_from_euler_xyz(r, p, y)

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

        # Reset PIDs and velocity limiter to avoid residual terms
        for pid in self.joint_pids:
            pid.reset()
        self._prev_target_angles = None

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
                    'sample_count': 0
                }

            loop_times_ms = np.array(self._loop_times) * 1000.0
            compute_times_ms = np.array(self._compute_times) * 1000.0
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
                'sample_count': self._loop_count
            }

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

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        with self._perf_lock:
            self._loop_times = []
            self._compute_times = []
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

            try:
                self._update_current_state()
                with self._lock:
                    has_target = self._target_position is not None
                if has_target:
                    self._compute_control_commands()
                else:
                    # Only send a neutral command once when there is no target
                    if not self._outputs_zeroed:
                        self.hardware.reset(reset_pump=False)
                        self._outputs_zeroed = True
            except Exception as e:
                print(f"Control loop error: {e}")
                self.hardware.reset(reset_pump=True)
                break

            # === Track performance metrics ===
            if self._enable_perf_tracking:
                compute_end_time = time.perf_counter()
                actual_compute_time = compute_end_time - compute_start_time
                actual_loop_period = loop_start_time - last_loop_start

                if actual_loop_period > 0.001:  # Ignore first loop
                    with self._perf_lock:
                        self._loop_times.append(actual_loop_period)
                        self._compute_times.append(actual_compute_time)
                        self._loop_count += 1

                        # Check for timing violations
                        if actual_compute_time > loop_period:
                            self._timing_violations += 1

                        # Keep only last 1000 samples
                        if len(self._loop_times) > 1000:
                            self._loop_times.pop(0)
                            self._compute_times.pop(0)

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

        except Exception as e:
            print(f"Error in state update: {e}")

    def _compute_control_commands(self) -> None:
        """Compute and send control commands based on current state and target."""
        try:
            # Get cached state from _update_current_state()
            with self._lock:
                if self._current_projected_quats is None:
                    return

                target_pos = self._target_position.copy()
                target_quat = self._target_orientation.copy()
                current_pos = self._current_position.copy()
                projected_quats = self._current_projected_quats  # Use cached quaternions

            # Extract current joint angles and end-effector orientation
            current_joint_angles = np.array([
                diff_ik.extract_axis_rotation(q, axis)
                for q, axis in zip(projected_quats, self.robot_config.rotation_axes)
            ])

            # End-effector orientation is the last joint's orientation (already propagated)
            current_ee_quat_full = projected_quats[-1].copy()

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
                    self.ik_controller.set_command(delta_pos, ee_pos=current_pos, ee_quat=current_ee_quat)
                else:
                    # Pose: 6D delta [dx, dy, dz, rx, ry, rz]
                    # Apply axis locks to target orientation (preserve slew yaw, lock roll by default)
                    locked_target_quat = self._apply_orientation_locks(current_ee_quat, target_quat)
                    pos_err, axis_angle_err = diff_ik.compute_pose_error(
                        current_pos, current_ee_quat, target_pos, locked_target_quat
                    )
                    delta_pos = self._relative_pos_gain * pos_err
                    delta_rot = self._relative_rot_gain * axis_angle_err
                    delta_pose = np.concatenate([delta_pos, delta_rot])
                    self.ik_controller.set_command(delta_pose, ee_pos=current_pos, ee_quat=current_ee_quat)
            else:
                if self.ik_config.command_type == "position":
                    # Position-only mode: command is just [x, y, z]
                    position_command = target_pos
                    self.ik_controller.set_command(position_command, ee_quat=current_ee_quat)
                else:
                    # Pose mode: command is [x, y, z, qw, qx, qy, qz]
                    locked_target_quat = self._apply_orientation_locks(current_ee_quat, target_quat)
                    pose_command = np.concatenate([target_pos, locked_target_quat])
                    self.ik_controller.set_command(pose_command)

            target_joint_angles = self.ik_controller.compute(
                current_pos,
                current_ee_quat,  # Full orientation incl. slew yaw
                current_joint_angles,
                joint_quats=projected_quats  # Pass actual quaternions for accurate Jacobian
            )

            # Velocity limiting - prevent huge joint jumps when extended
            if target_joint_angles is not None and self.config.enable_velocity_limiting:
                if self._prev_target_angles is None:
                    self._prev_target_angles = current_joint_angles.copy()

                max_angle_change = self.config.max_joint_velocity
                delta = target_joint_angles - self._prev_target_angles
                delta = np.clip(delta, -max_angle_change, max_angle_change)
                target_joint_angles = self._prev_target_angles + delta
                self._prev_target_angles = target_joint_angles.copy()

        except Exception as e:
            print(f"Error in control computation: {e}")
            return

        if target_joint_angles is None:
            # If IK failed, command neutral once (avoid constant zeroing)
            if not self._outputs_zeroed:
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

        # DEBUG center rot
        #print(f"Slew command: {pi_outputs[0]:.3f}")
        self.hardware.send_named_pwm_commands(named_commands)
        # Mark that we are actively commanding
        self._outputs_zeroed = False

    def __del__(self):
        self.stop()
