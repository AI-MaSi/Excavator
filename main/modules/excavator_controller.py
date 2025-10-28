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

# Import project modules
from . import diff_ik
from .pid import PIDController
from .quaternion_math import quat_from_axis_angle


@dataclass
class ControllerConfig:
    """Configuration for the excavator controller."""
    kp0: float = 70.0   # Slew # 55.0
    ki0: float = 1.0
    kd0: float = 0.00

    # 22.10.2025 3.0 / 0.5

    kp1: float = 20.0#12.0 # lift
    ki1: float = 5.0#1.0
    kd1: float = 0.0#1.0

    kp2: float = 10.0#10.0  # tilt
    ki2: float = 1.0#1.0
    kd2: float = 1.0#3.0

    kp3: float = 7.0#10.0  # scoop
    ki3: float = 1.0#0.5
    kd3: float = 0.200#3.0

    output_limits: Tuple[float, float] = (-1.0, 1.0)
    control_frequency: float = 100.0  # Hz
    enable_velocity_limiting: bool = False
    max_joint_velocity: float = 0.005  # rad/iter when velocity limiting enabled (currently disabled)


class ExcavatorController:
    def __init__(self, hardware_interface, config: Optional[ControllerConfig] = None,
                 enable_perf_tracking: bool = False):
        self.hardware = hardware_interface
        self.config = config or ControllerConfig()
        self._enable_perf_tracking = enable_perf_tracking

        # Robot configuration
        self.robot_config = diff_ik.create_excavator_config()
        diff_ik.warmup_numba_functions()

        # IK controller setup
        self.ik_config = diff_ik.IKControllerConfig(
            command_type="pose",  # Full 6DOF control (position + orientation)
            #command_type="position",  # Position-only control (no orientation)
            ik_method="svd",
            use_relative_mode=False,
            ik_params={
                "k_val": 1.0,
                "min_singular_value": 1e-6,
                "lambda_val": 0.1,
                "position_weight": 1.0,
                "rotation_weight": 0.6,
                # Direction-based joint prioritization
                # Note: not used with SimpleIKController!
                "joint_weights": None,
                "direction_strengths_X": [1.0, 1.0, 1.0, 1.0],  # [slew, boom, arm, bucket] for X (forward)
                "direction_strengths_Y": [1.0, 1.0, 1.0, 1.0],  # [slew, boom, arm, bucket] for Y (lateral)
                "direction_strengths_Z": [1.0, 1.0, 1.0, 1.0],  # [slew, boom, arm, bucket] for Z (lift)
                "direction_scale_range": [1.0, 1.0]  # [min, max] Jacobian multipliers (set to [1.0, 1.0] to disable)
            }
        )
        # self.ik_controller = diff_ik.IKController(self.ik_config, self.robot_config)  # Advanced with weighting
        self.ik_controller = diff_ik.SimpleIKController(self.ik_config, self.robot_config)  # Simple baseline

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

            return {
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

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        with self._perf_lock:
            self._loop_times = []
            self._compute_times = []
            self._timing_violations = 0
            self._loop_count = 0

    def _get_processed_quaternions(self) -> Optional[np.ndarray]:
        """
        Read and process all sensor data into projected quaternions.

        Returns:
            Processed quaternions [slew, boom, arm, bucket] or None if hardware not ready
        """
        try:
            # Read IMU data (3 IMUs: boom, arm, bucket)
            quaternions = self.hardware.read_imu_data()
            if quaternions is None or len(quaternions) != 3:
                return None

            # Read slew quaternion directly from encoder (already computed in hardware)
            slew_quat = self.hardware.read_slew_quaternion()

            # Combine: [slew] + [boom, arm, bucket]
            all_quaternions = [slew_quat] + quaternions

            # Apply IMU mounting offset corrections
            corrected_quats = diff_ik.apply_imu_offsets(all_quaternions, self.robot_config)

            # Project to rotation axes (remove unwanted rotations like yaw)
            projected_quats = diff_ik.project_to_rotation_axes(corrected_quats, self.robot_config)

            return projected_quats

        except Exception as e:
            print(f"Error processing quaternions: {e}")
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
            # Get processed quaternions from sensors
            projected_quats = self._get_processed_quaternions()
            if projected_quats is None:
                return

            # Compute forward kinematics
            ee_pos = diff_ik.get_end_effector_position(projected_quats, self.robot_config)
            ee_quat = diff_ik.get_end_effector_orientation(projected_quats, self.robot_config)

            # Extract Y-axis rotation for end-effector orientation
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ee_y_angle_rad = diff_ik.extract_axis_rotation(ee_quat, y_axis)
            ee_y_angle_deg = np.degrees(ee_y_angle_rad)

            # Update shared state
            with self._lock:
                self._current_position = ee_pos
                self._current_orientation_y_deg = ee_y_angle_deg
                self._current_projected_quats = projected_quats

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
            current_ee_quat_full = diff_ik.get_end_effector_orientation(projected_quats, self.robot_config)

            # Use full orientation (including Z-rotation from slew)
            # The Jacobian's structure naturally constrains which rotations each joint can achieve
            # based on rotation_axes config (slew=Z, boom/arm/bucket=Y)
            current_ee_quat = current_ee_quat_full

            # Outer Loop: Task-space IK control
            if self.ik_config.command_type == "position":
                # Position-only mode: command is just [x, y, z]
                position_command = target_pos
                self.ik_controller.set_command(position_command, ee_quat=current_ee_quat)
            else:
                # Pose mode: command is [x, y, z, qw, qx, qy, qz]
                pose_command = np.concatenate([target_pos, target_quat])
                self.ik_controller.set_command(pose_command)

            target_joint_angles = self.ik_controller.compute(
                current_pos,
                current_ee_quat,  # Y-only orientation (no Z-rotation error)
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
