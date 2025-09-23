#!/usr/bin/env python3
"""
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
    kp1: float = 1.6  # Boom
    ki1: float = 0.4
    kd1: float = 0.00

    kp2: float = 1.6  # Arm
    ki2: float = 0.4
    kd2: float = 0.00

    kp3: float = 1.6  # Bucket
    ki3: float = 0.4
    kd3: float = 0.00

    output_limits: Tuple[float, float] = (-1.0, 1.0)
    control_frequency: float = 60.0  # Hz


class ExcavatorController:
    def __init__(self, hardware_interface, config: Optional[ControllerConfig] = None):
        self.hardware = hardware_interface
        self.config = config or ControllerConfig()

        # Robot configuration
        self.robot_config = diff_ik.create_excavator_config()
        diff_ik.warmup_numba_functions()

        # IK controller setup
        self.ik_config = diff_ik.IKControllerConfig(
            command_type="pose",
            ik_method="svd",
            use_relative_mode=False,
            ik_params={
                "k_val": 1.75,
                "min_singular_value": 1e-6,
                "lambda_val": 0.1,
                "position_weight": 1.0,
                "rotation_weight": 1.0
            }
        )
        self.ik_controller = diff_ik.IKController(self.ik_config, self.robot_config)

        # PID controllers for joints
        joint_configs = [
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
        self._current_joint_angles = np.zeros(3, dtype=np.float32)

        print("ExcavatorController initialized")

    def start(self) -> None:
        if self._control_thread is not None:
            print("Controller already running!")
            return

        self._stop_event.clear()
        self._pause_event.clear()  # Start unpaused
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("Background control loop started")

    def stop(self) -> None:
        if self._control_thread is None:
            return

        self._stop_event.set()
        self._control_thread.join(timeout=2.0)
        self._control_thread = None
        self.hardware.reset(reset_pump=True)
        print("Background control loop stopped")

    def pause(self) -> None:
        """Pause the control loop (IK, PID, PWM updates) without stopping hardware."""
        if self._control_thread is None:
            print("Controller not running - cannot pause")
            return
        
        self._pause_event.set()
        print("Controller paused (hardware stays active)")

    def resume(self) -> None:
        """Resume the control loop from paused state."""
        if self._control_thread is None:
            print("Controller not running - cannot resume")
            return
        
        # Clear stale PID states to prevent old integral/derivative terms
        for pid in self.joint_pids:
            pid.reset()
        
        # IMPORTANT: Update current state before resuming
        # This prevents IK confusion from stale position data
        self._update_current_state()
        
        self._pause_event.clear()
        print("Controller resumed (PIDs reset, position updated)")

    def give_pose(self, position, rotation_y_deg: float = 0.0) -> None:
        with self._lock:
            self._target_position = np.array(position, dtype=np.float32)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            y_rotation_rad = np.radians(rotation_y_deg)
            self._target_orientation = quat_from_axis_angle(y_axis, y_rotation_rad)

    def get_pose(self) -> Tuple[np.ndarray, float]:
        with self._lock:
            return self._current_position.copy(), self._current_orientation_y_deg

    def get_joint_angles(self) -> np.ndarray:
        with self._lock:
            return np.degrees(self._current_joint_angles.copy())

    def _control_loop(self) -> None:
        loop_period = 1.0 / self.config.control_frequency
        print(f"Control loop running at {self.config.control_frequency}Hz")

        while not self._stop_event.is_set():
            loop_start_time = time.time()
            
            # Check if paused - if so, just sleep and continue
            if self._pause_event.is_set():
                time.sleep(0.1)  # Sleep while paused
                continue
                
            try:
                self._update_current_state()
                with self._lock:
                    has_target = self._target_position is not None
                if has_target:
                    self._compute_control_commands()
                else:
                    self.hardware.reset(reset_pump=False)
            except Exception as e:
                print(f"Control loop error: {e}")
                self.hardware.emergency_stop()
                break

            elapsed = time.time() - loop_start_time
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_current_state(self) -> None:
        try:
            quaternions = self.hardware.read_imu_data()
            if quaternions is None or len(quaternions) != 3:
                return
            corrected_quats = diff_ik.apply_imu_offsets(quaternions, self.robot_config)
            projected_quats = diff_ik.project_to_rotation_axes(corrected_quats, self.robot_config)
            # Get true end-effector position with ee_offset applied
            ee_pos = diff_ik.get_end_effector_position(projected_quats, self.robot_config)
            joint_angles = np.array([
                diff_ik.extract_axis_rotation(q, axis)
                for q, axis in zip(projected_quats, self.robot_config.rotation_axes)
            ])
            ee_quat = diff_ik.get_end_effector_orientation(projected_quats, self.robot_config)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ee_y_angle_rad = diff_ik.extract_axis_rotation(ee_quat, y_axis)
            ee_y_angle_deg = np.degrees(ee_y_angle_rad)
            with self._lock:
                self._current_position = ee_pos
                self._current_orientation_y_deg = ee_y_angle_deg
                self._current_joint_angles = joint_angles
        except Exception as e:
            print(f"Error in state update: {e}")

    def _compute_control_commands(self) -> None:
        try:
            with self._lock:
                target_pos = self._target_position.copy()
                target_quat = self._target_orientation.copy()
                current_pos = self._current_position.copy()
                current_joint_angles = self._current_joint_angles.copy()

            # Get current end-effector orientation
            current_quaternions = self.hardware.read_imu_data()
            if current_quaternions is None or len(current_quaternions) != 3:
                return
            corrected_quats = diff_ik.apply_imu_offsets(current_quaternions, self.robot_config)
            projected_quats = diff_ik.project_to_rotation_axes(corrected_quats, self.robot_config)
            current_ee_quat = diff_ik.get_end_effector_orientation(projected_quats, self.robot_config)

            # Outer Loop: Task-space IK control
            pose_command = np.concatenate([target_pos, target_quat])
            self.ik_controller.set_command(pose_command)
            target_joint_angles = self.ik_controller.compute(
                current_pos,
                current_ee_quat,
                current_joint_angles,
                joint_quats=None
            )
        except Exception as e:
            print(f"Error in control computation: {e}")
            return

        if target_joint_angles is None:
            self.hardware.reset(reset_pump=False)
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

        pwm_commands = [
            pi_outputs[2],  # bucket/scoop
            pi_outputs[0],  # boom/lift
            0,
            0,
            pi_outputs[1],  # arm/tilt
            0,
            0,
            0
        ]

        self.hardware.send_pwm_commands(pwm_commands)

    def __del__(self):
        self.stop()
