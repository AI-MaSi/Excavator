"""
Differential IK controller with numpy+numba optimization.

Contains:
- Axis-rotation helpers (generic quaternion utilities)
- Numba-optimized FK, Jacobian, and IK method functions
- Base rotation propagation and Jacobian wrapper
- IKController class (the main solver)

IMPORTANT: Origin and end-effector offset handling:
- forward_kinematics_core(): Returns joint positions (includes origin_offset, no ee_offset)
- _compute_ee_position_core(): Shared helper that applies ee_offset in last joint's local frame
- forward_kinematics_with_ee_offset_core(): Returns joint positions + ee position with both offsets
- origin_offset: Applied to all positions - shifts entire robot from world origin
- ee_offset: Applied in last joint's local coordinate frame - extends end-effector

Notes:
- All quaternions are float32 and use [w, x, y, z] convention
- We assume corrected joint-frame quaternions on input; no mounting-offset correction is applied here.
- We assume FULL quaternions on input; no axis projection is applied.
  Orientation components are filtered in IK via the ignore_axes system.
- The active controller uses canonical joint angles as the source of truth;
  quaternion inputs are already absolute link orientations unless explicitly
  converted from joint angles in this module.
"""

import numpy as np
import numba
import logging
from typing import Optional, List

from .differential_ik_cfg import IKControllerConfig, RobotConfig
from .quaternion_math import (
    quat_normalize, quat_multiply, quat_conjugate, quat_rotate_vector,
    quat_from_axis_angle, axis_angle_from_quat, compute_pose_error,
    apply_delta_pose
)


# ----------------------------
# Axis-rotation helpers (FULL input -> axis twist)
# ----------------------------

def extract_axis_rotation(quat: np.ndarray, axis: np.ndarray) -> float:
    """
    Extract twist angle about `axis` from quaternion `quat` using swing-twist.

    Args:
        quat: [w, x, y, z] unit quaternion (will be normalized)
        axis: [3] unit axis (will be normalized)

    Returns:
        angle (radians) in (-pi, pi]
    """
    q = quat_normalize(np.asarray(quat, dtype=np.float32))
    a = np.asarray(axis, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    v = np.array([x, y, z], dtype=np.float32)
    s = float(np.dot(v, a))  # signed component along axis
    ang = 2.0 * np.arctan2(s, w)
    # Wrap angle to (-pi, pi] without relying on external helpers.
    pi = np.float32(3.141592653589793)
    two_pi = np.float32(6.283185307179586)
    ang = (np.float32(ang) + pi) % two_pi - pi
    return float(ang)


def project_to_rotation_axes(quats: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """
    Project each quaternion to a pure rotation about its corresponding axis.
    Input quats are assumed FULL; output quats have only axis twist preserved.
    """
    quats = np.asarray(quats, dtype=np.float32)
    axes = np.asarray(axes, dtype=np.float32)
    out = np.zeros_like(quats, dtype=np.float32)
    for i in range(len(quats)):
        axis = axes[i]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        theta = extract_axis_rotation(quats[i], axis)
        out[i] = quat_from_axis_angle(axis, np.float32(theta))
    return out


# ----------------------------
# Numba-optimized kinematics functions
# ----------------------------

@numba.njit(fastmath=True, nogil=True)
def forward_kinematics_core(quats, link_lengths, link_directions, origin_offset):
    """
    Core forward kinematics - joint positions only (no end-effector offset).

    This is the building block function used by:
    - Jacobian computation (needs joint positions without ee_offset)
    - get_joint_positions() wrapper for visualization
    - forward_kinematics_with_ee_offset_core() internally

    Args:
        quats: Normalized absolute quaternions from IMUs
        link_lengths: Length of each link
        link_directions: Direction vector for each link in local frame
        origin_offset: Position offset of first joint from world origin

    Returns:
        Joint positions with origin_offset applied (no ee_offset)
    """
    n = len(link_lengths)
    joint_positions = np.zeros((n, 3), dtype=np.float32)

    # Start at origin offset (first joint position)
    pos = origin_offset.copy()

    for i in range(n):
        # Link vector in local frame -> rotate into world using absolute orientation
        link_vec = link_lengths[i] * link_directions[i]
        world_link_vec = quat_rotate_vector(quats[i], link_vec)

        # Add to current position
        pos = pos + world_link_vec
        joint_positions[i] = pos

    return joint_positions


@numba.njit(fastmath=True, nogil=True)
def _compute_ee_position_core(joint_positions, quats, origin_offset, ee_offset):
    """Apply ee_offset in the last joint's local frame to get end-effector world position."""
    if len(joint_positions) > 0:
        world_ee_offset = quat_rotate_vector(quats[-1], ee_offset)
        return joint_positions[-1] + world_ee_offset
    return np.asarray(origin_offset, dtype=np.float32).copy()


@numba.njit(fastmath=True, nogil=True)
def forward_kinematics_with_ee_offset_core(quats, link_lengths, link_directions, origin_offset, ee_offset):
    """
    Forward kinematics with full end-effector offset calculation.

    Used for getting the actual end-effector position including tool offset.
    This is what control systems need for accurate positioning.

    Args:
        quats: Normalized absolute quaternions from IMUs
        link_lengths: Length of each link
        link_directions: Direction vector for each link in local frame
        origin_offset: Position offset of first joint from world origin
        ee_offset: Tool offset from last joint in local frame

    Returns:
        Tuple of (joint_positions, ee_position) both with offsets applied
    """
    joint_positions = forward_kinematics_core(quats, link_lengths, link_directions, origin_offset)
    ee_position = _compute_ee_position_core(joint_positions, quats, origin_offset, ee_offset)
    return joint_positions, ee_position


@numba.njit(fastmath=True, nogil=True)
def compute_jacobian_core(quats, link_lengths, link_directions, rotation_axes, origin_offset, ee_offset):
    n = len(link_lengths)
    jacobian = np.zeros((6, n), dtype=np.float32)

    joint_positions = forward_kinematics_core(quats, link_lengths, link_directions, origin_offset)
    ee_pos = _compute_ee_position_core(joint_positions, quats, origin_offset, ee_offset)

    # Base position is now the origin offset.  `quats` are absolute link
    # orientations, so joint axes are rotated by the parent link orientation,
    # not by cumulatively multiplying absolute quaternions again.
    base_pos = np.asarray(origin_offset, dtype=np.float32).copy()

    for i in range(n):
        local_axis = np.asarray(rotation_axes[i], dtype=np.float32)
        if i == 0:
            world_axis = local_axis
        else:
            world_axis = quat_rotate_vector(quats[i - 1], local_axis)

        if i == 0:
            joint_pos = base_pos
        else:
            joint_pos = joint_positions[i-1]

        # Linear velocity component
        jacobian[0:3, i] = np.cross(world_axis, ee_pos - joint_pos)
        # Angular velocity component
        jacobian[3:6, i] = world_axis

    return jacobian


def joint_angles_to_absolute_quaternions(joint_angles: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Compose absolute link quaternions from canonical relative joint angles."""
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    axes = robot_config.rotation_axes
    absolute = np.zeros((len(joint_angles), 4), dtype=np.float32)

    cumulative = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i, angle in enumerate(joint_angles):
        local = quat_from_axis_angle(axes[i], np.float32(angle))
        cumulative = quat_normalize(quat_multiply(cumulative, local))
        absolute[i] = cumulative

    return absolute


def get_all_poses_from_joint_angles(joint_angles: np.ndarray, robot_config: RobotConfig):
    """Return joint poses and EE pose from canonical joint angles."""
    quats = joint_angles_to_absolute_quaternions(joint_angles, robot_config)
    joint_positions, ee_position = forward_kinematics_with_ee_offset_core(
        quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.origin_offset,
        robot_config.ee_offset,
    )
    return joint_positions, quats, ee_position, quats[-1].copy()


def get_pose_from_joint_angles(joint_angles: np.ndarray, robot_config: RobotConfig):
    """Return end-effector pose from canonical joint angles."""
    _, _, ee_position, ee_orientation = get_all_poses_from_joint_angles(joint_angles, robot_config)
    return ee_position, ee_orientation


def compute_jacobian_from_joint_angles(joint_angles: np.ndarray, robot_config: RobotConfig):
    """Compute the geometric Jacobian from canonical joint angles."""
    quats = joint_angles_to_absolute_quaternions(joint_angles, robot_config)
    return compute_jacobian_core(
        quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.rotation_axes,
        robot_config.origin_offset,
        robot_config.ee_offset,
    )


# ----------------------------
# Numba-optimized IK methods - ALL FLOAT32
# ----------------------------

@numba.njit(fastmath=True, nogil=True)
def compute_jacobian_metrics(jacobian):
    """Return (condition_number, singular_values_f32, yoshikawa_index) from one SVD.

    Single-pass replacement for compute_condition_number — all three values are
    byproducts of the same factorisation, so computing them together avoids a
    redundant SVD and gives a clean extension point for future Jacobian telemetry.
    """
    _, S, _ = np.linalg.svd(jacobian.astype(np.float64), full_matrices=False)
    cond = S[0] / (S[-1] + 1e-12)
    yoshikawa = np.float32(1.0)
    for s in S:
        yoshikawa *= np.float32(s)
    return float(cond), S.astype(np.float32), float(yoshikawa)


@numba.njit(fastmath=True, nogil=True)
def ik_method_pinv(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Pseudo-inverse IK method."""
    jacobian_pinv = np.linalg.pinv(np.asarray(jacobian, dtype=np.float32))
    return k_val * np.dot(jacobian_pinv, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=True, nogil=True)
def ik_method_svd(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32, min_singular_value: np.float32) -> np.ndarray:
    """SVD-based IK method with singular value thresholding."""
    jac_f32 = np.asarray(jacobian, dtype=np.float32)
    delta_f32 = np.asarray(delta_pose, dtype=np.float32)

    U, S, Vh = np.linalg.svd(jac_f32, full_matrices=False)

    # Compute U^T * delta_pose
    ut_delta = np.dot(U.T, delta_f32)

    # Apply pseudoinverse of singular values with thresholding
    result = np.zeros(jac_f32.shape[1], dtype=np.float32)
    for i in range(len(S)):
        if S[i] > min_singular_value:
            result += Vh[i] * (ut_delta[i] / S[i])

    return k_val * result


@numba.njit(fastmath=True, nogil=True)
def ik_method_transpose(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Jacobian transpose IK method."""
    return k_val * np.dot(np.asarray(jacobian, dtype=np.float32).T, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=True, nogil=True)
def ik_method_damped_least_squares(jacobian: np.ndarray, delta_pose: np.ndarray, lambda_val: np.float32) -> np.ndarray:
    """Damped least squares IK method."""
    jac_f32 = np.asarray(jacobian, dtype=np.float32)
    delta_f32 = np.asarray(delta_pose, dtype=np.float32)

    # Solve (J * J^T + lambda^2 I) directly instead of forming an explicit inverse.
    jjt = np.dot(jac_f32, jac_f32.T)
    lambda_matrix = (lambda_val ** 2) * np.eye(jjt.shape[0], dtype=np.float32)
    solved = np.linalg.solve(jjt + lambda_matrix, delta_f32)

    # delta_q = J^T * (J*J^T + lambda^2 I)^-1 * dp
    return np.dot(jac_f32.T, solved)


# ----------------------------
# Base rotation propagation & Jacobian wrapper
# ----------------------------

def propagate_base_rotation(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Propagate base joint rotation to downstream joints in kinematic chain.

    For 6-axis IMUs without magnetometers, yaw (Z-rotation) is unmeasurable.
    When the base joint (slew) rotates, downstream links (boom/arm/bucket)
    physically rotate in the world XY plane, but their IMUs cannot detect this.

    This function projects downstream IMU quaternions to their rotation axes only
    (removing roll/yaw drift), then composes with the slew rotation to get
    world-frame orientations.

    Args:
        quats: Input quaternions [base, link1, link2, ...] where base rotation is
               provided as a clean Z-rotation and downstream orientations are from IMUs
        robot_config: Robot configuration containing rotation axes for each joint

    Returns:
        Quaternions with base rotation propagated to all downstream joints
    """
    quats = np.asarray(quats, dtype=np.float32)
    propagated_quats = quats.copy()

    # Base stays as-is (already a clean Z-rotation)
    base_quat = quats[0]

    # Project downstream joints to their rotation axes (removes roll/yaw drift)
    if len(quats) > 1:
        downstream_quats = quats[1:]
        downstream_axes = np.array(robot_config.rotation_axes[1:], dtype=np.float32)

        # Extract ONLY axis twist component (e.g., Y-axis pitch for boom/arm/bucket)
        projected_quats = project_to_rotation_axes(downstream_quats, downstream_axes)

        # Compose with base rotation: q_world = q_base * q_local_projected
        for i, proj_quat in enumerate(projected_quats):
            propagated_quats[i+1] = quat_normalize(quat_multiply(base_quat, proj_quat))

    return propagated_quats


def compute_jacobian(quats: np.ndarray, robot_config: RobotConfig):
    """Jacobian computation wrapper using absolute link quaternions."""
    quats = np.asarray(quats, dtype=np.float32)
    return compute_jacobian_core(
        quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.rotation_axes,
        robot_config.origin_offset,
        robot_config.ee_offset
    )


# ----------------------------
# IKController
# ----------------------------

class IKController:
    """Differential inverse kinematics controller with numpy+numba optimization."""

    def __init__(
        self,
        cfg: IKControllerConfig,
        robot_config: RobotConfig,
        verbose: bool = True,
        log_level: str = "INFO",
        default_dt: float = 0.01,
    ):
        """Initialize IK controller.

        Args:
            cfg: IK controller configuration
            robot_config: Robot kinematic configuration
            verbose: Legacy parameter, maps to DEBUG level if True
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            default_dt: Fallback timestep (s) when none is provided to compute()
        """
        self.cfg = cfg
        self.robot_config = robot_config
        self.default_dt = float(max(1e-4, default_dt))

        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.IKController")
        if not verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize debug counters
        self._ik_raw_debug_counter = 0

        # Target pose buffers
        self.ee_pos_des = np.zeros(3, dtype=np.float32)
        self.ee_quat_des = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._command = np.zeros(self.action_dim, dtype=np.float32)

        # Velocity mode settings
        self.velocity_mode = bool(getattr(cfg, "velocity_mode", False))
        self.velocity_error_gain = float(getattr(cfg, "velocity_error_gain", 1.0))
        self.use_rotational_velocity = bool(cfg.use_rotational_velocity)

        # Velocity limits (default to ~2 degrees per cycle if not specified)
        if cfg.max_joint_velocities is None:
            self.max_joint_velocities = np.full(robot_config.num_joints, 0.035, dtype=np.float32)  # ~2 deg
        else:
            self.max_joint_velocities = np.asarray(cfg.max_joint_velocities, dtype=np.float32)

        # Joint limits (default to [-pi, pi] if not specified)
        if cfg.joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi) for _ in range(robot_config.num_joints)]
        else:
            self.joint_limits = cfg.joint_limits

        # Debug/telemetry values (for external logging when enabled)
        self.last_adaptive_lambda: float = 0.0
        self.last_condition_number: float = 0.0
        self.last_yoshikawa_index: float = 0.0
        self.last_singular_values: np.ndarray = np.zeros(4, dtype=np.float32)
        # Always compute controllable DOFs (auto-detect + ignore_axes)
        self.controllable_dofs = self._compute_controllable_dofs()
        self.n_controllable = len(self.controllable_dofs)

    def _compute_controllable_dofs(self) -> List[int]:
        """Determine which DOFs (rows of Jacobian) are controllable based on joint axes.

        For excavator with slew (Z) + boom/arm/bucket (Y):
        - Position: [0, 1, 2] - always controllable (3D translation)
        - Roll (3): Only if any joint has X-axis rotation
        - Pitch (4): Only if any joint has Y-axis rotation
        - Yaw (5): Only if any joint has Z-axis rotation

        Returns:
            List of controllable DOF indices (0-5 mapping to [x, y, z, roll, pitch, yaw])
        """
        # Position is always controllable
        controllable = [0, 1, 2]

        # Check which rotation axes are present
        has_x_rotation = False
        has_y_rotation = False
        has_z_rotation = False

        for axis in self.robot_config.rotation_axes:
            axis_normalized = axis / (np.linalg.norm(axis) + 1e-12)

            if abs(axis_normalized[0]) > 0.866:  # cos(30 deg)
                has_x_rotation = True
            if abs(axis_normalized[1]) > 0.866:
                has_y_rotation = True
            if abs(axis_normalized[2]) > 0.866:
                has_z_rotation = True

        if has_x_rotation:
            controllable.append(3)  # Roll
        if has_y_rotation:
            controllable.append(4)  # Pitch
        if has_z_rotation:
            controllable.append(5)  # Yaw

        return controllable

    def _get_reduced_jacobian(self, jacobian_full: np.ndarray) -> np.ndarray:
        """Extract reduced Jacobian containing only controllable DOF rows.

        Args:
            jacobian_full: Full 6xn Jacobian matrix

        Returns:
            Reduced mxn Jacobian where m = number of controllable DOFs
            For excavator: 5x4 (position + pitch + yaw)
        """
        return jacobian_full[self.controllable_dofs, :]

    def _get_reduced_error(self, position_error: np.ndarray,
                          axis_angle_error: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract reduced error vector containing only controllable DOFs.

        Args:
            position_error: 3D position error [x, y, z]
            axis_angle_error: 3D rotation error [roll, pitch, yaw] (optional)

        Returns:
            Reduced error vector with only controllable DOFs
        """
        if axis_angle_error is None:
            return position_error

        # Build full 6D error
        full_error = np.concatenate([position_error, axis_angle_error])

        # Extract only controllable components
        return full_error[self.controllable_dofs]

    def _prepare_desired_velocity(
        self,
        desired_ee_velocity: Optional[np.ndarray],
        required_size: int
    ) -> np.ndarray:
        """Align desired EE velocity vector with the reduced task dimension."""
        if desired_ee_velocity is None:
            return np.zeros(required_size, dtype=np.float32)

        desired_vec = np.asarray(desired_ee_velocity, dtype=np.float32).flatten()

        # Expand position-only inputs when orientation components are expected
        if desired_vec.size == 3 and required_size > 3:
            full_vec = np.concatenate([desired_vec, np.zeros(3, dtype=np.float32)])
            if required_size != 6:
                desired_vec = full_vec[self.controllable_dofs][:required_size]
            else:
                desired_vec = full_vec[:required_size]

        # Map full 6D inputs into the reduced task space (drops uncontrollable axes)
        if required_size != 6 and desired_vec.size == 6:
            desired_vec = desired_vec[self.controllable_dofs][:required_size]

        # Position-only mode: ignore any extra components
        if required_size == 3 and desired_vec.size > 3:
            desired_vec = desired_vec[:3]

        if desired_vec.size != required_size:
            raise ValueError(
                f"desired_ee_velocity size {desired_vec.size} does not match required size {required_size}"
            )

        return desired_vec

    @property
    def action_dim(self) -> int:
        """Command dimension based on configuration."""
        if self.cfg.command_type == "position":
            return 3
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, rx, ry, rz)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    def reset(self):
        """Reset controller state."""
        self.ee_pos_des.fill(0.0)
        self.ee_quat_des = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._command.fill(0.0)

    def _transform_to_robot_local_frame(self, quat: np.ndarray, joint_quats: np.ndarray) -> np.ndarray:
        """
        Transform a quaternion from global frame to robot's local frame by removing slew rotation.

        This removes the base (slew) yaw component so that pitch/roll are expressed
        relative to the robot's current heading, not the global frame.

        Args:
            quat: Quaternion in global frame [w, x, y, z]
            joint_quats: Current joint quaternions (for extracting slew angle)

        Returns:
            Quaternion in robot's local frame [w, x, y, z]
        """
        # Extract slew angle from joint 0 (Z-axis rotation)
        slew_angle = extract_axis_rotation(joint_quats[0], self.robot_config.rotation_axes[0])

        # Create slew quaternion (yaw rotation around Z)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        slew_quat = quat_from_axis_angle(z_axis, np.float32(slew_angle))
        slew_quat_inv = quat_conjugate(slew_quat)

        # Remove slew from quaternion: q_local = q_slew^-1 * q_global
        quat_local = quat_normalize(quat_multiply(slew_quat_inv, quat))

        return quat_local

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def set_command(self, command: np.ndarray, ee_pos: np.ndarray = None, ee_quat: np.ndarray = None):
        """Set target end-effector command."""

        self._command = np.asarray(command, dtype=np.float32)

        if self.cfg.command_type == "position":
            if ee_quat is None:
                raise ValueError("ee_quat required for position command type")

            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("ee_pos required for relative mode")
                self.ee_pos_des = np.asarray(ee_pos, dtype=np.float32) + self._command
                self.ee_quat_des = np.asarray(ee_quat, dtype=np.float32)
            else:
                self.ee_pos_des = self._command.copy()
                self.ee_quat_des = np.asarray(ee_quat, dtype=np.float32)
        else:  # pose command
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("ee_pos and ee_quat required for relative pose mode")
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(
                    np.asarray(ee_pos, dtype=np.float32), np.asarray(ee_quat, dtype=np.float32), self._command
                )
            else:
                self.ee_pos_des = self._command[0:3].copy()
                self.ee_quat_des = self._command[3:7].copy()

    def compute(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        joint_angles: np.ndarray,
        joint_quats: Optional[np.ndarray] = None,
        desired_ee_velocity: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        current_joint_velocities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        # Outer Loop: Task-space IK control
        Compute target joint angles for desired end-effector pose.

        Args:
            ee_pos: Current end-effector position [x, y, z]
            ee_quat: Current end-effector quaternion [w, x, y, z]
            joint_angles: Current RELATIVE joint angles in radians (for limit checking)
            joint_quats: Optional canonical absolute quaternions matching
                ``joint_angles``. Rebuilt from angles only when omitted.
            desired_ee_velocity: Optional desired EE twist (position 3D or pose 6D) for velocity mode
            dt: Optional timestep (seconds). Defaults to self.default_dt.
            current_joint_velocities: Optional measured joint velocities (rad/s) for future adaptive use

        Returns:
            Target joint angles (relative)

        Note: joint_angles are canonical relative joint angles [slew, boom, arm, bucket].
        """

        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        ee_quat = np.asarray(ee_quat, dtype=np.float32)
        joint_angles = np.asarray(joint_angles, dtype=np.float32)
        dt_val = self.default_dt if dt is None else float(dt)
        dt_val = float(np.clip(dt_val, 1e-4, 1.0))

        # Reuse caller-provided canonical quats so FK/reachability/IK evaluate
        # the same orientation set; rebuild only as a fallback.
        if joint_quats is None:
            joint_quats = joint_angles_to_absolute_quaternions(joint_angles, self.robot_config)
        else:
            joint_quats = np.asarray(joint_quats, dtype=np.float32)

        # Compute full Jacobian from the canonical absolute orientations.
        jacobian_full = compute_jacobian_core(
            joint_quats,
            self.robot_config.link_lengths,
            self.robot_config.link_directions,
            self.robot_config.rotation_axes,
            self.robot_config.origin_offset,
            self.robot_config.ee_offset,
        )

        # Frame transform: express position and Jacobian in base (cab) frame to
        # avoid world-frame coupling between slew yaw and Cartesian position commands.
        current_pos = ee_pos
        target_pos = self.ee_pos_des

        # Extract slew angle (joint 0 rotates around Z)
        slew_angle = extract_axis_rotation(joint_quats[0], self.robot_config.rotation_axes[0])
        c = float(np.cos(-slew_angle))
        s = float(np.sin(-slew_angle))
        base_rot = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Rotate poses into base frame
        current_pos = (base_rot @ current_pos.astype(np.float32)).astype(np.float32)
        target_pos = (base_rot @ target_pos.astype(np.float32)).astype(np.float32)

        # Rotate full Jacobian rows before reduction (matches sim pipeline)
        jacobian_full[:3, :] = base_rot @ jacobian_full[:3, :]
        jacobian_full[3:, :] = base_rot @ jacobian_full[3:, :]

        # Extract reduced Jacobian (always -- auto-detected + ignore_axes)
        jacobian = self._get_reduced_jacobian(jacobian_full)

        # Initialize error containers
        position_error = np.zeros(3, dtype=np.float32)
        axis_angle_error = np.zeros(3, dtype=np.float32)

        # Compute pose error and solve IK
        if self.cfg.command_type == "position":
            position_error = target_pos - current_pos
            reduced_error = position_error
            # Use only position rows of the reduced Jacobian
            jacobian = jacobian[0:3, :]
        else:
            # Pose error is computed in world frame, then rotated into base (cab)
            # frame to match the body-frame Jacobian and body-frame position error
            # built above.  This is what makes a body-frame pitch command (boom/arm
            # /bucket Y axes) appear in the pitch row of the reduced Jacobian at
            # any slew angle — without it, at slew=π/2 the pitch error becomes
            # pure roll in world frame and is discarded by the controllable-DOF
            # reduction (no X-axis joint on the excavator).
            # The current target-quat recipe (slew_quat * pitch_quat) keeps the
            # yaw error at zero by construction, but extra tool roll/yaw axes will
            # be body-frame too — so this transform is forward-compatible.
            ee_quat_local = ee_quat
            target_quat_local = self.ee_quat_des

            # Debug: show orientation error
            if self.logger.level <= logging.DEBUG:
                try:
                    slew_angle = extract_axis_rotation(joint_quats[0], self.robot_config.rotation_axes[0])
                    slew_deg = float(np.degrees(slew_angle))
                    self.logger.debug(
                        "[IK dbg] slew_deg=%.2f | target_quat=%s | ee_quat=%s",
                        slew_deg,
                        np.round(self.ee_quat_des, 4),
                        np.round(ee_quat_local, 4),
                    )
                except Exception:
                    pass

            # Compute pose error in world frame, then rotate the orientation
            # component into base (cab) frame to match the body-frame Jacobian.
            position_error, axis_angle_error = compute_pose_error(
                current_pos, ee_quat_local, target_pos, target_quat_local
            )
            axis_angle_error = (base_rot @ axis_angle_error.astype(np.float32)).astype(np.float32)

            # Get reduced error vector (only auto-detected controllable DOFs)
            reduced_error = self._get_reduced_error(position_error, axis_angle_error)

        # Choose task vector based on mode (position vs velocity)
        if self.velocity_mode:
            desired_vec = self._prepare_desired_velocity(desired_ee_velocity, reduced_error.size)
            task_vec = desired_vec + (self.velocity_error_gain * reduced_error)
            joint_rate_cmd = self._compute_delta_joint_angles(task_vec, jacobian)
            delta_joint_angles = joint_rate_cmd * dt_val
        else:
            delta_joint_angles = self._compute_delta_joint_angles(reduced_error, jacobian)

        # DEBUG: Log raw IK output before any limiting
        self._ik_raw_debug_counter += 1
        if self.logger.level <= logging.DEBUG and self._ik_raw_debug_counter % 50 == 0:
            raw_delta_deg = np.degrees(delta_joint_angles)
            raw_max = np.max(np.abs(raw_delta_deg))
            if raw_max > 0.01:
                self.logger.debug(f"IK raw delta: {raw_delta_deg} (max={raw_max:.3f}deg)")

        # Apply velocity limiting (if enabled)
        if self.cfg.enable_velocity_limiting:
            delta_joint_angles = self._apply_velocity_limits(delta_joint_angles)

        # Add joint limit avoidance (uses RELATIVE angles)
        if self.cfg.enable_joint_limit_avoidance:
            delta_joint_angles = self._add_joint_limit_avoidance(
                delta_joint_angles, joint_angles
            )

        target_joint_angles = joint_angles + delta_joint_angles
        if self.joint_limits is not None:
            for i, (q_min, q_max) in enumerate(self.joint_limits):
                target_joint_angles[i] = np.clip(target_joint_angles[i], q_min, q_max)

        return target_joint_angles

    def _compute_adaptive_damping(self) -> float:
        """Compute adaptive damping factor based on Jacobian conditioning.

        Uses ``self.last_condition_number`` which must be set before calling.

        Returns:
            Adaptive damping value (lambda)
        """
        if not self.cfg.enable_adaptive_damping:
            lam = self.cfg.ik_params.get("lambda_val", 0.01)
            self.last_adaptive_lambda = float(lam)
            return lam

        cond = self.last_condition_number

        base_lambda = self.cfg.ik_params.get("lambda_val", 0.01)
        # Derive scaling so lambda peaks at base * max_multiplier exactly when
        # cond == condition_number_threshold (where the IK gate also fires).
        # This ties the two knobs together and makes max_multiplier meaningful.
        scaling = (self.cfg.adaptive_damping_max_multiplier - 1.0) / np.log(1.0 + self.cfg.condition_number_threshold)
        adaptive_lambda = base_lambda * (1.0 + scaling * np.log(1.0 + cond))
        lam = float(np.clip(adaptive_lambda, base_lambda, base_lambda * self.cfg.adaptive_damping_max_multiplier))
        self.last_adaptive_lambda = float(lam)
        return lam

    def _apply_velocity_limits(self, delta_joint_angles: np.ndarray) -> np.ndarray:
        """Limit joint velocities to prevent large jumps.

        Args:
            delta_joint_angles: Desired joint angle changes

        Returns:
            Velocity-limited joint angle changes
        """
        return np.clip(
            delta_joint_angles,
            -self.max_joint_velocities,
            self.max_joint_velocities
        )

    def _add_joint_limit_avoidance(
        self, delta_joint_angles: np.ndarray, joint_angles: np.ndarray
    ) -> np.ndarray:
        """Add repulsion forces to avoid joint limits.

        Args:
            delta_joint_angles: Current desired joint changes
            joint_angles: Current RELATIVE joint angles

        Returns:
            Modified joint changes with limit avoidance
        """
        repulsion_strength = 0.1  # Strength of repulsion force
        margin_fraction = 0.15    # Start repulsion at 15% from limits

        modified_delta = delta_joint_angles.copy()

        for i, (q_min, q_max) in enumerate(self.joint_limits):
            q_range = q_max - q_min
            margin = margin_fraction * q_range

            # Repulsion from lower limit
            if joint_angles[i] < q_min + margin:
                distance_ratio = np.clip((joint_angles[i] - q_min) / margin, 0.0, 1.0)
                repulsion = repulsion_strength * (1.0 - distance_ratio) ** 2
                modified_delta[i] += repulsion

            # Repulsion from upper limit
            elif joint_angles[i] > q_max - margin:
                distance_ratio = np.clip((q_max - joint_angles[i]) / margin, 0.0, 1.0)
                repulsion = repulsion_strength * (1.0 - distance_ratio) ** 2
                modified_delta[i] -= repulsion

        return modified_delta

    def _compute_delta_joint_angles(
        self, delta_pose: np.ndarray, jacobian: np.ndarray,
    ) -> np.ndarray:
        """Compute joint angle changes using specified IK method.

        If ``joint_weights`` is set in ``ik_params``, applies weighted
        pseudoinverse via ``W @ pinv(J @ W)`` where ``W = diag(sqrt(w_i))``.
        This gives **linear** effective weighting: a user weight of 0.8
        produces ~0.8x movement, not 0.64x (which would be W squared).
        """

        delta_pose = np.asarray(delta_pose, dtype=np.float32)
        jacobian = np.asarray(jacobian, dtype=np.float32)

        # Build sqrt weight matrix for joint weighting (linear effective weighting)
        joint_weights = self.cfg.ik_params.get("joint_weights", None) if self.cfg.ik_params else None
        if joint_weights is not None:
            w = np.asarray(joint_weights, dtype=np.float32)
            w = np.clip(w, 1e-6, None)
            w_sqrt = np.diag(np.sqrt(w))
            jacobian = np.dot(jacobian, w_sqrt)
        else:
            w_sqrt = None

        # Condition number is always needed (adaptive damping + gating).
        # Yoshikawa index and singular values are optional diagnostics.
        _cond, _sv, _yosh = compute_jacobian_metrics(jacobian)
        self.last_condition_number = _cond
        if self.cfg.enable_jacobian_metrics:
            _sv_out = np.zeros(4, dtype=np.float32)
            _sv_out[:min(4, len(_sv))] = _sv[:min(4, len(_sv))]
            self.last_singular_values = _sv_out
            self.last_yoshikawa_index = _yosh

        # Compute adaptive damping (uses self.last_condition_number)
        adaptive_lambda = self._compute_adaptive_damping()

        method = self.cfg.ik_method
        if method == "pinv":
            dq = ik_method_pinv(jacobian, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
        elif method == "svd":
            dq = ik_method_svd(
                jacobian, delta_pose,
                np.float32(self.cfg.ik_params["k_val"]),
                np.float32(self.cfg.ik_params["min_singular_value"]),
            )
        elif method == "trans":
            dq = ik_method_transpose(jacobian, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
        elif method == "dls":
            dq = ik_method_damped_least_squares(
                jacobian, delta_pose, np.float32(adaptive_lambda),
            )
        else:
            raise ValueError(f"Unknown IK method: {self.cfg.ik_method}")

        # Post-multiply by W_sqrt to complete the weighted pseudoinverse
        if w_sqrt is not None:
            dq = np.dot(w_sqrt, dq)

        return dq
