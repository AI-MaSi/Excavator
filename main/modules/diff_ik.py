"""
Differential IK controller with numpy+numba optimization.

IMPORTANT: Origin and end-effector offset handling:
- get_joint_positions(): Returns joint positions (includes origin_offset, no ee_offset)
- get_pose(): Returns (ee_position, ee_orientation) tuple with both offsets applied
- origin_offset: Applied to all positions - shifts entire robot from world origin
- ee_offset: Applied in last joint's local coordinate frame - extends end-effector

Notes:
- All quaternions are float32 and use [w, x, y, z] convention
- We assume FULL IMU quaternions on input; we explicitly project to joint axes
  (e.g., Y for boom/arm/bucket) before kinematics and IK.
- Base rotation propagation handles slew (Z-axis) from encoder.
"""





import numpy as np
import numba
from dataclasses import dataclass
from typing import Literal, Dict, Optional, List, Tuple

from modules.quaternion_math import (
    quat_normalize, quat_multiply, quat_conjugate, quat_rotate_vector, 
    quat_from_axis_angle, axis_angle_from_quat, wrap_to_pi, compute_pose_error,
    apply_delta_pose
)


# ----------------------------
# Axis-rotation helpers (FULL input -> axis twist)
# ----------------------------

def extract_axis_rotation(quat: np.ndarray, axis: np.ndarray) -> float:
    """
    Extract twist angle about `axis` from quaternion `quat` using swing–twist.

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
    return float(wrap_to_pi(np.float32(ang))[()])


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


@dataclass
class IKControllerConfig:
    """Configuration for differential inverse kinematics controller."""

    command_type: Literal["position", "pose"]
    """Type of task-space command: 'position' (3D) or 'pose' (6D)."""

    ik_method: Literal["pinv", "svd", "trans", "dls"]
    """Method for computing Jacobian inverse."""

    use_relative_mode: bool = False
    """Whether commands are relative to current pose."""

    ik_params: Optional[Dict[str, float]] = None
    """Method-specific parameters."""

    def __post_init__(self):
        # Validate inputs
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Invalid command_type: {self.command_type}")
        if self.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Invalid ik_method: {self.ik_method}")

        # Set default parameters
        default_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 0.1}
        }

        params = default_params[self.ik_method].copy()
        
        # Add position/rotation weighting defaults
        params.update({
            "position_weight": 1.0,
            "rotation_weight": 1.0,
            "joint_weights": None  # Default to uniform weighting
        })
        
        if self.ik_params:
            params.update(self.ik_params)
        self.ik_params = params


@dataclass
class RobotConfig:
    """Robot kinematic configuration."""

    link_lengths: List[float]
    """Length of each link."""

    link_directions: Optional[List[np.ndarray]] = None
    """Local direction vector for each link (default: x-axis)."""

    rotation_axes: Optional[List[np.ndarray]] = None
    """Rotation axis for each joint (default: z for first, y for others)."""

    imu_offsets: Optional[List[np.ndarray]] = None
    """IMU mounting offset quaternions [w, x, y, z] for each joint."""

    ee_offset: Optional[np.ndarray] = None
    """End-effector position offset from last joint in local frame [x, y, z]."""

    origin_offset: Optional[np.ndarray] = None
    """Origin offset - position of first joint relative to world origin [x, y, z]."""

    def __post_init__(self):
        if not self.link_lengths:
            raise ValueError("link_lengths cannot be empty")

        self.num_links = len(self.link_lengths)
        self.num_joints = self.num_links

        # Set defaults
        if self.link_directions is None:
            self.link_directions = [np.array([1.0, 0.0, 0.0], dtype=np.float32)
                                    for _ in range(self.num_links)]

        if self.rotation_axes is None:
            self.rotation_axes = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]  # First joint: Z
            self.rotation_axes.extend([np.array([0.0, 1.0, 0.0], dtype=np.float32)
                                       for _ in range(1, self.num_joints)])  # Others: Y

        if self.imu_offsets is None:
            # Identity quaternions (no offset)
            self.imu_offsets = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                                for _ in range(self.num_joints)]

        # Validate dimensions
        if len(self.link_directions) != self.num_links:
            raise ValueError(f"Expected {self.num_links} link directions")
        if len(self.rotation_axes) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} rotation axes")
        if len(self.imu_offsets) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} IMU offsets")

        # Set default ee_offset if not provided
        if self.ee_offset is None:
            self.ee_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Set default origin_offset if not provided
        if self.origin_offset is None:
            self.origin_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ENSURE ALL ARRAYS ARE FLOAT32
        self.link_lengths = np.asarray(self.link_lengths, dtype=np.float32)  # Convert list to numpy array!
        self.link_directions = [np.asarray(arr, dtype=np.float32) for arr in self.link_directions]
        self.rotation_axes = [np.asarray(arr, dtype=np.float32) for arr in self.rotation_axes]
        self.imu_offsets = [np.asarray(arr, dtype=np.float32) for arr in self.imu_offsets]
        self.ee_offset = np.asarray(self.ee_offset, dtype=np.float32)
        self.origin_offset = np.asarray(self.origin_offset, dtype=np.float32)


class IKController:
    """Differential inverse kinematics controller with numpy+numba optimization."""

    def __init__(self, cfg: IKControllerConfig, robot_config: RobotConfig):
        self.cfg = cfg
        self.robot_config = robot_config

        # Target pose buffers
        self.ee_pos_des = np.zeros(3, dtype=np.float32)
        self.ee_quat_des = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._command = np.zeros(self.action_dim, dtype=np.float32)

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

    def compute(self, ee_pos: np.ndarray, ee_quat: np.ndarray, joint_angles: np.ndarray, joint_quats: np.ndarray) -> np.ndarray:
        """
        # Outer Loop: Task-space IK control
        Compute target joint angles for desired end-effector pose.

        Args:
            ee_pos: Current end-effector position [x, y, z]
            ee_quat: Current end-effector quaternion [w, x, y, z]
            joint_angles: Current joint angles in radians
            joint_quats: Current joint quaternions from IMUs (with mounting offsets applied)

        Returns:
            Target joint angles
        """

        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        ee_quat = np.asarray(ee_quat, dtype=np.float32)
        joint_angles = np.asarray(joint_angles, dtype=np.float32)

        # Handle quaternions: use IMU data if available, otherwise create from joint angles
        if joint_quats is not None:
            joint_quats = np.asarray(joint_quats, dtype=np.float32)
        else:
            # Create quaternions from joint angles and rotation axes
            joint_quats = np.zeros((self.robot_config.num_joints, 4), dtype=np.float32)
            for i in range(self.robot_config.num_joints):
                axis = self.robot_config.rotation_axes[i]
                angle = joint_angles[i] if i < len(joint_angles) else 0.0
                joint_quats[i] = quat_from_axis_angle(axis, angle)
        
        # Compute Jacobian
        jacobian = compute_jacobian(joint_quats, self.robot_config)

        # Use unweighted Jacobian; prioritize joints only via 'joint_weights'
        J_weighted = jacobian.copy()

        # Compute pose error and solve IK
        if self.cfg.command_type == "position":
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = J_weighted[0:3, :]
            delta_joint_angles = self._compute_delta_joint_angles(position_error, jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des
            )

            # Apply weighting to position and rotation errors (full axis-angle)
            pos_weight = self.cfg.ik_params.get("position_weight")
            rot_weight = self.cfg.ik_params.get("rotation_weight")

            pose_error = np.concatenate([
                np.float32(pos_weight) * position_error,
                np.float32(rot_weight) * axis_angle_error
            ])

            delta_joint_angles = self._compute_delta_joint_angles(pose_error, J_weighted)

        return joint_angles + delta_joint_angles


    def _compute_delta_joint_angles(self, delta_pose: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """Compute joint angle changes using specified IK method."""

        delta_pose = np.asarray(delta_pose, dtype=np.float32)
        jacobian = np.asarray(jacobian, dtype=np.float32)
        
        # Get joint weights (higher = more preferred/more movement)
        joint_weights = self.cfg.ik_params.get("joint_weights", None)
        if joint_weights is None:
            joint_weights = [1.0] * jacobian.shape[1]

        # Build weight matrix W (diagonal of multipliers)
        w_mat = self._build_weight_matrix(joint_weights)

        # Apply weighting depending on method
        method = self.cfg.ik_method
        if method == "pinv":
            # Use Jacobian with column scaling, map back via W
            jac_w = np.dot(jacobian, w_mat)
            dq_prime = ik_method_pinv(jac_w, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
            return np.dot(w_mat, dq_prime)
        elif method == "svd":
            return ik_method_svd(
                jacobian, delta_pose,
                np.float32(self.cfg.ik_params["k_val"]),
                np.float32(self.cfg.ik_params["min_singular_value"]),
                w_mat
            )
        elif method == "trans":
            # For transpose, scaling columns and using J_w^T gives W * J^T * e directly
            jac_w = np.dot(jacobian, w_mat)
            return ik_method_transpose(jac_w, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
        elif method == "dls":
            return ik_method_damped_least_squares(jacobian, delta_pose, np.float32(self.cfg.ik_params["lambda_val"]), w_mat)
        else:
            raise ValueError(f"Unknown IK method: {self.cfg.ik_method}")

    def _build_weight_matrix(self, joint_weights) -> np.ndarray:
        """Build weight matrix for joint weighting (multipliers).

        Args:
            joint_weights: List of per-joint multipliers (higher = more preferred)

        Returns:
            Diagonal weight matrix W of shape [num_joints, num_joints]
        """
        w = np.asarray(joint_weights, dtype=np.float32)
        # Prevent degenerate zeros; allow near-zero to effectively disable a joint
        w = np.clip(w, 1e-6, None)
        return np.diag(w)


# Numba-optimized kinematics functions


@numba.njit(fastmath=False)
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
    # Inputs should already be properly formatted arrays
    
    n = len(link_lengths)
    joint_positions = np.zeros((n, 3), dtype=np.float32)

    # Start at origin offset (first joint position)
    pos = origin_offset.copy()

    for i in range(n):
        # Link vector in local frame → rotate into world using absolute orientation
        link_vec = link_lengths[i] * link_directions[i]
        # quats[i] is already normalized, no need to normalize again
        world_link_vec = quat_rotate_vector(quats[i], link_vec)

        # Add to current position
        pos = pos + world_link_vec
        joint_positions[i] = pos

    return joint_positions


@numba.njit(fastmath=False)
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
    
    # Apply ee_offset in the last joint's local frame
    if len(joint_positions) > 0:
        last_joint_quat = quats[-1]
        # Transform ee_offset from local frame to world frame
        world_ee_offset = quat_rotate_vector(last_joint_quat, ee_offset)
        # Add offset to last joint position to get true end-effector position
        ee_position = joint_positions[-1] + world_ee_offset
    else:
        ee_position = np.asarray(origin_offset, dtype=np.float32).copy()
    
    return joint_positions, ee_position


@numba.njit(fastmath=False)
def compute_jacobian_core(quats, link_lengths, link_directions, rotation_axes, origin_offset, ee_offset):
    # Inputs should already be properly formatted arrays

    n = len(link_lengths)
    jacobian = np.zeros((6, n), dtype=np.float32)

    joint_positions = forward_kinematics_core(quats, link_lengths, link_directions, origin_offset)

    # Calculate actual end-effector position with offset
    if len(joint_positions) > 0:
        last_joint_quat = quats[-1]
        world_ee_offset = quat_rotate_vector(last_joint_quat, ee_offset)
        ee_pos = joint_positions[-1] + world_ee_offset
    else:
        ee_pos = np.asarray(origin_offset, dtype=np.float32).copy()

    # Base position is now the origin offset
    base_pos = np.asarray(origin_offset, dtype=np.float32).copy()
    rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(n):
        # Normalize after multiplication
        rot = quat_normalize(quat_multiply(rot, quats[i]))
        world_axis = quat_rotate_vector(rot, np.asarray(rotation_axes[i], dtype=np.float32))

        if i == 0:
            joint_pos = base_pos
        else:
            joint_pos = joint_positions[i-1]

        # Linear velocity component
        jacobian[0:3, i] = np.cross(world_axis, ee_pos - joint_pos)
        # Angular velocity component
        jacobian[3:6, i] = world_axis

    return jacobian


# Numba-optimized IK methods - ALL FLOAT32
@numba.njit(fastmath=False)
def ik_method_pinv(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Pseudo-inverse IK method."""
    jacobian_pinv = np.linalg.pinv(np.asarray(jacobian, dtype=np.float32))
    return k_val * np.dot(jacobian_pinv, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=False)
def ik_method_svd(jacobian: np.ndarray, delta_pose: np.ndarray,
                 k_val: np.float32, min_singular_value: np.float32, 
                 w_inv: np.ndarray) -> np.ndarray:
    """SVD-based IK method with singular value thresholding and joint weighting."""
    jac_f32 = np.asarray(jacobian, dtype=np.float32)
    delta_f32 = np.asarray(delta_pose, dtype=np.float32)
    w_inv_f32 = np.asarray(w_inv, dtype=np.float32)

    # Apply joint weighting: J_weighted = J * W^{-1}
    jac_weighted = np.dot(jac_f32, w_inv_f32)
    
    U, S, Vh = np.linalg.svd(jac_weighted, full_matrices=False)

    # Compute U^T * delta_pose
    ut_delta = np.dot(U.T, delta_f32)

    # Apply pseudoinverse of singular values with thresholding
    result = np.zeros(jac_weighted.shape[1], dtype=np.float32)
    for i in range(len(S)):
        if S[i] > min_singular_value:
            result += Vh[i] * (ut_delta[i] / S[i])

    # Apply weight matrix: delta_q = W^{-1} * J_pinv * delta_pose
    return k_val * np.dot(w_inv_f32, result)


@numba.njit(fastmath=False)
def ik_method_transpose(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Jacobian transpose IK method."""
    return k_val * np.dot(np.asarray(jacobian, dtype=np.float32).T, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=False)
def ik_method_damped_least_squares(jacobian: np.ndarray, delta_pose: np.ndarray, lambda_val: np.float32, w_inv: np.ndarray) -> np.ndarray:
    """Damped least squares IK method with joint weighting."""
    jac_f32 = np.asarray(jacobian, dtype=np.float32)
    delta_f32 = np.asarray(delta_pose, dtype=np.float32)
    w_inv_f32 = np.asarray(w_inv, dtype=np.float32)

    # Apply joint weighting: J_weighted = J * W^{-1}
    jac_weighted = np.dot(jac_f32, w_inv_f32)

    # (J_W * J_W^T + λ²I)^-1
    jjt = np.dot(jac_weighted, jac_weighted.T)
    lambda_matrix = (lambda_val ** 2) * np.eye(jjt.shape[0], dtype=np.float32)
    jjt_lambda_inv = np.linalg.inv(jjt + lambda_matrix)

    # delta_q = W^{-1} * J_W^T * (J_W*J_W^T + λ²I)^-1 * Δp
    return np.dot(w_inv_f32, np.dot(jac_weighted.T, np.dot(jjt_lambda_inv, delta_f32)))


def apply_imu_offsets(imu_quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Apply IMU mounting offset corrections using full quaternion system.
    Includes quaternion hemisphere selection for excavator constraints.
    Returns corrected quaternions preserving all rotation information.
    """
    imu_quats = np.asarray(imu_quats, dtype=np.float32)
    corrected_quats = np.zeros_like(imu_quats, dtype=np.float32)

    for i in range(len(imu_quats)):
        # Normalize raw IMU quaternion
        normalized_imu_quat = quat_normalize(imu_quats[i])

        # Apply mounting offset correction
        offset_quat_inv = quat_conjugate(robot_config.imu_offsets[i])
        # Normalize after multiplication
        corrected_quat = quat_normalize(
            quat_multiply(offset_quat_inv, normalized_imu_quat)
        )


        # Store the full corrected quaternion
        corrected_quats[i] = corrected_quat

    return corrected_quats




# Wrapper functions for Numba usage


def propagate_base_rotation(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Propagate base joint rotation to downstream joints in kinematic chain.

    For 6-axis IMUs without magnetometers, yaw (Z-rotation) is unmeasurable.
    When the base joint (slew) rotates, downstream links (boom/arm/bucket)
    physically rotate in the world XY plane, but their IMUs cannot detect this.

    This function propagates the base rotation (from encoder) through the kinematic
    chain by composing it with downstream IMU quaternions, transforming them from
    local frames to the world frame.

    Args:
        quats: Input quaternions [base, link1, link2, ...] where base rotation is
               measured by encoder and downstream orientations are from IMUs
        robot_config: Robot configuration (unused but kept for consistency)

    Returns:
        Quaternions with base rotation propagated to all downstream joints
    """
    quats = np.asarray(quats, dtype=np.float32)
    propagated_quats = quats.copy()

    # Extract base joint quaternion (Z-rotation from encoder)
    base_quat = quats[0]

    # Propagate base rotation to downstream joints
    # q_world = q_base * q_local
    for i in range(1, len(quats)):
        propagated_quats[i] = quat_normalize(quat_multiply(base_quat, quats[i]))

    return propagated_quats




def compute_jacobian(quats: np.ndarray, robot_config: RobotConfig):
    """Jacobian computation wrapper using RobotConfig.

    Pipeline (FULL input):
      1) Apply IMU mounting offsets
      2) Project to joint rotation axes (e.g., Y for boom/arm/bucket)
      3) Propagate base (slew) rotation to downstream joints
    """
    quats = np.asarray(quats, dtype=np.float32)
    corrected_quats = apply_imu_offsets(quats, robot_config)
    constrained_quats = project_to_rotation_axes(corrected_quats, robot_config.rotation_axes)
    propagated_quats = propagate_base_rotation(constrained_quats, robot_config)
    return compute_jacobian_core(
        propagated_quats,
        robot_config.link_lengths,  # Already np.array from RobotConfig
        robot_config.link_directions,  # Already list of np.arrays from RobotConfig
        robot_config.rotation_axes,  # Already list of np.arrays from RobotConfig
        robot_config.origin_offset,
        robot_config.ee_offset
    )


def get_joint_positions(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Get joint positions with origin_offset applied (no ee_offset).

    Returns:
        np.ndarray: Joint positions [n x 3] including origin_offset, without end-effector offset
    """
    quats = np.asarray(quats, dtype=np.float32)
    corrected_quats = apply_imu_offsets(quats, robot_config)
    constrained_quats = project_to_rotation_axes(corrected_quats, robot_config.rotation_axes)
    propagated_quats = propagate_base_rotation(constrained_quats, robot_config)
    return forward_kinematics_core(
        propagated_quats,
        robot_config.link_lengths,  # Already np.array from RobotConfig
        robot_config.link_directions,  # Already list of np.arrays from RobotConfig
        robot_config.origin_offset
    )


def get_all_poses(quats: np.ndarray, robot_config: RobotConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get all joint poses (positions + orientations) and end-effector pose.

    This is the most comprehensive FK function - returns everything computed.
    Zero overhead compared to get_pose() since all values are computed anyway.

    Args:
        quats: Joint quaternions from IMUs
        robot_config: Robot configuration

    Returns:
        Tuple of:
        - joint_positions [n x 3]: Position of each joint (with origin_offset)
        - joint_orientations [n x 4]: Orientation of each joint [w, x, y, z]
        - ee_position [3]: End-effector position (with origin_offset + ee_offset)
        - ee_orientation [4]: End-effector orientation [w, x, y, z]
    """
    quats = np.asarray(quats, dtype=np.float32)
    corrected_quats = apply_imu_offsets(quats, robot_config)
    constrained_quats = project_to_rotation_axes(corrected_quats, robot_config.rotation_axes)
    propagated_quats = propagate_base_rotation(constrained_quats, robot_config)

    # Get joint positions and ee_position with offsets
    joint_positions, ee_position = forward_kinematics_with_ee_offset_core(
        propagated_quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.origin_offset,
        robot_config.ee_offset
    )

    # Joint orientations are the propagated quaternions
    joint_orientations = propagated_quats.copy()

    # End-effector orientation is the last joint's orientation
    ee_orientation = propagated_quats[-1].copy()

    return joint_positions, joint_orientations, ee_position, ee_orientation


def get_pose(quats: np.ndarray, robot_config: RobotConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Get end-effector pose (position and orientation) only.

    Convenience wrapper around get_all_poses() for when you only need EE pose.

    Args:
        quats: Joint quaternions from IMUs
        robot_config: Robot configuration

    Returns:
        Tuple of (ee_position [x, y, z], ee_orientation [w, x, y, z])
    """
    _, _, ee_position, ee_orientation = get_all_poses(quats, robot_config)
    return ee_position, ee_orientation


def warmup_numba_functions():
    """
    Warmup Numba JIT compilation by calling functions with dummy data.
    This prevents compilation delays during actual operation.
    """
    # Create dummy data with correct types
    dummy_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.3, 0.0], 
        [0.95, 0.0, 0.2, 0.0]
    ], dtype=np.float32)
    
    dummy_link_lengths = np.array([0.2, 0.15, 0.1], dtype=np.float32)
    dummy_link_directions = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)
    dummy_rotation_axes = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], 
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    dummy_origin_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dummy_ee_offset = np.array([0.0, 0.0, -0.1], dtype=np.float32)
    
    # Warmup all Numba functions
    try:
        dummy_jacobian = np.random.rand(6, 3).astype(np.float32)
        dummy_delta_pose = np.random.rand(6).astype(np.float32)
        dummy_w_inv = np.eye(3, dtype=np.float32)
        
        for _ in range(3):  # Multiple calls to ensure compilation
            # Forward kinematics functions
            _ = forward_kinematics_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_origin_offset)
            _ = forward_kinematics_with_ee_offset_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_origin_offset, dummy_ee_offset)
            _ = compute_jacobian_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_rotation_axes, dummy_origin_offset, dummy_ee_offset)
            
            # IK method functions
            _ = ik_method_pinv(dummy_jacobian, dummy_delta_pose, np.float32(1.0))
            _ = ik_method_svd(dummy_jacobian, dummy_delta_pose, np.float32(1.0), np.float32(1e-5), dummy_w_inv)
            _ = ik_method_transpose(dummy_jacobian, dummy_delta_pose, np.float32(1.0))
            _ = ik_method_damped_least_squares(dummy_jacobian, dummy_delta_pose, np.float32(0.1), dummy_w_inv)
    except Exception as e:
        print(f"Numba warmup failed: {e}")


def create_imu_offset_quat(angle_degrees: float) -> np.ndarray:
    """Helper function to create IMU offset quaternion for Y-axis rotation.
    
    Args:
        angle_degrees: Rotation angle in degrees around Y-axis
        
    Returns:
        np.ndarray: Quaternion [w, x, y, z] for the offset
    """
    angle = np.float32(np.radians(angle_degrees))
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return quat_from_axis_angle(axis, angle)


# Example robot configuration
def create_excavator_config(boom_length: float = 0.468, arm_length: float = 0.250,
                            bucket_length: float = 0.031) -> RobotConfig:
    """Create excavator robot configuration with 4 joints: slew (Z-axis) + boom/arm/bucket (Y-axis).

    Physical geometry:
    - Slew link: Offset from slew rotation axis to boom mounting point (x+16.5mm, z+64.5mm)
    - Boom/Arm/Bucket: Serial links extending in local X direction
    - End-effector: Tool tip 140mm below bucket mounting center

    Args:
        boom_length: Length of boom link in meters
        arm_length: Length of arm link in meters
        bucket_length: Length of bucket link in meters

    Returns:
        RobotConfig: Configured excavator robot with proper IMU offsets, ee_offset, and origin_offset
    """

    # Physical offset from slew axis to boom mounting point (rotates with slew!)
    slew_offset_vec = np.array([0.0165, 0.0, 0.0645], dtype=np.float32)  # x+16.5mm, z+64.5mm
    slew_length = float(np.linalg.norm(slew_offset_vec))
    slew_direction = slew_offset_vec / slew_length  # Normalized direction

    # Create IMU offsets as quaternions using helper function
    imu_offsets = [
        create_imu_offset_quat(0.0),    # Slew joint - no IMU offset needed for encoder
        create_imu_offset_quat(13.85),  # Boom IMU/joint
        create_imu_offset_quat(-0.61),    # Arm IMU/joint # TODO: check offset orientation!
        create_imu_offset_quat(0.0)     # Bucket IMU/joint - no offset
    ]

    return RobotConfig(
        link_lengths=[slew_length, float(boom_length), float(arm_length), float(bucket_length)],
        link_directions=[
            slew_direction,  # Slew: from axis to boom mount (radial + vertical offset).
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Boom: horizontal (X-axis)
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Arm: horizontal (X-axis)
            np.array([1.0, 0.0, 0.0], dtype=np.float32)   # Bucket: horizontal (X-axis)
        ],
        rotation_axes=[
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Slew: Z-axis rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Boom: Y-axis rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Arm: Y-axis rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32)   # Bucket: Y-axis rotation
        ],
        imu_offsets=imu_offsets,
        ee_offset=np.array([0.0, 0.0, -0.142], dtype=np.float32),  # below bucket center
        origin_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32)   # Slew axis is world origin
    )
