"""
Differential IK controller with numpy+numba optimization.

IMPORTANT: Origin and end-effector offset handling:
- forward_kinematics(): Returns joint positions (includes origin_offset, no ee_offset)
- forward_kinematics_with_ee_offset(): Returns (joint_positions, ee_position) tuple (both offsets applied)
- get_joint_positions(): Returns joint positions (includes origin_offset, no ee_offset)
- get_end_effector_position(): Returns actual end-effector position (both offsets applied)
- get_end_effector_orientation(): Returns end-effector orientation (same as last joint)
- origin_offset: Applied to all positions - shifts entire robot from world origin
- ee_offset: Applied in last joint's local coordinate frame - extends end-effector

To avoid confusion:
- All functions now include origin_offset in their position outputs
- Use get_end_effector_position() and get_end_effector_orientation() for control
- The IK controller automatically uses correct positions with both offsets
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

        # Compute pose error
        if self.cfg.command_type == "position":
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[0:3, :]
            delta_joint_angles = self._compute_delta_joint_angles(position_error, jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des
            )
            
            # Apply weighting to position and rotation errors
            pos_weight = self.cfg.ik_params.get("position_weight")
            rot_weight = self.cfg.ik_params.get("rotation_weight")

            # Combine weighted errors for IK solving
            pose_error = np.concatenate([
                np.float32(pos_weight) * position_error, 
                np.float32(rot_weight) * axis_angle_error
            ])

            delta_joint_angles = self._compute_delta_joint_angles(pose_error, jacobian)

        return joint_angles + delta_joint_angles


    def _compute_delta_joint_angles(self, delta_pose: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """Compute joint angle changes using specified IK method."""

        delta_pose = np.asarray(delta_pose, dtype=np.float32)
        jacobian = np.asarray(jacobian, dtype=np.float32)
        
        # Get joint weights
        joint_weights = self.cfg.ik_params.get("joint_weights", None)
        if joint_weights is None:
            joint_weights = [1.0] * jacobian.shape[1]  # Default to uniform weighting
        
        # Build weight matrix
        w_inv = self._build_weight_matrix_inv(joint_weights)

        if self.cfg.ik_method == "pinv":
            return ik_method_pinv(jacobian, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
        elif self.cfg.ik_method == "svd":
            return ik_method_svd(
                jacobian, delta_pose,
                np.float32(self.cfg.ik_params["k_val"]),
                np.float32(self.cfg.ik_params["min_singular_value"]),
                w_inv
            )
        elif self.cfg.ik_method == "trans":
            return ik_method_transpose(jacobian, delta_pose, np.float32(self.cfg.ik_params["k_val"]))
        elif self.cfg.ik_method == "dls":
            return ik_method_damped_least_squares(jacobian, delta_pose, np.float32(self.cfg.ik_params["lambda_val"]), w_inv)
        else:
            raise ValueError(f"Unknown IK method: {self.cfg.ik_method}")
    
    def _build_weight_matrix_inv(self, joint_weights) -> np.ndarray:
        """Build inverse weight matrix for joint weighting.
        
        Args:
            joint_weights: List of weights for each joint (higher = more preferred)
            
        Returns:
            Inverse weight matrix W^{-1} of shape [num_joints, num_joints]
        """
        w = np.asarray(joint_weights, dtype=np.float32)
        
        # Validation: prevent division by zero
        w = np.clip(w, 1e-6, None)  # Ensure all weights are at least 1e-6
        
        # Create diagonal matrix with inverse weights
        w_inv = np.diag(1.0 / w)
        return w_inv


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




class AngleUnwrapper:
    """
    Angle unwrapping filter for continuous joint angle tracking.
    Handles quaternion -> angle wrap-around issues at ±π boundaries.
    Provides encoder-like continuous angle streams for PID controllers.
    """
    
    def __init__(self, num_joints: int = 3):
        self.num_joints = num_joints
        self.prev_unwrapped_angles = None
        self.is_initialized = False
        
    def unwrap_angles(self, raw_wrapped_angles: np.ndarray) -> np.ndarray:
        """
        Apply angle unwrapping to maintain continuity.
        
        Args:
            raw_wrapped_angles: Array of angles in [-π, π] from quaternion extraction
            
        Returns:
            Continuous unwrapped angles that can exceed ±π for smooth PID control
        """
        raw_wrapped_angles = np.asarray(raw_wrapped_angles, dtype=np.float32)
        
        if not self.is_initialized:
            # First reading: use as-is, no unwrapping needed
            self.prev_unwrapped_angles = raw_wrapped_angles.copy()
            self.is_initialized = True
            return raw_wrapped_angles
            
        unwrapped_angles = np.zeros_like(raw_wrapped_angles)
        
        for i in range(len(raw_wrapped_angles)):
            raw_angle = raw_wrapped_angles[i]
            prev_unwrapped = self.prev_unwrapped_angles[i]
            
            # Calculate what the previous angle would be if wrapped to [-π, π]
            prev_wrapped = np.arctan2(np.sin(prev_unwrapped), np.cos(prev_unwrapped))
            
            # Calculate the wrapped difference
            diff = raw_angle - prev_wrapped
            
            # Detect wrap-around jumps and correct them
            if diff > np.pi:
                # Jumped from +π to -π (e.g., +179° -> -179°)
                unwrapped_angle = prev_unwrapped + diff - 2*np.pi
            elif diff < -np.pi:
                # Jumped from -π to +π (e.g., -179° -> +179°)
                unwrapped_angle = prev_unwrapped + diff + 2*np.pi
            else:
                # Normal case: no wrap-around
                unwrapped_angle = prev_unwrapped + diff
                
            unwrapped_angles[i] = unwrapped_angle
        
        # Update state for next iteration
        self.prev_unwrapped_angles = unwrapped_angles.copy()
        return unwrapped_angles
        
    def reset(self):
        """Reset unwrapper state (e.g., on system restart)."""
        self.prev_unwrapped_angles = None
        self.is_initialized = False


@numba.njit(fastmath=False)
def extract_axis_rotation(quat, axis):
    """Extract rotation angle around specified axis from quaternion.
    
    Note: Assumes quat is already normalized
    For Y-axis rotations, handles quaternion sign ambiguity for >90° rotations.
    Optimized for excavator use (limited rotation ranges, no fast motion).
    """
    quat = np.asarray(quat, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)

    # Check if this is Y-axis extraction (excavator joints are Y-axis)
    if np.allclose(axis, np.array([0.0, 1.0, 0.0], dtype=np.float32)):
        # Robust Y-axis extraction with quaternion sign ambiguity handling
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Method 1: Direct extraction with original quaternion
        angle_1 = np.float32(2.0) * np.arctan2(y, w)
        
        # Method 2: Try negated quaternion (handles sign ambiguity)
        # For quaternion sign ambiguity, try -quat which represents same rotation
        neg_quat = -quat
        neg_w, neg_x, neg_y, neg_z = neg_quat[0], neg_quat[1], neg_quat[2], neg_quat[3]
        angle_2 = np.float32(2.0) * np.arctan2(neg_y, neg_w)
        
        # Try both original and negated quaternion and choose more reasonable result
        wrapped_angle1 = wrap_to_pi(angle_1)[()]
        wrapped_angle2 = wrap_to_pi(angle_2)[()]
        
        # Since axis projection works correctly (x,z = 0), use a simpler hemisphere check
        # For Y-axis rotations, both +q and -q represent the same rotation
        # Choose the quaternion that gives the smaller magnitude angle (closer to 0)
        if abs(wrapped_angle2) < abs(wrapped_angle1):
            return wrapped_angle2
        
        # Default: return the wrapped angle from original quaternion
        return wrapped_angle1
    else:
        # For other axes, use the general axis-angle projection method
        axis_angle = axis_angle_from_quat(quat)
        projected_magnitude = np.dot(np.asarray(axis_angle, dtype=np.float32), np.asarray(axis, dtype=np.float32))
        return wrap_to_pi(projected_magnitude)[()]  # Extract scalar from 0-d array




# Wrapper functions for Numba usage
def project_to_rotation_axes(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Project quaternions to only their configured rotation axes (removes yaw for Y-axis joints).
    This is the universal yaw projection function for debugging.
    
    Args:
        quats: Input quaternions [n x 4] 
        robot_config: Robot configuration with rotation axes
        
    Returns:
        Projected quaternions with only rotation around specified axes
    """
    quats = np.asarray(quats, dtype=np.float32)
    
    # Create axis-constrained quaternions (removes unwanted rotations like yaw)
    constrained_quats = np.zeros_like(quats, dtype=np.float32)
    for i in range(len(quats)):
        # Extract rotation angle around the specified axis only
        axis = robot_config.rotation_axes[i]
        angle = extract_axis_rotation(quats[i], axis)
        # Create pure single-axis quaternion (eliminates unwanted rotations)
        constrained_quats[i] = quat_from_axis_angle(axis, angle)
    
    return constrained_quats






def compute_jacobian(quats: np.ndarray, robot_config: RobotConfig):
    """Jacobian computation wrapper using RobotConfig."""
    quats = np.asarray(quats, dtype=np.float32)
    return compute_jacobian_core(
        quats,
        robot_config.link_lengths,  # Already np.array from RobotConfig
        robot_config.link_directions,  # Already list of np.arrays from RobotConfig
        robot_config.rotation_axes,  # Already list of np.arrays from RobotConfig
        robot_config.origin_offset,
        robot_config.ee_offset
    )


def get_end_effector_position(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Get end-effector position with offset applied.
    
    Returns:
        np.ndarray: End-effector position [x, y, z] including ee_offset
    """
    constrained_quats = project_to_rotation_axes(quats, robot_config)
    _, ee_position = forward_kinematics_with_ee_offset_core(
        constrained_quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.origin_offset,
        robot_config.ee_offset
    )
    return ee_position


def get_joint_positions(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Get joint positions with origin_offset applied (no ee_offset).
    
    Returns:
        np.ndarray: Joint positions [n x 3] including origin_offset, without end-effector offset
    """
    constrained_quats = project_to_rotation_axes(quats, robot_config)
    return forward_kinematics_core(
        constrained_quats,
        robot_config.link_lengths,  # Already np.array from RobotConfig
        robot_config.link_directions,  # Already list of np.arrays from RobotConfig  
        robot_config.origin_offset
    )


def get_end_effector_orientation(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Get end-effector orientation quaternion.
    
    For a serial manipulator, the end-effector orientation is the orientation
    of the last joint (bucket joint in the excavator case).
    
    Args:
        quats: Joint quaternions from IMUs
        robot_config: Robot configuration
        
    Returns:
        np.ndarray: End-effector orientation quaternion [w, x, y, z]
    """
    constrained_quats = project_to_rotation_axes(quats, robot_config)

    # End-effector orientation is the last joint's orientation
    # float32 ?
    return constrained_quats[-1].copy()


def warmup_numba_functions():
    """
    Warmup Numba JIT compilation by calling functions with dummy data.
    This prevents compilation delays during actual operation.
    """
    print("Warming up Numba functions...")
    
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
            
        print("SUCCESS: Numba functions compiled successfully")
    except Exception as e:
        print(f"WARNING: Numba warmup failed: {e}")
        print("  Functions will compile on first use")


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
                            bucket_length: float = 0.030) -> RobotConfig:

    # bucket len = center to tool mount (30mm)
    # ee offset = 140mm below tool mount center

    # origin offset (from robot origin to first joint) = x+16,5mm z+64,5mm


    """Create excavator robot configuration with 3 links, Y-axis rotation.
    
    Args:
        boom_length: Length of boom link in meters (default: 0.468m)
        arm_length: Length of arm link in meters (default: 0.250m)  
        bucket_length: Length of bucket link in meters (default: 0.059m)
        ee_offset: Optional end-effector offset [x, y, z] in meters from last joint.
                  If None, uses default excavator ee_offset configuration.
        origin_offset: Optional origin offset [x, y, z] in meters for first joint position.
                      If None, uses default excavator origin_offset configuration.
    
    Returns:
        RobotConfig: Configured excavator robot with proper IMU offsets, ee_offset, and origin_offset
    """

    # Create IMU offsets as quaternions using helper function
    imu_offsets = [
        create_imu_offset_quat(13.85),  # First IMU/joint - measured calibration value
        create_imu_offset_quat(0.0),    # Second IMU/joint - no offset
        create_imu_offset_quat(0.0)     # Third IMU/joint - no offset
    ]


    return RobotConfig(
        link_lengths=[float(boom_length), float(arm_length), float(bucket_length)],
        link_directions=[
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ],
        rotation_axes=[
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ],
        imu_offsets=imu_offsets,
        ee_offset=np.array([0.0, 0.0, -0.140], dtype=np.float32),      # x+0mm y+0mm z-140mm
        origin_offset=np.array([0.0165, 0.0, 0.0645], dtype=np.float32) # x+16,5mm z+64,5mm
    )