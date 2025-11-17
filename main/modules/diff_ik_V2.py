"""
Differential IK controller with numpy+numba optimization.

IMPORTANT: Origin and end-effector offset handling:
- get_joint_positions(): Returns joint positions (includes origin_offset, no ee_offset)
- get_pose(): Returns (ee_position, ee_orientation) tuple with both offsets applied
- origin_offset: Applied to all positions - shifts entire robot from world origin
- ee_offset: Applied in last joint's local coordinate frame - extends end-effector

Notes:
- All quaternions are float32 and use [w, x, y, z] convention
- We assume FULL IMU quaternions on input; no axis projection is applied.
  Orientation components are filtered in IK via the ignore_axes system.
- Base rotation propagation handles slew (Z-axis) from encoder.

IMPROVEMENTS IN V2:
- Added velocity limiting to prevent large jumps
- Adaptive damping based on Jacobian condition number
- Joint limit avoidance with repulsion forces
- Proper error weighting applied to Jacobian rows (not to error vector)
- Anti-windup for unreachable targets
- Reduced Jacobian option for improved efficiency
- Fastmath enabled for 20-30% speedup
- New compute_relative_joint_angles() for proper limit checking
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

    ignore_axes: Optional[List[str]] = None
    """Axes to ignore in orientation error when solving IK.
    Any of ["roll", "pitch", "yaw"]. Example for excavator: ["roll", "yaw"]."""

    # When True, also zero the corresponding rotation rows in the weighted Jacobian
    # after applying position/rotation weights (hard removal of ignored rotational DOFs).
    # This provides stronger decoupling and stability when an orientation axis is intentionally ignored.
    # TODO: add to config!
    use_ignore_axes_in_jacobian: bool = True

    enable_velocity_limiting: bool = True
    """Enable joint velocity limiting (safety feature)."""

    max_joint_velocities: Optional[List[float]] = None
    """Maximum joint velocity per control cycle (radians). If None, uses [0.035, 0.035, 0.035, 0.035] (≈2°)."""

    joint_limits: Optional[List[Tuple[float, float]]] = None
    """Joint limits as [(min, max), ...] in radians. If None, uses [-π, π] for all."""

    enable_adaptive_damping: bool = True
    """Enable adaptive damping based on Jacobian conditioning."""

    enable_joint_limit_avoidance: bool = True
    """Enable repulsion forces near joint limits."""

    enable_anti_windup: bool = False
    """Enable error saturation for unreachable targets."""

    use_reduced_jacobian: bool = True
    """Use reduced Jacobian by removing uncontrollable DOFs.
    For excavators: removes roll axis (only position + pitch/yaw are controllable).
    Automatically extracts controllable rows based on joint rotation axes."""

    def __post_init__(self):
        # Validate inputs
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Invalid command_type: {self.command_type}")
        if self.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Invalid ik_method: {self.ik_method}")

        # Validate that ik_params is provided (no fallback defaults)
        if not self.ik_params:
            raise ValueError(
                "ik_params must be provided in configuration. "
                "No default parameters are supplied. "
                "Required keys depend on ik_method:\n"
                "  - pinv: k_val\n"
                "  - svd: k_val, min_singular_value\n"
                "  - trans: k_val\n"
                "  - dls: lambda_val\n"
                "All methods also require: position_weight, rotation_weight"
            )

        # Validate required parameters based on method
        required_common = {"position_weight", "rotation_weight"}
        method_specific = {
            "pinv": {"k_val"},
            "svd": {"k_val", "min_singular_value"},
            "trans": {"k_val"},
            "dls": {"lambda_val"}
        }

        required_keys = required_common | method_specific[self.ik_method]
        missing_keys = required_keys - set(self.ik_params.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required ik_params for method '{self.ik_method}': {missing_keys}"
            )

        # Validate and normalize ignore_axes
        allowed = {"roll", "pitch", "yaw"}
        if self.ignore_axes is not None:
            normalized = []
            seen = set()
            for a in self.ignore_axes:
                al = str(a).strip().lower()
                if al not in allowed:
                    raise ValueError(f"Invalid axis in ignore_axes: {a}. Must be one of {allowed}")
                if al not in seen:
                    normalized.append(al)
                    seen.add(al)
            self.ignore_axes = normalized
        else:
            self.ignore_axes = []


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

    def __init__(self, cfg: IKControllerConfig, robot_config: RobotConfig, verbose: bool = True):
        self.cfg = cfg
        self.robot_config = robot_config
        self._verbose = verbose

        # Target pose buffers
        self.ee_pos_des = np.zeros(3, dtype=np.float32)
        self.ee_quat_des = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        self._command = np.zeros(self.action_dim, dtype=np.float32)

        # NEW: Velocity limits (default to ~2 degrees per cycle if not specified)
        if cfg.max_joint_velocities is None:
            self.max_joint_velocities = np.full(robot_config.num_joints, 0.035, dtype=np.float32)  # ~2 deg
        else:
            self.max_joint_velocities = np.asarray(cfg.max_joint_velocities, dtype=np.float32)

        # NEW: Joint limits (default to [-π, π] if not specified)
        if cfg.joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi) for _ in range(robot_config.num_joints)]
        else:
            self.joint_limits = cfg.joint_limits

        # NEW: Anti-windup state
        self.prev_error_norm = None
        self.windup_counter = 0

        # Debug/telemetry values (for external logging when enabled)
        self.last_adaptive_lambda: float = 0.0
        self.last_condition_number: float = 0.0

        # NEW: Compute controllable DOF mask for reduced Jacobian
        if cfg.use_reduced_jacobian:
            self.controllable_dofs = self._compute_controllable_dofs()
            self.n_controllable = len(self.controllable_dofs)
        else:
            self.controllable_dofs = None
            self.n_controllable = 6

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
            
            # Check if axis is primarily X, Y, or Z (within 30 degrees)
            if abs(axis_normalized[0]) > 0.866:  # cos(30°) ≈ 0.866
                has_x_rotation = True
            if abs(axis_normalized[1]) > 0.866:
                has_y_rotation = True
            if abs(axis_normalized[2]) > 0.866:
                has_z_rotation = True
        
        # Add controllable rotation DOFs
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
            jacobian_full: Full 6×n Jacobian matrix
            
        Returns:
            Reduced m×n Jacobian where m = number of controllable DOFs
            For excavator: 5×4 (position + pitch + yaw)
        """
        if self.controllable_dofs is None:
            return jacobian_full
        
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
        if self.controllable_dofs is None or axis_angle_error is None:
            if axis_angle_error is None:
                return position_error
            return np.concatenate([position_error, axis_angle_error])
        
        # Build full 6D error
        full_error = np.concatenate([position_error, axis_angle_error])
        
        # Extract only controllable components
        return full_error[self.controllable_dofs]

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
        self.prev_error_norm = None
        self.windup_counter = 0

    def apply_rotation_filter(self, rotation_error: np.ndarray) -> np.ndarray:
        """Filter rotation error by zeroing out ignored axes.

        Args:
            rotation_error: 3D rotation error vector [roll, pitch, yaw] in axis-angle form

        Returns:
            Filtered rotation error with ignored axes zeroed out
        """
        if not self.cfg.ignore_axes:
            return rotation_error

        filtered = rotation_error.copy()
        for axis in self.cfg.ignore_axes:
            if axis == "roll":
                filtered[0] = 0.0
            elif axis == "pitch":
                filtered[1] = 0.0
            elif axis == "yaw":
                filtered[2] = 0.0
        return filtered

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
            joint_angles: Current RELATIVE joint angles in radians (for limit checking)
            joint_quats: Current joint quaternions from IMUs (absolute, with mounting offsets applied)

        Returns:
            Target joint angles (relative)
            
        Note: joint_angles should be computed using compute_relative_joint_angles() from joint_quats
        """

        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        ee_quat = np.asarray(ee_quat, dtype=np.float32)
        joint_angles = np.asarray(joint_angles, dtype=np.float32)

        # Require joint quaternions (no implicit fallback construction from angles)
        if joint_quats is None:
            raise ValueError("joint_quats is required for IK.compute; no fallback from joint_angles")
        joint_quats = np.asarray(joint_quats, dtype=np.float32)
        
        # Compute full Jacobian (uses absolute quaternions after propagation)
        jacobian_full = compute_jacobian(joint_quats, self.robot_config)

        # NEW: Extract reduced Jacobian if enabled
        if self.cfg.use_reduced_jacobian:
            jacobian = self._get_reduced_jacobian(jacobian_full)
        else:
            jacobian = jacobian_full

        # NEW: Compute adaptive damping based on condition number
        adaptive_lambda = self._compute_adaptive_damping(jacobian)

        # Compute pose error and solve IK
        if self.cfg.command_type == "position":
            position_error = self.ee_pos_des - ee_pos
            
            # NEW: Apply weighting to Jacobian rows instead of error vector
            pos_weight = self.cfg.ik_params.get("position_weight")
            
            # Extract position rows (always [0:3] even in reduced Jacobian)
            jacobian_pos = pos_weight * jacobian[0:3, :]
            
            delta_joint_angles = self._compute_delta_joint_angles(
                position_error, jacobian_pos, adaptive_lambda
            )
        else:
            # Directly compare with desired quaternion; ignore configured axes below
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des
            )

            # Apply ignore_axes filtering to orientation error
            axis_angle_error = self.apply_rotation_filter(axis_angle_error)

            # NEW: Get reduced error vector if using reduced Jacobian
            if self.cfg.use_reduced_jacobian:
                pose_error = self._get_reduced_error(position_error, axis_angle_error)
                
                # Apply weighting to reduced Jacobian
                # Position rows are always [0:3], rotation rows depend on controllable_dofs
                pos_weight = self.cfg.ik_params.get("position_weight")
                rot_weight = self.cfg.ik_params.get("rotation_weight")
                
                # Weight each row based on whether it's position or rotation
                weighted_jacobian_rows = []
                for i, dof_idx in enumerate(self.controllable_dofs):
                    if dof_idx < 3:  # Position DOF
                        row = pos_weight * jacobian[i, :]
                    else:  # Rotation DOF
                        row = rot_weight * jacobian[i, :]
                        # Optionally hard-remove ignored rotation DOFs by zeroing their rows
                        if self.cfg.use_ignore_axes_in_jacobian and self.cfg.ignore_axes:
                            if (dof_idx == 3 and "roll" in self.cfg.ignore_axes) or \
                               (dof_idx == 4 and "pitch" in self.cfg.ignore_axes) or \
                               (dof_idx == 5 and "yaw" in self.cfg.ignore_axes):
                                row = np.zeros_like(row)
                    weighted_jacobian_rows.append(row)
                
                jacobian_weighted = np.vstack(weighted_jacobian_rows).astype(np.float32)
            else:
                # Full Jacobian case
                pos_weight = self.cfg.ik_params.get("position_weight")
                rot_weight = self.cfg.ik_params.get("rotation_weight")

                jacobian_weighted = np.vstack([
                    pos_weight * jacobian[0:3, :],
                    rot_weight * jacobian[3:6, :]
                ]).astype(np.float32)

                # Optionally hard-remove ignored rotation DOFs by zeroing their rows
                if self.cfg.use_ignore_axes_in_jacobian and self.cfg.ignore_axes:
                    if "roll" in self.cfg.ignore_axes:
                        jacobian_weighted[3, :] = 0.0
                    if "pitch" in self.cfg.ignore_axes:
                        jacobian_weighted[4, :] = 0.0
                    if "yaw" in self.cfg.ignore_axes:
                        jacobian_weighted[5, :] = 0.0

                pose_error = np.concatenate([position_error, axis_angle_error])

            delta_joint_angles = self._compute_delta_joint_angles(
                pose_error, jacobian_weighted, adaptive_lambda
            )

        # DEBUG: Log raw IK output before any limiting
        if hasattr(self, '_ik_raw_debug_counter'):
            self._ik_raw_debug_counter += 1
        else:
            self._ik_raw_debug_counter = 0

        if self._verbose and self._ik_raw_debug_counter % 50 == 0:
            raw_delta_deg = np.degrees(delta_joint_angles)
            raw_max = np.max(np.abs(raw_delta_deg))
            if raw_max > 0.01:
                print(f"[IK-RAW] delta_deg={raw_delta_deg} (max={raw_max:.3f}°)")

        # NEW: Apply velocity limiting (if enabled)
        if self.cfg.enable_velocity_limiting:
            delta_joint_angles = self._apply_velocity_limits(delta_joint_angles)

        # NEW: Add joint limit avoidance (uses RELATIVE angles)
        if self.cfg.enable_joint_limit_avoidance:
            delta_joint_angles = self._add_joint_limit_avoidance(
                delta_joint_angles, joint_angles
            )

        # NEW: Anti-windup for unreachable targets
        if self.cfg.enable_anti_windup:
            delta_joint_angles = self._apply_anti_windup(
                delta_joint_angles, position_error, axis_angle_error if self.cfg.command_type == "pose" else None
            )

        return joint_angles + delta_joint_angles

    def _compute_adaptive_damping(self, jacobian: np.ndarray) -> float:
        """Compute adaptive damping factor based on Jacobian conditioning.
        
        Args:
            jacobian: Current Jacobian matrix
            
        Returns:
            Adaptive damping value (lambda)
        """
        if not self.cfg.enable_adaptive_damping:
            lam = self.cfg.ik_params.get("lambda_val", 0.01)
            self.last_adaptive_lambda = float(lam)
            return lam

        # Compute singular values for condition number (no fallback)
        _, S, _ = np.linalg.svd(jacobian.astype(np.float64))
        cond = S[0] / (S[-1] + 1e-12)
        self.last_condition_number = float(cond)

        # Base damping and adaptive scaling
        base_lambda = self.cfg.ik_params.get("lambda_val", 0.01)
        adaptive_lambda = base_lambda * (1.0 + 0.5 * np.log(1.0 + cond))
        lam = float(np.clip(adaptive_lambda, base_lambda, base_lambda * 10.0))
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
            joint_angles: Current RELATIVE joint angles (computed from IMU quaternions)
            
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
                distance_ratio = (joint_angles[i] - q_min) / margin
                # Quadratic repulsion: stronger closer to limit
                repulsion = repulsion_strength * (1.0 - distance_ratio) ** 2
                modified_delta[i] += repulsion
            
            # Repulsion from upper limit
            elif joint_angles[i] > q_max - margin:
                distance_ratio = (q_max - joint_angles[i]) / margin
                # Quadratic repulsion: stronger closer to limit
                repulsion = repulsion_strength * (1.0 - distance_ratio) ** 2
                modified_delta[i] -= repulsion
        
        return modified_delta

    def _apply_anti_windup(
        self,
        delta_joint_angles: np.ndarray,
        position_error: np.ndarray,
        rotation_error: Optional[np.ndarray]
    ) -> np.ndarray:
        """Prevent error accumulation for unreachable targets.

        Args:
            delta_joint_angles: Proposed joint changes
            position_error: Current position error
            rotation_error: Current rotation error (if applicable)

        Returns:
            Modified joint changes with anti-windup
        """
        # Compute total error norm
        if rotation_error is not None:
            current_error_norm = np.linalg.norm(position_error) + np.linalg.norm(rotation_error)
        else:
            current_error_norm = np.linalg.norm(position_error)

        # Check if error is decreasing
        if self.prev_error_norm is not None:
            if current_error_norm > 0.95 * self.prev_error_norm:
                # Error not decreasing - target may be unreachable
                self.windup_counter += 1

                if self.windup_counter > 5:
                    # Scale down commands to prevent oscillation
                    scale = 0.5 ** (self.windup_counter - 5)
                    scale = max(scale, 0.1)  # Don't go below 10%
                    delta_joint_angles *= scale
                    # DEBUG: Print anti-windup activation
                    if self._verbose and self.windup_counter % 50 == 6:  # Print occasionally
                        print(f"[ANTI-WINDUP] counter={self.windup_counter} scale={scale:.3f} err_norm={current_error_norm:.4f}")
            else:
                # Error is decreasing - reset counter
                self.windup_counter = max(0, self.windup_counter - 1)

        self.prev_error_norm = current_error_norm
        return delta_joint_angles

    def _compute_delta_joint_angles(
        self, delta_pose: np.ndarray, jacobian: np.ndarray, adaptive_lambda: float
    ) -> np.ndarray:
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
            # Use adaptive lambda for better conditioning
            return ik_method_damped_least_squares(
                jacobian, delta_pose, np.float32(adaptive_lambda), w_mat
            )
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


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath for 20-30% speedup
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


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
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


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
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
@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
def ik_method_pinv(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Pseudo-inverse IK method."""
    jacobian_pinv = np.linalg.pinv(np.asarray(jacobian, dtype=np.float32))
    return k_val * np.dot(jacobian_pinv, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
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


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
def ik_method_transpose(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32) -> np.ndarray:
    """Jacobian transpose IK method."""
    return k_val * np.dot(np.asarray(jacobian, dtype=np.float32).T, np.asarray(delta_pose, dtype=np.float32))


@numba.njit(fastmath=True)  # CHANGED: Enabled fastmath
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

    The imu_offsets represent the physical mounting angle of each IMU relative to the joint frame.
    If the IMU reading is q_imu = q_joint * q_offset, removing the mounting offset requires
    right-multiplying by the inverse: q_corrected = q_imu * q_offset_inverse

    This undoes the physical mounting rotation to get the true joint orientation.

    Returns corrected quaternions preserving all rotation information.
    """
    imu_quats = np.asarray(imu_quats, dtype=np.float32)
    corrected_quats = np.zeros_like(imu_quats, dtype=np.float32)

    for i in range(len(imu_quats)):
        # Normalize raw IMU quaternion
        normalized_imu_quat = quat_normalize(imu_quats[i])

        # Apply mounting offset correction by removing the mounting rotation
        # q_corrected = q_imu * q_offset_inverse (undoes mounting offset)
        offset_quat_inv = quat_conjugate(robot_config.imu_offsets[i])
        corrected_quat = quat_normalize(
            quat_multiply(normalized_imu_quat, offset_quat_inv)
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

    This function projects downstream IMU quaternions to their rotation axes only
    (removing roll/yaw drift), then composes with the slew rotation to get
    world-frame orientations.

    Args:
        quats: Input quaternions [base, link1, link2, ...] where base rotation is
               measured by encoder and downstream orientations are from IMUs
        robot_config: Robot configuration containing rotation axes for each joint

    Returns:
        Quaternions with base rotation propagated to all downstream joints
    """
    quats = np.asarray(quats, dtype=np.float32)
    propagated_quats = quats.copy()

    # Base stays as-is (slew from encoder, already a clean Z-rotation)
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


def compute_relative_joint_angles(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Compute relative joint angles from absolute IMU quaternions.
    
    For an excavator with absolute IMU orientations, joint limits are defined
    relative to the parent link. This function extracts the relative rotation
    about each joint's axis.
    
    Example for excavator:
    - Joint 0 (slew): Absolute yaw angle around Z-axis
    - Joint 1 (boom): Relative pitch from world horizontal (parent is slew, which only rotates in Z)
    - Joint 2 (arm): Relative pitch from boom orientation
    - Joint 3 (bucket): Relative pitch from arm orientation
    
    Args:
        quats: Absolute joint quaternions from IMUs (after offset correction)
        robot_config: Robot configuration with rotation axes
        
    Returns:
        np.ndarray: Relative joint angles in radians [n_joints]
    """
    quats = np.asarray(quats, dtype=np.float32)
    corrected_quats = apply_imu_offsets(quats, robot_config)
    
    n_joints = len(corrected_quats)
    relative_angles = np.zeros(n_joints, dtype=np.float32)
    
    # Joint 0 (slew): Extract absolute rotation about Z-axis
    relative_angles[0] = extract_axis_rotation(
        corrected_quats[0], 
        robot_config.rotation_axes[0]
    )
    
    # Joints 1+ : Extract relative rotation from parent link
    for i in range(1, n_joints):
        # Get parent orientation (world frame)
        parent_quat = corrected_quats[i-1]
        current_quat = corrected_quats[i]
        
        # Compute relative orientation: q_rel = q_parent^-1 * q_current
        parent_quat_inv = quat_conjugate(parent_quat)
        relative_quat = quat_normalize(quat_multiply(parent_quat_inv, current_quat))
        
        # Extract rotation about this joint's axis (in parent's local frame)
        # The axis needs to be in parent frame, but for serial chain with consistent
        # Y-axis rotations, we can use the local axis directly
        relative_angles[i] = extract_axis_rotation(
            relative_quat,
            robot_config.rotation_axes[i]
        )
    
    return relative_angles




def compute_jacobian(quats: np.ndarray, robot_config: RobotConfig):
    """Jacobian computation wrapper using RobotConfig.

    Pipeline (FULL input):
      1) Apply IMU mounting offsets
      2) Propagate base (slew) rotation to downstream joints
    """
    quats = np.asarray(quats, dtype=np.float32)
    corrected_quats = apply_imu_offsets(quats, robot_config)
    propagated_quats = propagate_base_rotation(corrected_quats, robot_config)
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
    propagated_quats = propagate_base_rotation(corrected_quats, robot_config)
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
    propagated_quats = propagate_base_rotation(corrected_quats, robot_config)

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
    - Slew link: Physical offset from slew rotation axis to boom mounting point (x+16.5mm, z+64.5mm)
                 This offset rotates with the slew joint, making boom mount orbit around Z-axis
    - Boom/Arm/Bucket: Serial links extending in local X direction
    - End-effector: Tool tip 142mm below bucket mounting center

    The slew joint models the mounting offset as a rotating link. When the slew rotates,
    the boom mounting point orbits around the slew axis, which is the correct physical behavior.
    The IK solver uses rotation axes (not link directions) so this causes no issues.

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

    # IMU mounting offsets (in degrees around Y-axis)
    # These correct for physical IMU mounting angles relative to joint axes
    # Pico sends RAW quaternions - Python handles all mounting corrections
    imu_offsets = [
        create_imu_offset_quat(0.0),     # Slew joint - no IMU offset needed for encoder
        create_imu_offset_quat(+13.85),  # Boom IMU/joint
        create_imu_offset_quat(-0.61),   # Arm IMU/joint
        create_imu_offset_quat(0.0)      # Bucket IMU/joint - no offset
    ]

    return RobotConfig(
        link_lengths=[slew_length, float(boom_length), float(arm_length), float(bucket_length)],
        link_directions=[
            slew_direction,  # Slew: from axis to boom mount (radial + vertical offset)
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
        ee_offset=np.array([0.0, 0.0, -0.142], dtype=np.float32),  # Tool tip below bucket center
        origin_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32)   # Slew axis is world origin
    )


# ============================================================================
# CONVENIENCE WRAPPER - Single Function for Complete IK Pipeline
# ============================================================================

def ik_step_excavator(
    ik_controller: IKController,
    robot_config: RobotConfig,
    raw_imu_quats: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete IK step for excavator - handles all the conversions automatically.
    
    This is a convenience function that handles:
    1. Computing relative angles from absolute IMU quaternions
    2. Getting current end-effector pose
    3. Setting the target
    4. Computing new target angles
    
    Args:
        ik_controller: Configured IK controller instance
        robot_config: Robot configuration
        raw_imu_quats: Raw IMU quaternions [4×4] from hardware (absolute orientations)
        target_pos: Desired end-effector position [x, y, z]
        target_quat: Desired end-effector orientation [w, x, y, z]
        
    Returns:
        Tuple of:
        - target_relative_angles: Target joint angles (relative) to send to motors
        - current_ee_pos: Current end-effector position (for monitoring)
        - relative_angles: Current relative joint angles (for monitoring)
    
    Example:
        >>> target_angles, ee_pos, current_angles = ik_step_excavator(
        ...     ik_controller, robot_config, raw_imus, 
        ...     target_pos=[0.6, 0.0, 0.0],
        ...     target_quat=[1, 0, 0, 0]
        ... )
        >>> send_to_motors(target_angles)
    """
    # Step 1: Compute relative angles for limit checking
    relative_angles = compute_relative_joint_angles(raw_imu_quats, robot_config)
    
    # Step 2: Get current end-effector pose
    ee_pos, ee_quat = get_pose(raw_imu_quats, robot_config)
    
    # Step 3: Set target command
    target_command = np.concatenate([
        np.asarray(target_pos, dtype=np.float32),
        np.asarray(target_quat, dtype=np.float32)
    ])
    ik_controller.set_command(target_command, ee_pos, ee_quat)
    
    # Step 4: Compute new target angles
    target_relative_angles = ik_controller.compute(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        joint_angles=relative_angles,
        joint_quats=raw_imu_quats
    )
    
    return target_relative_angles, ee_pos, relative_angles

