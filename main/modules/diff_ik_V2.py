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
import logging
from dataclasses import dataclass
from typing import Literal, Dict, Optional, List, Tuple, Any

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


@dataclass
class IKControllerConfig:
    """Configuration for differential inverse kinematics controller."""

    command_type: Literal["position", "pose"]
    """Type of task-space command: 'position' (3D) or 'pose' (6D)."""

    ik_method: Literal["pinv", "svd", "trans", "dls"]
    """Method for computing Jacobian inverse."""

    use_rotational_velocity: bool
    """Whether to accept rotational velocity components in velocity_mode."""

    use_relative_mode: bool = False
    """Whether commands are relative to current pose."""

    velocity_mode: bool = False
    """When True, interpret commands as desired EE velocities and integrate joint rates."""

    velocity_error_gain: float = 1.0
    """Proportional gain applied to pose error when in velocity_mode."""

    ik_params: Optional[Dict[str, float]] = None
    """Method-specific parameters."""

    ignore_axes: Optional[List[str]] = None
    """Axes to ignore in orientation error when solving IK.
    Any of ["roll", "pitch", "yaw"]. Example for excavator: ["roll", "yaw"]."""

    # When True, also zero the corresponding rotation rows in the weighted Jacobian
    # after applying position/rotation weights (hard removal of ignored rotational DOFs).
    # This provides stronger decoupling and stability when an orientation axis is intentionally ignored.
    use_ignore_axes_in_jacobian: bool = True

    enable_frame_transform: bool = True
    """Transform target orientation to robot's local frame when using ignore_axes.
    Ensures pitch commands are relative to robot heading, not global frame.
    Critical for slew-based systems to prevent wonky behavior at large yaw angles."""

    enable_velocity_limiting: bool = True
    """Enable joint velocity limiting (safety feature)."""

    max_joint_velocities: Optional[List[float]] = None
    """Maximum joint velocity per control cycle (radians). If None, uses [0.035, 0.035, 0.035, 0.035] (≈2°)."""

    joint_limits: Optional[List[Tuple[float, float]]] = None
    """Joint limits as [(min, max), ...] in radians. If None, uses [-π, π] for all."""

    enable_adaptive_damping: bool = True
    """Enable adaptive damping based on Jacobian conditioning."""

    adaptive_damping_scaling: float = 0.5
    """Scaling factor for adaptive damping: lambda = base * (1 + scaling * log(1 + cond))."""

    adaptive_damping_max_multiplier: float = 10.0
    """Maximum multiplier for adaptive damping (lambda_max = base * max_multiplier)."""

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


class CheckDOF:
    # TODO: not used yet on anything meaningful!
    # used in the future to automate "ignore_axes" setup
    """
    Jacobian-based DOF checker with nullspace analysis.

    - Position DOFs: inferred from J_pos row norms and rank.
    - Orientation DOFs: inferred from J_rot, but counted as "independent"
      only if they can be excited while keeping position fixed
      (i.e. non-zero component in the nullspace of J_pos).

    This correctly identifies when orientation DOFs are coupled with position
    (e.g., without rotating tool head the excavator slew affects both position and yaw simultaneously).
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        row_tol: float = 1e-4,
        svd_tol: float = 1e-5,
        null_rot_tol: float = 1e-4,
    ):
        """
        Args:
            robot_config: RobotConfig instance.
            row_tol: minimum row norm to consider a DOF "present".
            svd_tol: tolerance for rank/nullspace from SVD.
            null_rot_tol: minimum magnitude of rotation in the position
                nullspace to call an orientation DOF "independent".
        """
        self.robot_config = robot_config
        self.row_tol = row_tol
        self.svd_tol = svd_tol
        self.null_rot_tol = null_rot_tol

    def analyze(self, quats: np.ndarray) -> Dict[str, Any]:
        """
        Analyze controllable DOFs at a given joint configuration.

        Args:
            quats: joint orientations [num_joints x 4], wxyz.

        Returns:
            dict with:
                - "pos_present": list[bool] for [x, y, z] - which position DOFs exist
                - "rot_present": list[bool] for [roll, pitch, yaw] - which rotation DOFs exist
                - "rot_independent": list[bool] for [roll, pitch, yaw] - which can be controlled
                                     independently without affecting position
                - "position_rank": int - rank of J_pos (max independent position DOFs)
                - "independent_orientation_dofs": int - number of independently controllable orientations
                - "total_independent_dofs": int - position_rank + independent_orientation_dofs
        """
        # 6 x n Jacobian
        J = compute_jacobian(quats, self.robot_config)
        J = np.asarray(J, dtype=np.float64)

        if J.shape[0] != 6:
            raise ValueError(f"Expected 6xN Jacobian, got {J.shape}")

        J_pos = J[0:3, :]  # x, y, z
        J_rot = J[3:6, :]  # roll, pitch, yaw

        # --- Presence based on row norms ---
        pos_present = [np.linalg.norm(J_pos[i]) > self.row_tol for i in range(3)]
        rot_present = [np.linalg.norm(J_rot[i]) > self.row_tol for i in range(3)]

        # --- Position rank (how many independent positional DOFs) ---
        position_rank = int(np.linalg.matrix_rank(J_pos, tol=self.svd_tol))

        # --- Nullspace of J_pos: motions that keep EE position fixed ---
        # J_pos: 3 x n
        # SVD: J_pos = U S V^T
        # Nullspace basis = columns of V corresponding to zero singular values.
        U, S, Vt = np.linalg.svd(J_pos, full_matrices=True)
        # number of non-zero singular values
        r = int(np.sum(S > self.svd_tol))
        # Vt is (n x n); rows [r:] correspond to nullspace basis vectors
        # Nullspace N: n x (n-r)
        N = Vt[r:, :].T if r < Vt.shape[0] else np.zeros((J_pos.shape[1], 0))

        rot_independent = [False, False, False]

        if N.shape[1] > 0:
            # For each rotational axis, check its component in null(J_pos)
            for k in range(3):  # 0=roll, 1=pitch, 2=yaw
                if not rot_present[k]:
                    continue

                r_k = J_rot[k, :]   # 1 x n
                # Restrict to position-nullspace
                # r_k_null = r_k @ N  -> 1 x (n-r)
                r_k_null = r_k @ N

                if np.linalg.norm(r_k_null) > self.null_rot_tol:
                    rot_independent[k] = True
        # If N has zero columns, there is no way to move without affecting position,
        # so no independent orientation DOF (even if orientation is "present").

        independent_orientation_dofs = int(sum(rot_independent))
        total_independent_dofs = position_rank + independent_orientation_dofs

        return {
            "pos_present": pos_present,
            "rot_present": rot_present,
            "rot_independent": rot_independent,
            "position_rank": position_rank,
            "independent_orientation_dofs": independent_orientation_dofs,
            "total_independent_dofs": total_independent_dofs,
        }


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
        # If verbose flag is explicitly False, use WARNING level, otherwise use provided log_level
        if not verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
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
            if self.cfg.use_reduced_jacobian and self.controllable_dofs is not None and required_size != 6:
                desired_vec = full_vec[self.controllable_dofs][:required_size]
            else:
                desired_vec = full_vec[:required_size]

        # Map full 6D inputs into the reduced task space (drops uncontrollable axes)
        if self.cfg.use_reduced_jacobian and self.controllable_dofs is not None and required_size != 6:
            if desired_vec.size == 6:
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
        joint_quats: np.ndarray,
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
            joint_quats: Current joint quaternions from IMUs (absolute, with mounting offsets applied)
            desired_ee_velocity: Optional desired EE twist (position 3D or pose 6D) for velocity mode
            dt: Optional timestep (seconds). Defaults to self.default_dt.
            current_joint_velocities: Optional measured joint velocities (rad/s) for future adaptive use

        Returns:
            Target joint angles (relative)
            
        Note: joint_angles should be computed using compute_relative_joint_angles() from joint_quats
        """

        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        ee_quat = np.asarray(ee_quat, dtype=np.float32)
        joint_angles = np.asarray(joint_angles, dtype=np.float32)
        dt_val = self.default_dt if dt is None else float(dt)
        dt_val = float(np.clip(dt_val, 1e-4, 1.0))

        # Require joint quaternions (no implicit fallback construction from angles)
        if joint_quats is None:
            raise ValueError("joint_quats is required for IK.compute; no fallback from joint_angles")
        joint_quats = np.asarray(joint_quats, dtype=np.float32)
        
        # Compute full Jacobian (uses absolute quaternions after propagation)
        jacobian_full = compute_jacobian(joint_quats, self.robot_config)

        # Optional frame transform: express everything in base (cab) frame to
        # avoid world-frame coupling between slew yaw and pitch commands.
        current_pos = ee_pos
        target_pos = self.ee_pos_des
        if self.cfg.enable_frame_transform:
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

        # NEW: Extract reduced Jacobian if enabled
        if self.cfg.use_reduced_jacobian:
            jacobian = self._get_reduced_jacobian(jacobian_full)
        else:
            jacobian = jacobian_full

        # NEW: Compute adaptive damping based on condition number
        adaptive_lambda = self._compute_adaptive_damping(jacobian)

        # Initialize error containers for downstream logging/anti-windup
        position_error = np.zeros(3, dtype=np.float32)
        axis_angle_error = np.zeros(3, dtype=np.float32)

        # Compute pose error and solve IK
        if self.cfg.command_type == "position":
            position_error = target_pos - current_pos

            # NEW: Apply weighting to Jacobian rows instead of error vector
            pos_weight = self.cfg.ik_params.get("position_weight")

            # Extract position rows (always [0:3] even in reduced Jacobian)
            jacobian_weighted = pos_weight * jacobian[0:3, :]
            reduced_error = position_error
        else:
            # Transform both current and target orientations to robot's local frame if enabled
            # This makes pitch/roll errors relative to robot heading, not global frame
            # Critical for avoiding pitch->roll coupling at large slew angles
            if self.cfg.enable_frame_transform:
                ee_quat_local = self._transform_to_robot_local_frame(ee_quat, joint_quats)
                target_quat_local = self._transform_to_robot_local_frame(self.ee_quat_des, joint_quats)
            else:
                ee_quat_local = ee_quat
                target_quat_local = self.ee_quat_des

            # Debug: show frame transformation
            if self.logger.level <= logging.DEBUG:
                try:
                    slew_angle = extract_axis_rotation(joint_quats[0], self.robot_config.rotation_axes[0])
                    slew_deg = float(np.degrees(slew_angle))
                    self.logger.debug(
                        "[IK dbg] slew_deg=%.2f | target_global=%s target_local=%s | ee_local=%s",
                        slew_deg,
                        np.round(self.ee_quat_des, 4),
                        np.round(target_quat_local, 4),
                        np.round(ee_quat_local, 4),
                    )
                except Exception:
                    pass

            # Compute pose error in local frame
            position_error, axis_angle_error = compute_pose_error(
                current_pos, ee_quat_local, target_pos, target_quat_local
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
                reduced_error = pose_error
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

                reduced_error = np.concatenate([position_error, axis_angle_error])

        # Choose task vector based on mode (position vs velocity)
        if self.velocity_mode:
            desired_vec = self._prepare_desired_velocity(desired_ee_velocity, reduced_error.size)
            task_vec = desired_vec + (self.velocity_error_gain * reduced_error)
            joint_rate_cmd = self._compute_delta_joint_angles(task_vec, jacobian_weighted, adaptive_lambda)
            delta_joint_angles = joint_rate_cmd * dt_val
        else:
            delta_joint_angles = self._compute_delta_joint_angles(
                reduced_error, jacobian_weighted, adaptive_lambda
            )

        # DEBUG: Log raw IK output before any limiting
        self._ik_raw_debug_counter += 1
        if self.logger.level <= logging.DEBUG and self._ik_raw_debug_counter % 50 == 0:
            raw_delta_deg = np.degrees(delta_joint_angles)
            raw_max = np.max(np.abs(raw_delta_deg))
            if raw_max > 0.01:
                self.logger.debug(f"IK raw delta: {raw_delta_deg} (max={raw_max:.3f}°)")

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
        adaptive_lambda = base_lambda * (1.0 + self.cfg.adaptive_damping_scaling * np.log(1.0 + cond))
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
                    # DEBUG: Log anti-windup activation
                    if self.logger.level <= logging.DEBUG and self.windup_counter % 50 == 6:  # Log occasionally
                        self.logger.debug(f"Anti-windup active: counter={self.windup_counter} scale={scale:.3f} err_norm={current_error_norm:.4f}")
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
def ik_method_svd(jacobian: np.ndarray, delta_pose: np.ndarray, k_val: np.float32, min_singular_value: np.float32, w_inv: np.ndarray) -> np.ndarray:
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
    
    For a robot with absolute IMU orientations, joint limits are defined
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
        # Note: Using print here as this is a module-level function called at import
        # before any logger instances are created
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
        create_imu_offset_quat(+0.61),   # -0.61 Arm IMU/joint
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
