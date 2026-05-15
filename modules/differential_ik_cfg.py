"""
Configuration dataclasses and robot model loading for differential IK.

Contains:
- IKControllerConfig: Task-space IK controller settings
- RobotConfig: Kinematic chain definition
- CheckDOF: Jacobian-based DOF analysis (experimental)
- load_excavator_robot_config(): Build excavator model from YAML
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Dict, Optional, List, Tuple, Any
import yaml


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

    ik_params: Optional[Dict[str, Any]] = None
    """Method-specific parameters (k_val, lambda_val, min_singular_value).

    Optional: ``joint_weights`` (list of per-joint weights). Higher weight = more
    movement for that joint. Uses sqrt(w) internally for linear effective weighting."""

    enable_velocity_limiting: bool = True
    """Enable joint velocity limiting (safety feature)."""

    max_joint_velocities: Optional[List[float]] = None
    """Maximum joint velocity per control cycle (radians). If None, uses [0.035, 0.035, 0.035, 0.035] (~2 deg)."""

    joint_limits: Optional[List[Tuple[float, float]]] = None
    """Joint limits as [(min, max), ...] in radians. If None, uses [-pi, pi] for all."""

    enable_adaptive_damping: bool = True
    """Enable adaptive damping based on Jacobian conditioning."""

    adaptive_damping_max_multiplier: float = 2.0
    """Peak lambda = base_lambda * max_multiplier, reached at condition_number_threshold.
    Tune this to control how much extra damping is applied near singularities.
    NOTE: 2.0 gives lambda 0.001 → 0.002 — likely needs increasing for real hardware."""

    condition_number_threshold: float = 40.0
    """Condition number at which adaptive damping peaks and IK commands are gated (0 = no gating).
    Adaptive scaling is derived so lambda reaches base * max_multiplier exactly at this value."""

    enable_joint_limit_avoidance: bool = True
    """Enable repulsion forces near joint limits."""

    enable_jacobian_metrics: bool = True
    """Compute and store Yoshikawa manipulability index and singular values each step.
    Condition number is always computed (needed for adaptive damping and gating).
    Disable for high-rate or minimal-overhead use cases where full diagnostics are not needed."""

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
                "  - dls: lambda_val"
            )

        # Validate required parameters based on method
        required_common = set()
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


@dataclass
class RobotConfig:
    """Robot kinematic configuration."""

    link_lengths: List[float]
    """Length of each link."""

    link_directions: Optional[List[np.ndarray]] = None
    """Local direction vector for each link (default: x-axis)."""

    rotation_axes: Optional[List[np.ndarray]] = None
    """Rotation axis for each joint (default: z for first, y for others)."""

    ee_offset: Optional[np.ndarray] = None
    """End-effector position offset from last joint in local frame [x, y, z]."""

    origin_offset: Optional[np.ndarray] = None
    """Origin offset - position of first joint relative to world origin [x, y, z]."""

    imu_chain: Optional[List[Dict[str, Any]]] = None
    """Configured IMU-to-joint extraction chain."""

    imu_mapping: Optional[Dict[str, int]] = None
    """Logical IMU role -> physical sensor index mapping."""

    imu_sensor_roles: Optional[List[str]] = None
    """Configured IMU role order expected by canonical extraction."""

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

        # Validate dimensions
        if len(self.link_directions) != self.num_links:
            raise ValueError(f"Expected {self.num_links} link directions")
        if len(self.rotation_axes) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} rotation axes")

        # Set default ee_offset if not provided
        if self.ee_offset is None:
            self.ee_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Set default origin_offset if not provided
        if self.origin_offset is None:
            self.origin_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.imu_chain is None:
            self.imu_chain = [
                {
                    'joint': 'slew',
                    'output_index': 0,
                    'source': 'all',
                    'axis': 'z',
                    'extraction': 'average_z_yaw',
                },
                {
                    'joint': 'lift',
                    'role': 'boom',
                    'parent_role': 'base',
                    'output_index': 1,
                    'axis': 'y',
                    'extraction': 'gravity_pitch_delta',
                },
                {
                    'joint': 'arm',
                    'role': 'arm',
                    'parent_role': 'boom',
                    'output_index': 2,
                    'axis': 'y',
                    'extraction': 'gravity_pitch_delta',
                },
                {
                    'joint': 'bucket',
                    'role': 'bucket',
                    'parent_role': 'arm',
                    'output_index': 3,
                    'axis': 'y',
                    'extraction': 'gravity_pitch_delta',
                },
            ]
        if self.imu_mapping is None:
            self.imu_mapping = {'base': 0, 'boom': 1, 'arm': 2, 'bucket': 3}

        sensor_roles: List[str] = []

        def add_sensor_role(role):
            if role and role != 'all' and role in self.imu_mapping and role not in sensor_roles:
                sensor_roles.append(role)

        for item in self.imu_chain:
            if not isinstance(item, dict):
                continue
            add_sensor_role(item.get('parent_role'))
            add_sensor_role(item.get('role'))
        self.imu_sensor_roles = sensor_roles if sensor_roles else ['base', 'boom', 'arm', 'bucket']

        # ENSURE ALL ARRAYS ARE FLOAT32
        self.link_lengths = np.asarray(self.link_lengths, dtype=np.float32)  # Convert list to numpy array!
        self.link_directions = np.array([np.asarray(arr, dtype=np.float32) for arr in self.link_directions])
        self.rotation_axes = np.array([np.asarray(arr, dtype=np.float32) for arr in self.rotation_axes])
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
        # Lazy import to avoid circular dependency (compute_jacobian is in differential_ik.py)
        from .differential_ik import compute_jacobian

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
        U, S, Vt = np.linalg.svd(J_pos, full_matrices=True)
        r = int(np.sum(S > self.svd_tol))
        N = Vt[r:, :].T if r < Vt.shape[0] else np.zeros((J_pos.shape[1], 0))

        rot_independent = [False, False, False]

        if N.shape[1] > 0:
            for k in range(3):  # 0=roll, 1=pitch, 2=yaw
                if not rot_present[k]:
                    continue

                r_k = J_rot[k, :]   # 1 x n
                r_k_null = r_k @ N

                if np.linalg.norm(r_k_null) > self.null_rot_tol:
                    rot_independent[k] = True

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


def _load_control_config(path: str = "configuration_files/control_config.yaml") -> Dict[str, Any]:
    """Load control configuration YAML file."""
    try:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent.parent / path
        if not p.exists():
            return {}
        with p.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _require_xyz_vector(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of 3 numbers")
    return np.asarray(value, dtype=np.float32)


def load_excavator_robot_config(path: str = "configuration_files/control_config.yaml") -> RobotConfig:
    """Build the excavator kinematic model from control_config.yaml."""
    cfg = _load_control_config(path)
    robot_cfg = cfg.get('robot', {}) if isinstance(cfg, dict) else {}
    if not isinstance(robot_cfg, dict) or not robot_cfg:
        raise RuntimeError("Missing 'robot' section in control_config.yaml")

    frame_cfg = robot_cfg.get('frame', {})
    links_cfg = robot_cfg.get('links', {})
    tools_cfg = robot_cfg.get('tools', {})
    if not isinstance(frame_cfg, dict) or not isinstance(links_cfg, dict) or not isinstance(tools_cfg, dict):
        raise RuntimeError("robot.frame, robot.links, and robot.tools are required in control_config.yaml")

    try:
        origin_height_m = float(frame_cfg['origin_height_m'])
        boom_mount_offset = _require_xyz_vector(links_cfg['boom_mount_offset_m'], "robot.links.boom_mount_offset_m")
        boom_length_m = float(links_cfg['boom_length_m'])
        arm_length_m = float(links_cfg['arm_length_m'])
        coupler_length_m = float(links_cfg['coupler_length_m'])
        active_tool = tools_cfg['active']
        presets = tools_cfg['presets']
    except KeyError as e:
        raise RuntimeError(f"Missing required robot configuration in control_config.yaml: {e}") from e

    if not isinstance(active_tool, str) or not active_tool:
        raise RuntimeError("robot.tools.active must be a non-empty string")
    if not isinstance(presets, dict):
        raise RuntimeError("robot.tools.presets must be a dict")
    if active_tool not in presets or not isinstance(presets[active_tool], dict):
        raise RuntimeError(f"robot.tools.active '{active_tool}' not found in robot.tools.presets")
    tool_tip_offset = _require_xyz_vector(
        presets[active_tool].get('tip_offset_m'),
        f"robot.tools.presets.{active_tool}.tip_offset_m",
    )

    slew_length = float(np.linalg.norm(boom_mount_offset))
    if slew_length <= 1e-9:
        raise RuntimeError("robot.links.boom_mount_offset_m must not be zero")
    slew_direction = boom_mount_offset / slew_length

    imu_cfg = cfg.get('imu', {}) if isinstance(cfg, dict) else {}
    imu_chain = imu_cfg.get('chain') if isinstance(imu_cfg, dict) else None
    imu_mapping = imu_cfg.get('imu_mapping') if isinstance(imu_cfg, dict) else None

    return RobotConfig(
        link_lengths=[slew_length, boom_length_m, arm_length_m, coupler_length_m],
        link_directions=[
            slew_direction,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ],
        rotation_axes=[
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ],
        ee_offset=tool_tip_offset,
        origin_offset=np.array([0.0, 0.0, origin_height_m], dtype=np.float32),
        imu_chain=imu_chain,
        imu_mapping=imu_mapping,
    )


def create_excavator_config() -> RobotConfig:
    """Compatibility wrapper for the config-backed excavator model."""
    return load_excavator_robot_config()
