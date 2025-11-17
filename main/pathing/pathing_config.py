"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

from typing import Tuple, List, Union, Dict, Literal
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Environment setup configuration for both sim and real systems."""

    # TODO: irl / sim Y pos flipped!

    # Point A configuration
    point_a_pos: Tuple[float, float, float] = (0.60, 0.15, -0.1)
    point_a_rotation_deg: float = 0.0 # y axis rotation. 0 = horizontal
    
    # Point B configuration  
    point_b_pos: Tuple[float, float, float] = (0.45, -0.15, -0.1)
    point_b_rotation_deg: float = 0.0
    
    # Single wall configuration (matches real hardware format)
    wall_size: Tuple[float, float, float] = (0.08, 0.500, 0.30)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (1.55, 0.0, -0.15)     # Wall center position
    wall_rot: Tuple[float, float, float, float] = (0.9238795, 0.0, 0.0, 0.3826834)  # Quaternion rotation


@dataclass
class PathExecutionConfig:
    """Unified configuration for path execution in both sim and real systems."""

    # Motion parameters ------------------------------
    speed_mps: float = 0.020  # Target constant speed for standardized execution (m/s)
    dt: float = 0.20          # Execution sample period for standardized paths (s)
    max_points: int = 30      # Max control waypoints after standardization (for UI/coarse control)
    smoothing: Dict[str, float] | None = None  # e.g., {"window": 3, "strength": 1.0}

    update_frequency: float = 100.0  # Hz - target loop frequency.

    # Acceleration/deceleration limits for trapezoid velocity profile
    accel_mps2: float = 0.0#0.04  # Acceleration in m/s^2
    decel_mps2: float = 0.0#0.04  # Deceleration in m/s^2

    # Path planning general parameters (apply to all algorithms)
    grid_resolution: float = 0.020  # Grid cell size used for A* (and bounds for others)
    safety_margin: float = 0.06     # Obstacle safety margin in meters.

    # Algorithm dimensionality
    # Use full 3D planning vs X-Z plane only
    use_3d: bool = True



    # Inverse-kinematics controller
    ik_command_type: Literal["position", "pose"] = "pose"
    ik_use_relative_mode: bool = False
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "svd" # dls 0.08 good
    
    # Method/weighting parameters (values are passed through to the IK implementation)
    # Explicit defaults (overrides controller fallbacks)
    ik_params: Dict[str, Union[float, List[float]]] = field(default_factory=lambda: {
        "k_val": 1.00,
        "min_singular_value": 1e-5,
        "lambda_val": 0.08,
        "position_weight": 1.0,
        "rotation_weight": 0.6,
        "joint_weights": [1.0, 0.5, 0.5, 0.5],
    })

    # Relative-mode gains (applied to per-step delta pose when relative mode is enabled)
    relative_pos_gain: float = 1.0
    relative_rot_gain: float = 1.0

    # Axes to ignore in orientation error during IK solving.
    # Any combination of ["roll", "pitch", "yaw"].
    # For excavator: roll is locked (hardware), yaw follows slew automatically.
    # User controls position (X,Y,Z) and pitch only.
    ignore_axes: List[str] = field(default_factory=lambda: ["roll", "yaw"])

    # Use reduced Jacobian (removes uncontrollable DOFs like roll)
    # Recommended for cleaner excavator control
    use_reduced_jacobian: bool = True

    # Relative joint limits (degrees or radians) for [slew, boom, arm, bucket]
    # Provide as list of (min, max). If magnitudes > pi they are treated as degrees and converted.
    # joint_limits_relative: List[Tuple[float, float]] = [(-90.0, 90.0), (-45.0, 60.0), (-90.0, 90.0), (-100.0, 100.0)]
    joint_limits_relative: List[Tuple[float, float]] = field(default_factory=lambda: [])


    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.015  # Final target tolerance in meters (15 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)
    
    # Optional progress feedback
    progress_update_interval: float = 2.0  # How often to print progress (seconds)

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.speed_mps > 0, "Speed must be positive"
        assert self.dt > 0, "dt must be positive"
        assert self.max_points >= 2, "max_points must be at least 2"
        assert self.update_frequency > 0, "Update frequency must be positive"
        assert 0.001 <= self.grid_resolution <= 0.1, "Grid resolution should be 1mm-10cm"
        assert 0.01 <= self.safety_margin <= 0.2, "Safety margin should be 1cm-20cm"

        # Validate ignore_axes
        allowed_axes = {"roll", "pitch", "yaw"}
        for axis in self.ignore_axes:
            assert axis in allowed_axes, f"Invalid axis '{axis}' in ignore_axes. Must be one of {allowed_axes}"


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()
