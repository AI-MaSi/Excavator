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

    speed_mps: float = 0.020  # Target point speed in meters per second. Should be slower than what the robot can do.
    interpolation_factor: int = 5  # Points added between waypoints (higher = smoother). 1 = no interpolation.

    # TODO: only used in IRL currently!
    # Trapezoidal speed profile (accel/decel). Makes movements smoother.
    accel_mps2: float = 0.04  # Default linear acceleration (m/s^2)
    decel_mps2: float = 0.04  # Default linear deceleration (m/s^2)


    update_frequency: float = 100.0  # Hz - target loop frequency.

    # Path planning parameters
    grid_resolution: float = 0.020  # A* grid cell size in meters
    safety_margin: float = 0.06  # Obstacle safety margin in meters.
    max_iterations: int = 10000  # Max iterations for sampling-based planners (RRT, RRT*)
    num_samples: int = 1500  # Number of samples for PRM planner
    connection_radius: float = 0.4  # Connection radius for PRM planner in meters

    # Algorithm dimensionality
    # Use full 3D planning vs X-Z plane only
    # TODO: is this necessary (just set 0 Y coordinates)?
    use_3d: bool = True


    # Algo tuning parameters --------------------------------

    # Early termination threshold for RRT/RRT* (meters?).
    max_acceptable_cost: float = 0.768

    # RRT/RRT* tuning parameters
    rrt_max_step_size: float = 0.05
    rrt_goal_bias: float = 0.10
    rrt_rewire_radius: float = 0.08
    rrt_goal_tolerance: float = 0.02
    rrt_minimum_iterations: int = 1000
    rrt_cost_improvement_patience: int = 5000



    # Inverse-kinematics controller
    ik_command_type: Literal["position", "pose"] = "pose"
    ik_use_relative_mode: bool = True
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "svd"
    
    # Method/weighting parameters (values are passed through to the IK implementation)
    # Explicit defaults (overrides controller fallbacks)
    ik_params: Dict[str, Union[float, List[float]]] = field(default_factory=lambda: {
        "k_val": 10.0,
        "min_singular_value": 1e-4,
        "lambda_val": 0.08,
        "position_weight": 1.0,
        "rotation_weight": 1.0,
        "joint_weights": [1.0, 1.0, 1.0, 1.0],
    })

    # Orientation locks to reflect hardware capabilities.
    # Lock axes to the current EE orientation in the IK base frame.
    # Defaults reflect excavator: pitch (Y) controllable, roll (X) and yaw (Z) locked.
    lock_roll: bool = True
    lock_pitch: bool = False
    lock_yaw: bool = True

    # Relative-mode gains (applied to per-step delta pose when relative mode is enabled)
    relative_pos_gain: float = 0.35
    relative_rot_gain: float = 0.50


    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.015  # Final target tolerance in meters (15 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)
    
    # Optional progress feedback
    progress_update_interval: float = 2.0  # How often to print progress (seconds)

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.speed_mps > 0, "Speed must be positive"
        assert self.interpolation_factor >= 1, "Interpolation factor must be at least 1"
        assert self.accel_mps2 > 0, "accel_mps2 must be positive"
        assert self.decel_mps2 > 0, "decel_mps2 must be positive"
        assert 0 < self.final_target_tolerance <= 0.5, "Target tolerance should be 1-50cm" # non used parameter
        assert self.update_frequency > 0, "Update frequency must be positive"
        assert 0.001 <= self.grid_resolution <= 0.1, "Grid resolution should be 1mm-10cm"
        assert 0.01 <= self.safety_margin <= 0.2, "Safety margin should be 1cm-20cm"
        
        # Provide default IK params if not set, mirroring controller defaults
        if self.ik_params is None:
            base = {
                "k_val": 1.0,
                "position_weight": 1.0,
                "rotation_weight": 0.1,
            }
            if self.ik_method == "svd":
                base["min_singular_value"] = 1e-5
            elif self.ik_method == "dls":
                base["lambda_val"] = 0.01
            # For "pinv" and "trans", base is sufficient
            self.ik_params = base

        # Helper functions moved to pathing/path_utils.py


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()
