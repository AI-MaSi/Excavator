"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment setup configuration for both sim and real systems."""

    # Task waypoints — "in-and-out" preset (shared by run_sim_v2.py and run_hw_v2.py).
    # Point A is inside the KLT bin, Point B is outside (lift-out start).
    # rotation_deg is pitch about the Y axis.
    point_a_pos: Tuple[float, float, float] = (0.55, 0.25, -0.15)   # inside the bin
    point_a_rotation_deg: float = 0.0
    point_b_pos: Tuple[float, float, float] = (0.55, -0.25, -0.20)  # outside the bin
    point_b_rotation_deg: float = 0.0

    # KLT bin pose.
    # Sim side (run_sim_v2.py): READ AT RUNTIME — places the KLT_Bin rigid object
    # at this pose, then dumps the resulting cube collisions to obstacles.json.
    # HW side (run_hw_v2.py): documentation only — HW reads the dumped obstacles.json,
    # not this value. Workflow: edit here → re-run sim to regenerate obstacles.json
    # → ship the new obstacles.json to HW. Keep this in sync with the dumped
    # obstacles.json "klt_bin_pos"/"klt_bin_rot".
    klt_bin_pos: Tuple[float, float, float] = (0.55, 0.25, -0.15)
    klt_bin_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Single wall configuration (matches real hardware format).
    # NOTE: hardware obstacles are currently loaded from obstacles.json; this
    # wall entry is kept for sim/standalone use.
    wall_size: Tuple[float, float, float] = (0.03, 0.50, 0.60)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (0.55, 0.0, -0.45)     # Wall center position
    wall_rot: Tuple[float, float, float, float] = (0.985, 0.0, 0.0, -0.174)  # Quaternion rotation


@dataclass
class PathExecutionConfig:
    """Unified configuration for path execution in both sim and real systems."""

    # Motion parameters ------------------------------
    speed_mps: float = 0.070  #(0.02...70worked good) Target constant speed for standardized execution (m/s)
    # Jerk-limited controller smoothing settings.
    # NOTE: This jerk system is not hardware-tested yet and probably does not
    # work reliably; keep these values conservative until it is validated.
    max_accel_mps2: float = 0.5
    max_decel_mps2: float = 0.5
    max_jerk_mps3: float = 2.0
    enable_jerk: bool = False
    update_frequency: float = 100.0  # Hz - target harware loop frequency / simulation update rate
    dt: float = 1 / 100        # Pathing waypoint sample period — matches sim physics dt (headless).


    # Normalization / trajectory representation options
    # These are forwarded into NormalizerParams for all planners.
    normalizer_return_poses: bool = True  # Whether normalized planners return poses alongside positions (7D).
    normalizer_force_goal: bool = True    # Force exact goal as final waypoint when collision-free.



    # Path planning general parameters (apply to all algorithms)
    grid_resolution: float = 0.020  # 20mm. Grid cell size used for A* (and bounds for others)
    safety_margin: float = 0.05    # Obstacle safety margin in meters.

    # Workspace sampling bounds (world frame, meters). These act as a minimum
    # extent for the planner sampling region: the actual bounds are stretched
    # to enclose start/goal/obstacles + padding, then clamped to be at least
    # as wide as [workspace_min_bounds, workspace_max_bounds]. Tune these to
    workspace_min_bounds: Tuple[float, float, float] = (0.34, -0.36, 0.0)
    workspace_max_bounds: Tuple[float, float, float] = (0.85, 0.36, 0.1) #Z was 0.78
    workspace_padding: float = 0.1

    # Algorithm dimensionality
    # Use full 3D planning vs X-Z plane only
    use_3d: bool = True

    # Algorithm tuning knobs — shared by sim and HW.
    # These mirror the defaults in pathing.path_planning_algorithms (AStarParams,
    # RRTParams, RRTStarParams, PRMParams). Override here to tune both runtimes.
    astar_max_iterations: int = 200_000
    rrt_max_iterations: int = 10_000
    rrt_max_step_size: float = 0.05
    rrt_goal_bias: float = 0.1
    rrt_goal_tolerance: float = 0.02
    rrt_star_rewire_radius: float = 0.08
    rrt_star_min_iterations: int = 1000
    rrt_star_cost_patience: int = 2000
    prm_num_samples: int = 1500
    prm_connection_radius: float = 0.20
    prm_max_connections: int = 15

    # Radial-mode trajectory shaping (shared by sim and HW).
    radial_rdp_epsilon: float = 0.001              # waypoint simplification tolerance (m)
    radial_compensation_alpha: float = 0.8         # radial-error compensation blend
    radial_smoothing_samples: int = 120            # post-fit resample count

    # NOTE: IK configuration has been moved to configuration_files/control_config.yaml
    # See the 'ik' section in that file for: command_type, method, velocity_mode,
    # params, relative gains, ignore_axes, joint_limits_relative


    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.030  # Final target tolerance in meters (10 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)

    # Optional progress feedback
    progress_update_interval: int = 1  # How often to print progress (seconds)


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()
