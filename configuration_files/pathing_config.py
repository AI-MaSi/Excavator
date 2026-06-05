"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPreset:
    """Geometry and goal configuration for a named task.

    Shared by sim (run_sim_v2.py) and hardware — copy this file to both runtimes.
    The 'rotation' task appends a runtime home-position waypoint in the sim/HW
    runner; only the static legs are listed here.
    """

    name: str
    klt_bin_pos: Tuple[float, float, float]
    klt_bin_rot: Tuple[float, float, float, float]
    goals: Tuple[Tuple[float, float, float], ...]
    labels: Tuple[str, ...]
    use_klt_obstacles: bool = True


@dataclass
class EnvironmentConfig:
    """Per-task waypoints, pitches, and KLT bin poses for both sim and hardware.

    Single source of truth — every task in TASK_PRESETS pulls its data from
    here so editing a coordinate in one place propagates to sim (run_sim_v2.py)
    and HW (run_hw_v2.py). Point A is the inside / start waypoint; Point B is
    the outside / end waypoint. ``*_rotation_deg`` is pitch about the Y axis.

    Workflow for KLT bin geometry: edit ``*_klt_bin_pos``/``*_klt_bin_rot`` →
    re-run ``run_sim_v2.py --task <name> --dump-obstacles obstacles.json`` →
    ship the new ``obstacles.json`` to HW. The bin pose is read at sim runtime
    to place the rigid object; HW only uses the dumped obstacles.json.
    """

    # --- Task: in-and-out (cabin facing +X, A inside bin, B outside) ----------
    in_and_out_point_a_pos: Tuple[float, float, float] = (0.55, 0.25, -0.15)
    in_and_out_point_a_rotation_deg: float = 0.0
    in_and_out_point_b_pos: Tuple[float, float, float] = (0.55, -0.25, -0.20)
    in_and_out_point_b_rotation_deg: float = 0.0
    in_and_out_klt_bin_pos: Tuple[float, float, float] = (0.55, 0.25, -0.15)
    in_and_out_klt_bin_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # --- Task: xz-in-and-out (goals near y=0, exercises planar XZ planner) ---
    xz_point_a_pos: Tuple[float, float, float] = (0.43, 0.05, -0.10)
    xz_point_a_rotation_deg: float = 0.0
    xz_point_b_pos: Tuple[float, float, float] = (0.63, -0.05, -0.10)
    xz_point_b_rotation_deg: float = 0.0
    xz_klt_bin_pos: Tuple[float, float, float] = (0.75, 0.03, -0.15)
    xz_klt_bin_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # --- Task: rotation (mirrored across X — return leg exercises cabin slew) ---
    rotation_point_a_pos: Tuple[float, float, float] = (-0.55, 0.25, -0.15)
    rotation_point_a_rotation_deg: float = 0.0
    rotation_point_b_pos: Tuple[float, float, float] = (-0.55, -0.25, -0.20)
    rotation_point_b_rotation_deg: float = 0.0
    rotation_klt_bin_pos: Tuple[float, float, float] = (-0.55, 0.25, -0.15)
    rotation_klt_bin_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # --- Task: empty (rotation waypoints, KLT bin parked far away → no obstacles) ---
    empty_point_a_pos: Tuple[float, float, float] = (-0.55, 0.25, -0.15)
    empty_point_b_pos: Tuple[float, float, float] = (-0.55, -0.25, -0.20)
    empty_klt_bin_pos: Tuple[float, float, float] = (5.0, 5.0, -5.0)
    empty_klt_bin_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # --- Legacy single-wall obstacle (sim/standalone use only).
    #     HW loads obstacles.json instead; this wall is the fallback geometry
    #     used by run_hw_v2 when --obstacles-json is not supplied. Also written
    #     to metrics.csv for run identification.
    wall_size: Tuple[float, float, float] = (0.03, 0.50, 0.60)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (0.55, 0.0, -0.45)
    wall_rot: Tuple[float, float, float, float] = (0.985, 0.0, 0.0, -0.174)


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
    final_target_tolerance: float = 0.030  # Final target tolerance in meters (30 mm)
    orientation_tolerance: float = 0.0872665  # Orientation tolerance in radians (~5 deg)

    # Optional progress feedback
    progress_update_interval: int = 1  # How often to print progress (seconds)


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()

# ---------------------------------------------------------------------------
# Task presets — edit here, then copy this file to the hardware machine.
# ---------------------------------------------------------------------------
_e = DEFAULT_ENV_CONFIG

TASK_PRESETS: Dict[str, TaskPreset] = {
    # Standard bin pick-and-place: EE travels inside the KLT bin then back out.
    "in-and-out": TaskPreset(
        name="in-and-out",
        klt_bin_pos=_e.in_and_out_klt_bin_pos,
        klt_bin_rot=_e.in_and_out_klt_bin_rot,
        goals=(_e.in_and_out_point_a_pos, _e.in_and_out_point_b_pos),
        labels=("A_INSIDE", "B_OUTSIDE"),
    ),
    # Like in-and-out but goals are constrained to the XZ plane (y≈0).
    "xz-in-and-out": TaskPreset(
        name="xz-in-and-out",
        klt_bin_pos=_e.xz_klt_bin_pos,
        klt_bin_rot=_e.xz_klt_bin_rot,
        goals=(_e.xz_point_a_pos, _e.xz_point_b_pos),
        labels=("A_XZ_INSIDE", "B_XZ_OUTSIDE"),
    ),
    # Full cabin rotation: inside-behind → outside-behind → home.
    # The runner appends a third "home" waypoint at the EE start position.
    "rotation": TaskPreset(
        name="rotation",
        klt_bin_pos=_e.rotation_klt_bin_pos,
        klt_bin_rot=_e.rotation_klt_bin_rot,
        goals=(_e.rotation_point_a_pos, _e.rotation_point_b_pos),
        labels=("A_INSIDE_BEHIND", "B_OUTSIDE_BEHIND"),
    ),
    # Same rotation waypoints but KLT bin is parked far away so the planner
    # runs completely obstacle-free (useful for planner isolation).
    "empty": TaskPreset(
        name="empty",
        klt_bin_pos=_e.empty_klt_bin_pos,
        klt_bin_rot=_e.empty_klt_bin_rot,
        goals=(_e.empty_point_a_pos, _e.empty_point_b_pos),
        labels=("A_EMPTY_ROTATION", "B_EMPTY_ROTATION"),
        use_klt_obstacles=False,
    ),
}
