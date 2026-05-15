"""
Normalized (constant-speed) path planning wrappers.
===================================================

Public-facing helpers that:
- Call the low-level planners from path_planning_algorithms.
- Run the resulting paths through ``standardize_path`` to enforce a
  constant-speed, fixed-dt representation.

This keeps the core algorithms focused on geometry/search, and
collects the execution-shaping logic in one place.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Tuple, Optional

import logging
import numpy as np

from .path_utils import standardize_path, ObstacleChecker
from .radial_planner import plan_radial
from .path_planning_algorithms import (
    AStarParams,
    LINE_CHECK_MIN_SAMPLES,
    LINE_CHECK_SPACING_M,
    RRTParams,
    RRTStarParams,
    PRMParams,
    create_astar_3d_trajectory,
    create_astar_plane_trajectory,
    create_rrt_plane_trajectory,
    create_rrt_star_plane_trajectory,
    create_rrt_star_trajectory,
    create_rrt_trajectory,
    create_prm_trajectory,
    create_prm_plane_trajectory,
    get_last_plan_stats,
)


logger = logging.getLogger(__name__)


@dataclass
class NormalizerParams:
    """Configuration for converting raw paths into normalized trajectories.

    All fields must be provided explicitly by higher-level config (no internal defaults).
    """

    speed_mps: float  # Target constant speed (m/s)
    dt: float         # Sample period (s)
    # Whether to return full poses (4x4 matrices / 7D pose vectors) or just positions.
    return_poses: bool
    # When True, the normalized trajectory is forced to end exactly at the requested goal
    # (if a straight-line segment from the last planner waypoint to the goal is collision-free).
    force_goal: bool


def force_goal_as_final_waypoint(
    path: np.ndarray,
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    safety_margin: float,
    force_goal: bool,
) -> np.ndarray:
    """Optionally append the exact goal as final waypoint, in world coordinates.

    This is done before normalization so constant-speed resampling remains valid.
    A straight segment from the last waypoint to the goal is only added if it is
    collision-free with respect to the (already inflated) obstacles.
    """
    if not force_goal:
        return path

    if path is None:
        return path

    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return pts

    goal = np.asarray(goal_pos, dtype=np.float32)
    last = pts[-1]

    # If we're already effectively at the goal, do nothing.
    if np.linalg.norm(last - goal) < 1e-4:
        return pts

    checker = ObstacleChecker(
        obstacle_data or [],
        safety_margin=safety_margin,
        line_check_spacing_m=LINE_CHECK_SPACING_M,
        min_line_samples=LINE_CHECK_MIN_SAMPLES,
    )
    if checker.is_line_collision_free(tuple(last.tolist()), tuple(goal.tolist())):
        pts = np.vstack([pts, goal.astype(np.float32)])

    return pts


# ── Algorithm registry ────────────────────────────────────────────────────
# Maps algorithm name -> (raw_planner_fn, params_kwarg_name, default_params_class)

_ALGORITHM_REGISTRY: Dict[str, tuple] = {
    "a_star":           (create_astar_3d_trajectory,        "astar_params",     AStarParams),
    "a_star_plane":     (create_astar_plane_trajectory,     "astar_params",     AStarParams),
    "rrt":              (create_rrt_trajectory,              "rrt_params",       RRTParams),
    "rrt_plane":        (create_rrt_plane_trajectory,        "rrt_params",       RRTParams),
    "rrt_star":         (create_rrt_star_trajectory,         "rrt_star_params",  RRTStarParams),
    "rrt_star_plane":   (create_rrt_star_plane_trajectory,   "rrt_star_params",  RRTStarParams),
    "prm":              (create_prm_trajectory,              "prm_params",       PRMParams),
    "prm_plane":        (create_prm_plane_trajectory,        "prm_params",       PRMParams),
}

# Algorithms whose raw functions accept a `use_3d` parameter (3D variants only)
_ACCEPTS_USE_3D = {"rrt", "rrt_star", "prm"}
# Algorithms whose raw functions accept a `verbose` parameter
_ACCEPTS_VERBOSE = {"a_star", "a_star_plane"}

SUPPORTED_ALGORITHMS = tuple(_ALGORITHM_REGISTRY.keys())


def _plan_and_normalize(
    raw_fn,
    raw_kwargs: dict,
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    safety_margin: float,
    normalizer_params: NormalizerParams,
) -> Dict[str, np.ndarray]:
    """Run a raw planner, append goal if requested, then normalize to constant speed."""
    raw = raw_fn(**raw_kwargs)
    planning_stats = get_last_plan_stats()
    raw = force_goal_as_final_waypoint(raw, goal_pos, obstacle_data, safety_margin, normalizer_params.force_goal)
    result = standardize_path(
        raw,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )
    if planning_stats:
        result["planning_stats"] = planning_stats
    return result


def plan_to_target(
    start_pos_world: Tuple[float, float, float],
    target_pos_world: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    *,
    algorithm: str,
    grid_resolution: float,
    safety_margin: float,
    normalizer_params: NormalizerParams,
    astar_params: Optional[AStarParams] = None,
    rrt_params: Optional[RRTParams] = None,
    rrt_star_params: Optional[RRTStarParams] = None,
    prm_params: Optional[PRMParams] = None,
    use_3d: bool = True,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Plan a normalized path from an arbitrary start pose to a target.

    This is intended for feeding the *current* robot EE position directly
    into the planners (e.g. ee->A, ee->B instead of A->B, B->A).

    The underlying planners operate in position space only. Orientation in
    the returned trajectory is currently set to identity (no rotation).
    This matches the current demos where pitch is kept at 0 and other
    orientation axes are ignored / not actuated.
    """
    if algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Expected one of: {', '.join(repr(k) for k in SUPPORTED_ALGORITHMS)}."
        )

    raw_fn, params_key, default_params_cls = _ALGORITHM_REGISTRY[algorithm]

    # Resolve the algorithm-specific params (use provided or default)
    params_map = {
        "astar_params": astar_params,
        "rrt_params": rrt_params,
        "rrt_star_params": rrt_star_params,
        "prm_params": prm_params,
    }
    params = params_map.get(params_key)
    if params is None:
        params = default_params_cls()

    # Build raw planner kwargs: common args + unpacked algo params
    raw_kwargs = {
        "start_pos": tuple(start_pos_world),
        "goal_pos": tuple(target_pos_world),
        "obstacle_data": obstacle_data,
        "grid_resolution": grid_resolution,
        "safety_margin": safety_margin,
        **asdict(params),
    }
    if algorithm in _ACCEPTS_USE_3D:
        raw_kwargs["use_3d"] = use_3d
    if algorithm in _ACCEPTS_VERBOSE:
        raw_kwargs["verbose"] = verbose

    return _plan_and_normalize(
        raw_fn, raw_kwargs,
        goal_pos=tuple(target_pos_world),
        obstacle_data=obstacle_data,
        safety_margin=safety_margin,
        normalizer_params=normalizer_params,
    )


# ── Radial-first wrappers ─────────────────────────────────────────────────

# Algorithms that can be used as the raw avoidance backend inside radial-first planning
RADIAL_AVOIDANCE_ALGORITHMS = tuple(f"radial_avoid_{k}" for k in _ALGORITHM_REGISTRY)


def plan_radial_to_target(
    start_pos_world: Tuple[float, float, float],
    target_pos_world: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    *,
    fallback_algorithm: str,
    grid_resolution: float,
    safety_margin: float,
    normalizer_params: NormalizerParams,
    astar_params: Optional[AStarParams] = None,
    rrt_params: Optional[RRTParams] = None,
    rrt_star_params: Optional[RRTStarParams] = None,
    prm_params: Optional[PRMParams] = None,
    use_3d: bool = True,
    verbose: bool = False,
    execute_raw: bool = False,
    rdp_epsilon: float = 0.005,
    radial_compensation_alpha: float = 0.8,
    smoothing_samples: int = 120,
) -> Dict[str, np.ndarray]:
    """Plan a radial-first path with radial compensation and smoothing around obstacles.

    Thin wrapper around :func:`plan_radial` that matches the
    ``plan_to_target`` call convention.
    """
    return plan_radial(
        start_pos=tuple(start_pos_world),
        goal_pos=tuple(target_pos_world),
        obstacle_data=obstacle_data,
        fallback_algorithm=fallback_algorithm,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        normalizer_params=normalizer_params,
        astar_params=astar_params,
        rrt_params=rrt_params,
        rrt_star_params=rrt_star_params,
        prm_params=prm_params,
        use_3d=use_3d,
        verbose=verbose,
        execute_raw=execute_raw,
        rdp_epsilon=rdp_epsilon,
        radial_compensation_alpha=radial_compensation_alpha,
        smoothing_samples=smoothing_samples,
    )
