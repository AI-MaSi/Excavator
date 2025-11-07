"""
Unified NumPy-based path planning API for both sim and IRL.

This module provides a single, consistent interface for A*, RRT, RRT*, and PRM
path planning algorithms with obstacle avoidance. It reuses the enhanced
implementations (edge sampling to prevent slewing through thin obstacles and
corner-cut prevention in 2D) and normalizes all outputs to NumPy arrays.

Defaults (grid resolution, safety margin, iterations, etc.) are sourced from
`pathing_config.py` so sim and IRL remain 1:1 at this level.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Shared configuration (world/planner defaults)
from pathing_config import DEFAULT_CONFIG

# Reuse the robust, anti-slew implementations
import sim_path_planning_algorithms as _sim


# Re-export the obstacle checker (supports line sampling and rotated boxes)
ObstacleChecker = _sim.ObstacleChecker
GridConfig = _sim.GridConfig
AStar3D = _sim.AStar3D


def _resolve(value, default):
    return default if value is None else value


def _to_numpy_path(path) -> np.ndarray:
    """Convert a sequence/torch/np path to np.ndarray[float32, (N,3)]."""
    # A*: already np in sim implementation
    if isinstance(path, np.ndarray):
        return path.astype(np.float32, copy=False)
    try:
        import torch  # type: ignore
        if isinstance(path, torch.Tensor):
            return path.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(path, dtype=np.float32)


def create_astar_3d_trajectory(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: Optional[float] = None,
    safety_margin: Optional[float] = None,
    use_3d: Optional[bool] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    A* with obstacle avoidance (NumPy output).

    - Includes anti-slew edge sampling and 2D corner-cut prevention.
    - Uses config defaults when parameters are omitted.
    """
    cfg = DEFAULT_CONFIG
    path = _sim.create_astar_3d_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=_resolve(grid_resolution, cfg.grid_resolution),
        safety_margin=_resolve(safety_margin, cfg.safety_margin),
        use_3d=_resolve(use_3d, cfg.use_3d),
    )
    return _to_numpy_path(path)


def create_rrt_star_trajectory(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: Optional[float] = None,
    safety_margin: Optional[float] = None,
    use_3d: Optional[bool] = None,
    max_iterations: Optional[int] = None,
    max_acceptable_cost: Optional[float] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    RRT* with obstacle avoidance (NumPy output).
    """
    cfg = DEFAULT_CONFIG
    path = _sim.create_rrt_star_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=_resolve(grid_resolution, cfg.grid_resolution),
        safety_margin=_resolve(safety_margin, cfg.safety_margin),
        use_3d=_resolve(use_3d, cfg.use_3d),
        max_iterations=_resolve(max_iterations, cfg.max_iterations),
        max_acceptable_cost=_resolve(max_acceptable_cost, cfg.max_acceptable_cost),
        device=_resolve(device, "cpu"),
    )
    return _to_numpy_path(path)


def create_rrt_trajectory(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: Optional[float] = None,
    safety_margin: Optional[float] = None,
    use_3d: Optional[bool] = None,
    max_iterations: Optional[int] = None,
    max_acceptable_cost: Optional[float] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    RRT (basic) with obstacle avoidance (NumPy output).
    """
    cfg = DEFAULT_CONFIG
    path = _sim.create_rrt_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=_resolve(grid_resolution, cfg.grid_resolution),
        safety_margin=_resolve(safety_margin, cfg.safety_margin),
        use_3d=_resolve(use_3d, cfg.use_3d),
        max_iterations=_resolve(max_iterations, cfg.max_iterations),
        max_acceptable_cost=_resolve(max_acceptable_cost, cfg.max_acceptable_cost),
        device=_resolve(device, "cpu"),
    )
    return _to_numpy_path(path)


def create_prm_trajectory(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    grid_resolution: Optional[float] = None,
    safety_margin: Optional[float] = None,
    use_3d: Optional[bool] = None,
    num_samples: Optional[int] = None,
    connection_radius: Optional[float] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    PRM with obstacle avoidance (NumPy output).
    """
    cfg = DEFAULT_CONFIG
    path = _sim.create_prm_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=_resolve(grid_resolution, cfg.grid_resolution),
        safety_margin=_resolve(safety_margin, cfg.safety_margin),
        use_3d=_resolve(use_3d, cfg.use_3d),
        num_samples=_resolve(num_samples, cfg.num_samples),
        connection_radius=_resolve(connection_radius, cfg.connection_radius),
        device=_resolve(device, "cpu"),
    )
    return _to_numpy_path(path)


def create_astar_on_start_goal_plane(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    wall_obstacle: Dict[str, Any],
    grid_resolution: Optional[float] = None,
    safety_margin: Optional[float] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    A* restricted to the plane spanned by start/goal and wall constraint (NumPy output).
    """
    cfg = DEFAULT_CONFIG
    path = _sim.create_astar_on_start_goal_plane(
        start_pos=start_pos,
        goal_pos=goal_pos,
        wall_obstacle=wall_obstacle,
        grid_resolution=_resolve(grid_resolution, cfg.grid_resolution),
        safety_margin=_resolve(safety_margin, cfg.safety_margin),
        device=_resolve(device, "cpu"),
    )
    return _to_numpy_path(path)

