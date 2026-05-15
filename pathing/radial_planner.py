"""
Radial-first planner with separated raw avoidance, radial compensation, and smoothing.

Pipeline:
    1. Generate a radial seed (cylindrical interpolation start -> goal).
    2. If collision-free, execute the seed directly.
    3. Otherwise retreat along the seed to a collision-free handoff and run the
       base planner (A*/RRT/...) from there to the goal. Merge prefix + suffix.
       This merged path is the "red" raw avoidance (piecewise-linear, no smoothing,
       no radial compensation).
    4. Radial compensation pulls the raw suffix's RDP-simplified corners toward
       the seed's radius profile (blend, not snap). The merged prefix + compensated
       suffix is the "yellow" radial path. No smoothing.
    5. Smoothing is a generic Cartesian cubic-Hermite pass applied as a separate
       step on top of either red or yellow.
    6. Execution selection: smooth(yellow) -> smooth(red) -> red.
       In ``execute_raw`` mode, yellow is skipped: smooth(red) -> red.

Pure numpy -- no torch dependency.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .path_utils import ObstacleChecker, rdp_simplify, standardize_path
from .path_planning_algorithms import LINE_CHECK_MIN_SAMPLES, LINE_CHECK_SPACING_M

logger = logging.getLogger(__name__)


# ── Path primitives ───────────────────────────────────────────────────────


def generate_radial_path(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    num_samples: int = 50,
    r_min: float = 0.15,
    r_max: float = 0.65,
    z_min: float = -0.20,
    z_max: float = 0.50,
) -> np.ndarray:
    """Smooth cylindrical interpolation (yaw, r, z) from start to goal."""
    s = np.asarray(start_pos, dtype=np.float64)
    g = np.asarray(goal_pos, dtype=np.float64)

    start_yaw = np.arctan2(s[1], s[0])
    start_r = np.hypot(s[0], s[1])
    start_z = s[2]

    goal_yaw = np.arctan2(g[1], g[0])
    goal_r = np.hypot(g[0], g[1])
    goal_z = g[2]

    delta_yaw = goal_yaw - start_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi

    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    smooth = t * t * (3.0 - 2.0 * t)

    yaw_i = start_yaw + smooth * delta_yaw
    r_i = np.clip(start_r + smooth * (goal_r - start_r), r_min, r_max)
    z_i = np.clip(start_z + smooth * (goal_z - start_z), z_min, z_max)

    x = r_i * np.cos(yaw_i)
    y = r_i * np.sin(yaw_i)
    return np.stack([x, y, z_i], axis=-1).astype(np.float32)


def find_blocked_segments(path: np.ndarray, checker: ObstacleChecker) -> List[Tuple[int, int]]:
    """Walk path segments and return inclusive index ranges that are blocked."""
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) < 2:
        return []

    blocked: List[int] = []
    for i in range(len(pts) - 1):
        if not checker.is_line_collision_free(tuple(pts[i].tolist()), tuple(pts[i + 1].tolist())):
            blocked.append(i)

    if not blocked:
        return []

    merged: List[Tuple[int, int]] = []
    seg_start = blocked[0]
    seg_end = blocked[0] + 1
    for idx in blocked[1:]:
        if idx <= seg_end:
            seg_end = idx + 1
            continue
        merged.append((seg_start, seg_end))
        seg_start = idx
        seg_end = idx + 1
    merged.append((seg_start, seg_end))
    return merged


def _path_is_collision_free(path: np.ndarray, checker: ObstacleChecker) -> bool:
    return len(find_blocked_segments(path, checker)) == 0


def _dedupe_consecutive(path: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) <= 1:
        return pts
    deduped = [pts[0]]
    for point in pts[1:]:
        if np.linalg.norm(point - deduped[-1]) > tol:
            deduped.append(point)
    return np.asarray(deduped, dtype=np.float32)


def _merge_prefix_and_suffix(prefix: np.ndarray, suffix: np.ndarray) -> np.ndarray:
    prefix_pts = _dedupe_consecutive(prefix)
    suffix_pts = _dedupe_consecutive(suffix)
    if len(prefix_pts) == 0:
        return suffix_pts
    if len(suffix_pts) == 0:
        return prefix_pts
    if np.linalg.norm(prefix_pts[-1] - suffix_pts[0]) <= 1e-4:
        merged = np.concatenate([prefix_pts, suffix_pts[1:]], axis=0)
    else:
        merged = np.concatenate([prefix_pts, suffix_pts], axis=0)
    return _dedupe_consecutive(merged)


def _unwrap_yaw_sequence(yaws: np.ndarray) -> np.ndarray:
    unwrapped = np.asarray(yaws, dtype=np.float64).copy()
    for i in range(1, len(unwrapped)):
        delta = (unwrapped[i] - unwrapped[i - 1] + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped[i] = unwrapped[i - 1] + delta
    return unwrapped


# ── Radial compensation (yellow) ──────────────────────────────────────────


def radial_compensate(
    raw_suffix: np.ndarray,
    radial_seed: np.ndarray,
    *,
    rdp_epsilon: float,
    alpha: float = 0.8,
    r_min: float = 0.15,
    r_max: float = 0.65,
) -> np.ndarray:
    """Pull raw-suffix RDP corners' radius toward the seed's r(yaw) profile.

    Keeps the handoff and goal endpoints bit-exact. Interior corners get their
    radius blended toward the seed: ``r_new = alpha * r_seed(yaw) + (1-alpha) * r_raw``.
    Returns a piecewise-linear Cartesian XYZ path -- no smoothing.
    """
    raw = _dedupe_consecutive(raw_suffix)
    if len(raw) < 2:
        return raw

    waypoints = rdp_simplify(raw, epsilon=rdp_epsilon)
    if len(waypoints) < 2:
        waypoints = np.asarray([raw[0], raw[-1]], dtype=np.float32)
    waypoints = _dedupe_consecutive(waypoints).astype(np.float32)
    if len(waypoints) < 3:
        return waypoints

    seed = np.asarray(radial_seed, dtype=np.float64)
    if len(seed) < 2:
        return waypoints

    seed_yaws = _unwrap_yaw_sequence(np.arctan2(seed[:, 1], seed[:, 0]))
    seed_radii = np.hypot(seed[:, 0], seed[:, 1])
    if seed_yaws[-1] < seed_yaws[0]:
        seed_yaws = seed_yaws[::-1]
        seed_radii = seed_radii[::-1]

    anchor = float(seed_yaws[0])
    out = waypoints.copy()
    for i in range(1, len(waypoints) - 1):
        x, y, z = float(out[i, 0]), float(out[i, 1]), float(out[i, 2])
        wp_yaw = math.atan2(y, x)
        delta = (wp_yaw - anchor + math.pi) % (2.0 * math.pi) - math.pi
        wp_yaw_unwrapped = anchor + delta
        wp_r = math.hypot(x, y)

        seed_r = float(np.interp(wp_yaw_unwrapped, seed_yaws, seed_radii))
        r_new = alpha * seed_r + (1.0 - alpha) * wp_r
        r_new = float(np.clip(r_new, r_min, r_max))

        out[i, 0] = r_new * math.cos(wp_yaw)
        out[i, 1] = r_new * math.sin(wp_yaw)
        out[i, 2] = z

    out[0] = waypoints[0]
    out[-1] = waypoints[-1]
    return _dedupe_consecutive(out)


# ── Smoothing ─────────────────────────────────────────────────────────────


def smooth_path(path: np.ndarray, num_samples: int = 120) -> np.ndarray:
    """Round corners via cubic Hermite interpolation in Cartesian XYZ.

    Parameterized by cumulative arc length. Tangents are central differences
    (one-sided at endpoints). Pure geometry -- no knowledge of radius.
    """
    pts = _dedupe_consecutive(np.asarray(path, dtype=np.float64))
    n = len(pts)
    if n < 3:
        return pts.astype(np.float32)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-6:
        return pts.astype(np.float32)

    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            dt = max(s[1] - s[0], 1e-6)
            tangents[i] = (pts[1] - pts[0]) / dt
        elif i == n - 1:
            dt = max(s[-1] - s[-2], 1e-6)
            tangents[i] = (pts[-1] - pts[-2]) / dt
        else:
            dt_prev = max(s[i] - s[i - 1], 1e-6)
            dt_next = max(s[i + 1] - s[i], 1e-6)
            sec_prev = (pts[i] - pts[i - 1]) / dt_prev
            sec_next = (pts[i + 1] - pts[i]) / dt_next
            tangents[i] = 0.5 * (sec_prev + sec_next)

    sample_count = max(int(num_samples), n)
    sample_s = np.linspace(0.0, total, sample_count)
    out = np.empty((sample_count, 3), dtype=np.float64)
    seg_idx = 0
    for k, sv in enumerate(sample_s):
        while seg_idx < n - 2 and sv > s[seg_idx + 1]:
            seg_idx += 1
        s0 = s[seg_idx]
        s1 = s[seg_idx + 1]
        h = max(s1 - s0, 1e-6)
        u = np.clip((sv - s0) / h, 0.0, 1.0)

        p0 = pts[seg_idx]
        p1 = pts[seg_idx + 1]
        m0 = tangents[seg_idx] * h
        m1 = tangents[seg_idx + 1] * h

        h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
        h10 = u ** 3 - 2.0 * u ** 2 + u
        h01 = -2.0 * u ** 3 + 3.0 * u ** 2
        h11 = u ** 3 - u ** 2
        out[k] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    out[0] = pts[0]
    out[-1] = pts[-1]
    return out.astype(np.float32)


# ── Handoff search & raw planning ─────────────────────────────────────────


def _candidate_handoff_indices(
    radial_path: np.ndarray,
    first_blocked_segment: int,
    checker: ObstacleChecker,
    retreat_step: int = 2,
) -> List[int]:
    start_idx = min(max(int(first_blocked_segment), 0), len(radial_path) - 1)
    candidates: List[int] = []
    idx = start_idx
    while idx >= 0:
        if checker.is_point_collision_free(tuple(radial_path[idx].tolist())):
            candidates.append(idx)
        idx -= max(retreat_step, 1)
    if candidates and candidates[-1] != 0 and checker.is_point_collision_free(tuple(radial_path[0].tolist())):
        candidates.append(0)
    if not candidates:
        candidates = [0]
    return candidates


def _plan_raw_suffix(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    fallback_algorithm: str,
    grid_resolution: float,
    safety_margin: float,
    normalizer_params,
    astar_params=None,
    rrt_params=None,
    rrt_star_params=None,
    prm_params=None,
    use_3d: bool = True,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    from .normalized_planners import plan_to_target

    start = np.asarray(start_pos, dtype=np.float32)
    goal = np.asarray(goal_pos, dtype=np.float32)
    if np.linalg.norm(goal - start) <= 1e-6:
        return np.stack([start, goal], axis=0).astype(np.float32)

    algorithms = [fallback_algorithm]
    if fallback_algorithm.endswith("_plane"):
        algorithms.append(fallback_algorithm[:-6])

    for algorithm in algorithms:
        try:
            result = plan_to_target(
                start_pos_world=tuple(start.tolist()),
                target_pos_world=tuple(goal.tolist()),
                obstacle_data=obstacle_data,
                algorithm=algorithm,
                grid_resolution=grid_resolution,
                safety_margin=safety_margin,
                normalizer_params=normalizer_params,
                astar_params=astar_params,
                rrt_params=rrt_params,
                rrt_star_params=rrt_star_params,
                prm_params=prm_params,
                use_3d=use_3d,
                verbose=verbose,
            )
            exec_positions = result.get("exec_positions")
            if exec_positions is not None and len(exec_positions) > 0:
                return {
                    "positions": np.asarray(exec_positions, dtype=np.float32),
                    "planning_stats": result.get("planning_stats"),
                }
        except Exception as err:  # noqa: BLE001
            logger.info("[RadialPlanner] Raw suffix planner %s failed: %s: %s", algorithm, type(err).__name__, err)
    return None


# ── Result composition ───────────────────────────────────────────────────


def _compose_result(
    exec_path: np.ndarray,
    *,
    normalizer_params,
    radial_seed: np.ndarray | None = None,
    handoff_point: np.ndarray | None = None,
    raw_suffix: np.ndarray | None = None,
    radial_suffix: np.ndarray | None = None,
    chosen_variant: str = "seed",
    planning_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = standardize_path(
        exec_path,
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=normalizer_params.return_poses,
    )
    if radial_seed is not None:
        result["radial_seed_positions"] = np.asarray(radial_seed, dtype=np.float32)
    if handoff_point is not None:
        result["handoff_point_position"] = np.asarray(handoff_point, dtype=np.float32)
    if raw_suffix is not None:
        result["raw_suffix_positions"] = np.asarray(raw_suffix, dtype=np.float32)
    if radial_suffix is not None:
        result["radial_suffix_positions"] = np.asarray(radial_suffix, dtype=np.float32)
    result["chosen_variant"] = chosen_variant
    if planning_stats:
        result["planning_stats"] = planning_stats
    return result


# ── Main entry ───────────────────────────────────────────────────────────


def plan_radial(
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    obstacle_data: List[Dict[str, Any]],
    fallback_algorithm: str,
    *,
    grid_resolution: float = 0.10,
    safety_margin: float = 0.02,
    normalizer_params=None,
    astar_params=None,
    rrt_params=None,
    rrt_star_params=None,
    prm_params=None,
    use_3d: bool = True,
    verbose: bool = False,
    num_radial_samples: int = 50,
    r_min: float = 0.15,
    r_max: float = 0.65,
    z_min: float = -0.20,
    z_max: float = 0.50,
    rdp_epsilon: float = 0.005,
    radial_compensation_alpha: float = 0.8,
    smoothing_samples: int = 120,
    execute_raw: bool = False,
) -> Dict[str, Any]:
    """Plan a radial-first trajectory with separated compensation and smoothing."""
    from .normalized_planners import NormalizerParams

    checker = ObstacleChecker(
        obstacle_data or [],
        safety_margin=safety_margin,
        line_check_spacing_m=LINE_CHECK_SPACING_M,
        min_line_samples=LINE_CHECK_MIN_SAMPLES,
    )
    radial_seed = generate_radial_path(
        start_pos,
        goal_pos,
        num_samples=num_radial_samples,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )

    if _path_is_collision_free(radial_seed, checker):
        logger.info("[RadialPlanner] Direct radial seed is collision-free")
        return _compose_result(
            exec_path=radial_seed,
            normalizer_params=normalizer_params,
            radial_seed=radial_seed,
            chosen_variant="seed",
        )

    blocked_segments = find_blocked_segments(radial_seed, checker)
    first_blocked_segment, _ = blocked_segments[0]
    logger.info(
        "[RadialPlanner] Seed blocks at segment %d; searching for a clean handoff",
        first_blocked_segment,
    )

    detour_normalizer = NormalizerParams(
        speed_mps=normalizer_params.speed_mps,
        dt=normalizer_params.dt,
        return_poses=False,
        force_goal=True,
    )

    for handoff_idx in _candidate_handoff_indices(radial_seed, first_blocked_segment, checker):
        prefix = radial_seed[: handoff_idx + 1]
        handoff = tuple(prefix[-1].tolist())

        raw_suffix_result = _plan_raw_suffix(
            handoff,
            goal_pos,
            obstacle_data=obstacle_data,
            fallback_algorithm=fallback_algorithm,
            grid_resolution=grid_resolution,
            safety_margin=safety_margin,
            normalizer_params=detour_normalizer,
            astar_params=astar_params,
            rrt_params=rrt_params,
            rrt_star_params=rrt_star_params,
            prm_params=prm_params,
            use_3d=use_3d,
            verbose=verbose,
        )
        if raw_suffix_result is None:
            continue

        raw_suffix = raw_suffix_result["positions"]
        planning_stats = raw_suffix_result.get("planning_stats")
        raw_path = _merge_prefix_and_suffix(prefix, raw_suffix)
        if not _path_is_collision_free(raw_path, checker):
            continue

        handoff_arr = np.asarray(handoff, dtype=np.float32)

        if execute_raw:
            smoothed_red = smooth_path(raw_path, num_samples=smoothing_samples)
            if _path_is_collision_free(smoothed_red, checker):
                exec_path, chosen = smoothed_red, "smoothed_red"
            else:
                exec_path, chosen = raw_path, "red"
            logger.info("[RadialPlanner] handoff=%d mode=raw chosen=%s", handoff_idx, chosen)
            return _compose_result(
                exec_path=exec_path,
                normalizer_params=normalizer_params,
                radial_seed=radial_seed,
                handoff_point=handoff_arr,
                raw_suffix=raw_suffix,
                chosen_variant=chosen,
                planning_stats=planning_stats,
            )

        radial_suffix = radial_compensate(
            raw_suffix,
            radial_seed,
            rdp_epsilon=rdp_epsilon,
            alpha=radial_compensation_alpha,
            r_min=r_min,
            r_max=r_max,
        )
        radial_path = _merge_prefix_and_suffix(prefix, radial_suffix)

        smoothed_yellow = smooth_path(radial_path, num_samples=smoothing_samples)
        if _path_is_collision_free(smoothed_yellow, checker):
            exec_path, chosen = smoothed_yellow, "smoothed_yellow"
        else:
            smoothed_red = smooth_path(raw_path, num_samples=smoothing_samples)
            if _path_is_collision_free(smoothed_red, checker):
                exec_path, chosen = smoothed_red, "smoothed_red"
            else:
                exec_path, chosen = raw_path, "red"

        logger.info("[RadialPlanner] handoff=%d chosen=%s", handoff_idx, chosen)
        return _compose_result(
            exec_path=exec_path,
            normalizer_params=normalizer_params,
            radial_seed=radial_seed,
            handoff_point=handoff_arr,
            raw_suffix=raw_suffix,
            radial_suffix=radial_suffix,
            chosen_variant=chosen,
            planning_stats=planning_stats,
        )

    logger.warning("[RadialPlanner] No handoff produced a collision-free raw path; falling back to direct raw from start")
    raw_from_start_result = _plan_raw_suffix(
        start_pos,
        goal_pos,
        obstacle_data=obstacle_data,
        fallback_algorithm=fallback_algorithm,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        normalizer_params=detour_normalizer,
        astar_params=astar_params,
        rrt_params=rrt_params,
        rrt_star_params=rrt_star_params,
        prm_params=prm_params,
        use_3d=use_3d,
        verbose=verbose,
    )
    if raw_from_start_result is None:
        raise RuntimeError("Radial planner could not produce a fallback raw path")

    raw_from_start = raw_from_start_result["positions"]
    return _compose_result(
        exec_path=raw_from_start,
        normalizer_params=normalizer_params,
        radial_seed=radial_seed,
        raw_suffix=raw_from_start,
        chosen_variant="red",
        planning_stats=raw_from_start_result.get("planning_stats"),
    )
