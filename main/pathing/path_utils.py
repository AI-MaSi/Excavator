"""
Path utilities for planning and execution.

Moved from `pathing_config.py` to decouple configuration dataclasses
from generic helper functions that are reused across scripts.
"""

from typing import Any, Tuple
import numpy as np


def interpolate_path(path: np.ndarray, interpolation_factor: int) -> np.ndarray:
    """Linearly densify a waypoint path by an integer interpolation factor.

    Keeps endpoints fixed, removes consecutive duplicate points, and returns
    a float32 NumPy array for downstream numeric use.
    """
    if path is None:
        return path
    pts = np.asarray(path, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts

    dedup = [pts[0]]
    for i in range(1, n):
        if not np.allclose(pts[i], dedup[-1]):
            dedup.append(pts[i])
    pts = np.asarray(dedup, dtype=np.float32)
    if len(pts) < 2:
        return pts

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
    total_len = cum[-1]
    if total_len == 0.0:
        return pts[:1]

    target_samples = int((len(pts) - 1) * (interpolation_factor + 1) + 1)
    target_samples = max(2, target_samples)

    s_samples = np.linspace(0.0, float(total_len), target_samples, dtype=np.float32)

    def find_seg(s):
        i = np.searchsorted(cum, s, side="right") - 1
        return min(max(i, 0), len(seg_len) - 1)

    out = []
    for s in s_samples:
        i = find_seg(s)
        s0, s1 = float(cum[i]), float(cum[i + 1])
        t = 0.0 if s1 == s0 else (float(s) - s0) / (s1 - s0)
        P = pts[i] + t * (pts[i + 1] - pts[i])
        out.append(P.astype(np.float32))

    out[0] = pts[0]
    out[-1] = pts[-1]

    return np.asarray(out, dtype=np.float32)


def calculate_path_length(path: np.ndarray) -> float:
    """
    Calculate total length of path in meters.
    
    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
    Returns:
        Total path length in meters
    """
    if len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path) - 1):
        segment_length = np.linalg.norm(path[i + 1] - path[i])
        total_length += segment_length

    return total_length


def interpolate_along_path(path: np.ndarray, progress: float) -> np.ndarray:
    """
    Get position along path based on progress (0.0 to 1.0). Mainly for IRL debugging

    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
        progress: Progress along path from 0.0 (start) to 1.0 (end)
    Returns:
        Interpolated position as np.ndarray of shape [3]
    """
    if progress <= 0.0:
        return path[0]
    if progress >= 1.0:
        return path[-1]

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(len(path) - 1):
        segment_length = np.linalg.norm(path[i + 1] - path[i])
        distances.append(distances[-1] + segment_length)

    total_length = distances[-1]
    if total_length == 0:
        return path[0]

    target_distance = progress * total_length

    # Find which segment we're in
    for i in range(len(distances) - 1):
        if target_distance <= distances[i + 1]:
            # Interpolate within this segment
            segment_progress = (target_distance - distances[i]) / (distances[i + 1] - distances[i])
            return path[i] + segment_progress * (path[i + 1] - path[i])

    return path[-1]


def precompute_cumulative_distances(path: np.ndarray) -> Tuple[np.ndarray, float]:
    """Precompute cumulative arc-length distances for a polyline path.

    Returns (cum_dist, total_length) where cum_dist[i] is the distance from
    the start to waypoint i.
    """
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return np.asarray([0.0], dtype=np.float32), 0.0
    if len(pts) == 1:
        return np.asarray([0.0], dtype=np.float32), 0.0

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float32)])
    total_len = float(cum[-1])
    return cum.astype(np.float32), total_len


def interpolate_at_s(path: np.ndarray, cum_dist: np.ndarray, s: float) -> np.ndarray:
    """Interpolate a point along the path at absolute arc length s (meters)."""
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    if len(pts) == 1:
        return pts[0]

    total = float(cum_dist[-1])
    if s <= 0.0:
        return pts[0]
    if s >= total:
        return pts[-1]

    i = int(np.searchsorted(cum_dist, s, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    s0 = float(cum_dist[i])
    s1 = float(cum_dist[i + 1])
    t = 0.0 if s1 == s0 else (float(s) - s0) / (s1 - s0)
    return (pts[i] + t * (pts[i + 1] - pts[i])).astype(np.float32)


def project_point_onto_path(point: np.ndarray, path: np.ndarray, cum_dist: np.ndarray) -> Tuple[float, np.ndarray, int, float]:
    """Project a 3D point onto a polyline path.

    Returns:
        s_proj: Arc length (meters) to the projection point from start
        p_proj: 3D coordinates of the projection
        seg_idx: Index of segment used for projection (start waypoint index)
        t: Segment-local interpolation factor in [0, 1]
    """
    p = np.asarray(point, dtype=np.float32)
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) == 0:
        return 0.0, np.zeros(3, dtype=np.float32), 0, 0.0
    if len(pts) == 1:
        return 0.0, pts[0], 0, 0.0

    best_dist2 = float("inf")
    best_s = 0.0
    best_proj = pts[0]
    best_i = 0
    best_t = 0.0

    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 <= 1e-12:
            # Degenerate segment; treat as point
            t = 0.0
            proj = a
            s_here = float(cum_dist[i])
        else:
            t = float(np.dot(p - a, ab) / ab_len2)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj = a + t * ab
            seg_len = float(np.sqrt(ab_len2))
            s_here = float(cum_dist[i]) + t * seg_len

        d2 = float(np.dot(p - proj, p - proj))
        if d2 < best_dist2:
            best_dist2 = d2
            best_s = s_here
            best_proj = proj
            best_i = i
            best_t = t

    return best_s, best_proj.astype(np.float32), best_i, float(best_t)


def calculate_execution_time(path: np.ndarray, speed_mps: float) -> float:
    """
    Calculate estimated execution time for a path.

    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]
        speed_mps: Speed in meters per second
    Returns:
        Estimated execution time in seconds
    """
    total_distance = calculate_path_length(path)
    return total_distance / speed_mps if speed_mps > 0 else 0.0


def is_target_reached(current_pos: np.ndarray, target_pos: np.ndarray, config: Any) -> bool:
    """
    Check if target position is reached within tolerance.

    Args:
        current_pos: Current position [x, y, z]
        target_pos: Target position [x, y, z]
        config: Object with attribute `final_target_tolerance`
    Returns:
        True if target is reached within tolerance
    """
    distance = np.linalg.norm(current_pos - target_pos)
    return distance < getattr(config, "final_target_tolerance", 0.0)


def print_path_info(path: np.ndarray, config: Any, label: str = "Path") -> None:
    """
    Print detailed path information for debugging.

    Args:
        path: Path waypoints
        config: Object with attribute `speed_mps`
        label: Label for the path (e.g., "A* Path", "Smooth Path")
    """
    length = calculate_path_length(path)
    estimated_time = calculate_execution_time(path, getattr(config, "speed_mps", 0.0))
    avg_spacing = length / (len(path) - 1) if len(path) > 1 else 0

    print(f"[{label}] {len(path)} waypoints, {length:.3f}m total")
    print(f"[{label}] Speed: {getattr(config, 'speed_mps', 0.0):.3f}m/s, Est. time: {estimated_time:.1f}s")
    print(f"[{label}] Avg spacing: {avg_spacing:.4f}m")
