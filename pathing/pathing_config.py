"""
Copy this file to both Isaac/IRL. General configs are loaded from this file!

Task simplified to two points + single wall. Easy to add more if desired, but prob hard for irl :)
"""

import numpy as np
from typing import Tuple, List, Union
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment setup configuration for both sim and real systems."""
    
    # Point A configuration
    point_a_pos: Tuple[float, float, float] = (0.37, 0.0, -0.1)
    point_a_rotation_deg: float = 0.0 # y axis rotation. 0 = horizontal
    
    # Point B configuration  
    point_b_pos: Tuple[float, float, float] = (0.68, 0.0, -0.1)
    point_b_rotation_deg: float = 0.0
    
    # Single wall configuration (matches real hardware format)
    wall_size: Tuple[float, float, float] = (0.08, 0.500, 0.30)  # [width, depth, height]
    wall_pos: Tuple[float, float, float] = (0.55, 0.0, -0.15)     # Wall center position
    wall_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Quaternion rotation


@dataclass
class PathExecutionConfig:
    """Unified configuration for path execution in both sim and real systems."""
    
    # Motion parameters
    speed_mps: float = 0.02  # Speed in meters per second
    interpolation_factor: int = 10  # Points added between A* waypoints (higher = smoother)
    update_frequency: float = 60.0  # Hz - target update frequency. #TODO: this sets Sim dt /irl control loop, should we run the sim physics faster?
    
    # A* path planning parameters
    grid_resolution: float = 0.02  # A* grid cell size in meters
    safety_margin: float = 0.05  # Obstacle safety margin in meters.
    
    # Final target verification. No new *end* target point will be given until these are met.
    # Note: this does not affect the points between endpoints, these are followed blindly ("trying to keep up")
    final_target_tolerance: float = 0.010  # Final target reaching tolerance in meters (1.0cm)
    orientation_tolerance: float = 0.07  # Orientation tolerance in radians (~4 degrees)
    
    # Optional progress feedback
    progress_update_interval: float = 2.0  # How often to print progress (seconds)
    
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.speed_mps > 0, "Speed must be positive"
        assert self.interpolation_factor >= 1, "Interpolation factor must be at least 1"
        assert 0 < self.final_target_tolerance <= 0.1, "Target tolerance should be 1-10cm"
        assert self.update_frequency > 0, "Update frequency must be positive"
        assert 0.001 <= self.grid_resolution <= 0.1, "Grid resolution should be 1mm-10cm"
        assert 0.01 <= self.safety_margin <= 0.2, "Safety margin should be 1cm-20cm"


def interpolate_path(path: np.ndarray, interpolation_factor: int) -> np.ndarray:
    """
    Insert interpolated points between A* waypoints for smoother motion.
    
    Args:
        path: A* waypoints as np.ndarray of shape [N, 3]
        interpolation_factor: Number of points to add between each waypoint pair
        
    Returns:
        Smoothed path with interpolated points
    """
    # direct path
    if len(path) < 2:
        return path
    
    smooth_path = [path[0]]  # Start point
    
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]
        
        # Add interpolated points between waypoints
        for j in range(1, interpolation_factor + 1):
            t = j / (interpolation_factor + 1)
            interpolated = current + t * (next_point - current)
            smooth_path.append(interpolated)
        
        smooth_path.append(next_point)
    
    return np.array(smooth_path)


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
    Get position along path based on progress (0.0 to 1.0). Mainly for irl debugging
    
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


def calculate_execution_time(path: np.ndarray, speed_mps: float) -> float:
    """
    Calculate estimated execution time for a path. Mainly for matching sim and irl
    
    Args:
        path: Path waypoints as np.ndarray of shape [N, 3]  
        speed_mps: Speed in meters per second
        
    Returns:
        Estimated execution time in seconds
    """
    total_distance = calculate_path_length(path)
    return total_distance / speed_mps if speed_mps > 0 else 0.0


def is_target_reached(current_pos: np.ndarray, target_pos: np.ndarray, 
                     config: PathExecutionConfig) -> bool:
    """
    Check if target position is reached within tolerance.
    
    Args:
        current_pos: Current position [x, y, z]
        target_pos: Target position [x, y, z]
        config: Path execution configuration
        
    Returns:
        True if target is reached within tolerance
    """
    distance = np.linalg.norm(current_pos - target_pos)
    return distance < config.final_target_tolerance


def print_path_info(path: np.ndarray, config: PathExecutionConfig, label: str = "Path"):
    """
    Print detailed path information for debugging.
    
    Args:
        path: Path waypoints
        config: Configuration used
        label: Label for the path (e.g., "A* Path", "Smooth Path")
    """
    length = calculate_path_length(path)
    estimated_time = calculate_execution_time(path, config.speed_mps)
    avg_spacing = length / (len(path) - 1) if len(path) > 1 else 0
    
    print(f"[{label}] {len(path)} waypoints, {length:.3f}m total")
    print(f"[{label}] Speed: {config.speed_mps:.3f}m/s, Est. time: {estimated_time:.1f}s")
    print(f"[{label}] Avg spacing: {avg_spacing:.4f}m")


# Default configuration instances for quick access
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_CONFIG = PathExecutionConfig()