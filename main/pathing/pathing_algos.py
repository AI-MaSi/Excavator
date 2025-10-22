"""
Pathing algos, converted from original to run with Numpy. Used in Raspberry Pi 5 for path planning.
"""

import heapq
import math
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for the A* grid."""
    resolution: float = 0.01  # Grid cell size in meters
    bounds_min: Tuple[float, float, float] = (0.1, -1.0, -0.1)  # (x_min, y_min, z_min)
    bounds_max: Tuple[float, float, float] = (0.8, 1.0, 0.5)   # (x_max, y_max, z_max)
    safety_margin: float = 0.04  # Additional clearance around obstacles


class ObstacleChecker:
    """Handles obstacle collision detection for path planning."""

    def __init__(self, obstacle_data: List[Dict[str, Any]], safety_margin: float = 0.02):
        """
        Initialize obstacle checker.

        Args:
            obstacle_data: List of obstacle dictionaries with keys:
                - "size": np.array([x, y, z]) - obstacle dimensions
                - "pos": np.array([x, y, z]) - obstacle center position
                - "rot": np.array([w, x, y, z]) - quaternion rotation
            safety_margin: Additional clearance around obstacles
        """
        self.obstacles = obstacle_data
        self.safety_margin = safety_margin

        # Pre-compute axis-aligned bounding boxes for efficiency
        self._compute_aabbs()


    def _compute_aabbs(self):
        """Pre-compute axis-aligned bounding boxes for all obstacles."""
        self.aabbs = []

        for obs in self.obstacles:
            size = obs["size"] + self.safety_margin * 2  # Add safety margin
            pos = obs["pos"]

            # For rotated obstacles, we need to compute the AABB that encompasses
            # the entire rotated bounding box
            if np.allclose(obs["rot"], [1, 0, 0, 0]):  # No rotation
                min_bounds = pos - size / 2
                max_bounds = pos + size / 2
            else:
                # Compute oriented bounding box corners
                corners = self._get_box_corners(size, pos, obs["rot"])
                min_bounds = np.min(corners, axis=0)
                max_bounds = np.max(corners, axis=0)

            self.aabbs.append({
                "min": min_bounds,
                "max": max_bounds,
                "detailed": obs  # Keep reference for detailed collision checking
            })

    def _get_box_corners(self, size: np.ndarray, position: np.ndarray,
                        quaternion: np.ndarray) -> np.ndarray:
        """Get the 8 corners of a rotated box."""
        # Create local corners (before rotation)
        half_size = size / 2
        local_corners = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
        ])

        # Convert quaternion to rotation matrix
        rot_matrix = self._quaternion_to_rotation_matrix(quaternion)

        # Apply rotation and translation
        world_corners = np.array([rot_matrix @ corner + position for corner in local_corners])

        return world_corners

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q

        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

        return R

    def is_point_collision_free(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is collision-free."""
        point_array = np.array(point)

        for aabb in self.aabbs:
            # Quick AABB check first
            if np.all(point_array >= aabb["min"]) and np.all(point_array <= aabb["max"]):
                # Detailed check for rotated obstacles
                if not self._point_in_oriented_box(point_array, aabb["detailed"]):
                    continue
                return False

        return True

    def _point_in_oriented_box(self, point: np.ndarray, obstacle: Dict[str, Any]) -> bool:
        """Check if point is inside an oriented bounding box."""
        # For axis-aligned boxes (no rotation), use simple bounds check
        if np.allclose(obstacle["rot"], [1, 0, 0, 0], atol=1e-3):
            half_size = (obstacle["size"] + self.safety_margin * 2) / 2
            min_bounds = obstacle["pos"] - half_size
            max_bounds = obstacle["pos"] + half_size
            return np.all(point >= min_bounds) and np.all(point <= max_bounds)

        # For rotated boxes, transform point to local coordinates
        rot_matrix = self._quaternion_to_rotation_matrix(obstacle["rot"])
        local_point = rot_matrix.T @ (point - obstacle["pos"])
        half_size = (obstacle["size"] + self.safety_margin * 2) / 2

        return np.all(np.abs(local_point) <= half_size)

    def is_line_collision_free(self, start: Tuple[float, float, float],
                              end: Tuple[float, float, float],
                              num_samples: int = 10) -> bool:
        """Check if a line segment is collision-free by sampling points."""
        start_array = np.array(start)
        end_array = np.array(end)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = start_array + t * (end_array - start_array)
            if not self.is_point_collision_free(tuple(sample_point)):
                return False

        return True


class AStar3D:
    """3D A* pathfinding algorithm with obstacle avoidance."""

    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker,
                 use_3d: bool = True):
        """
        Initialize 3D A* planner.

        Args:
            grid_config: Grid configuration
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d

        # Cache for grid conversions
        self._world_to_grid_cache = {}
        self._grid_to_world_cache = {}

    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        cache_key = world_pos
        if cache_key in self._world_to_grid_cache:
            return self._world_to_grid_cache[cache_key]

        x = int(round((world_pos[0] - self.grid_config.bounds_min[0]) / self.grid_config.resolution))
        y = int(round((world_pos[1] - self.grid_config.bounds_min[1]) / self.grid_config.resolution))
        z = int(round((world_pos[2] - self.grid_config.bounds_min[2]) / self.grid_config.resolution))

        result = (x, y, z)
        self._world_to_grid_cache[cache_key] = result
        return result

    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid indices to world coordinates."""
        cache_key = grid_pos
        if cache_key in self._grid_to_world_cache:
            return self._grid_to_world_cache[cache_key]

        x = grid_pos[0] * self.grid_config.resolution + self.grid_config.bounds_min[0]
        y = grid_pos[1] * self.grid_config.resolution + self.grid_config.bounds_min[1]
        z = grid_pos[2] * self.grid_config.resolution + self.grid_config.bounds_min[2]

        result = (x, y, z)
        self._grid_to_world_cache[cache_key] = result
        return result

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1]) if self.use_3d else 0
        dz = abs(a[2] - b[2])
        return math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution

    def get_neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], float]]:
        """Get valid neighbors of a node with their costs."""
        neighbors = []
        x, y, z = node

        if self.use_3d:
            # 26-connectivity in 3D
            directions = [
                # Face neighbors (6)
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                # Edge neighbors (12)
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                # Corner neighbors (8)
                (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
            ]
        else:
            # 8-connectivity in X-Z plane (Y fixed)
            directions = [
                (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),  # 4-connectivity
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)  # Diagonal
            ]

        for dx, dy, dz in directions:
            new_x, new_y, new_z = x + dx, y + dy, z + dz

            # Check bounds
            max_x = int((self.grid_config.bounds_max[0] - self.grid_config.bounds_min[0]) / self.grid_config.resolution)
            max_y = int((self.grid_config.bounds_max[1] - self.grid_config.bounds_min[1]) / self.grid_config.resolution)
            max_z = int((self.grid_config.bounds_max[2] - self.grid_config.bounds_min[2]) / self.grid_config.resolution)

            if not (0 <= new_x <= max_x and 0 <= new_y <= max_y and 0 <= new_z <= max_z):
                continue

            neighbor = (new_x, new_y, new_z)
            world_pos = self.grid_to_world(neighbor)

            # Check collision
            if not self.obstacle_checker.is_point_collision_free(world_pos):
                continue

            # Calculate cost (Euclidean distance)
            cost = math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution
            neighbors.append((neighbor, cost))

        return neighbors

    def plan_path(self, start: Tuple[float, float, float],
                  goal: Tuple[float, float, float],
                  max_iterations: int = 200000) -> List[Tuple[float, float, float]]:
        """
        Plan a path from start to goal using A*.

        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            max_iterations: Maximum search iterations

        Returns:
            List of waypoints in world coordinates, or empty list if no path found
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            print(f"[A* 3D] Start position {start} is in collision!")
            return []

        if not self.obstacle_checker.is_point_collision_free(goal):
            print(f"[A* 3D] Goal position {goal} is in collision!")
            return []

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start_grid, goal_grid), 0, start_grid))

        came_from = {}
        cost_so_far = {start_grid: 0}

        iteration = 0
        print(f"[A* 3D] Starting search from {start} to {goal}")
        print(f"[A* 3D] Grid start: {start_grid}, Grid goal: {goal_grid}")

        while open_set and iteration < max_iterations:
            iteration += 1

            if iteration % 5000 == 0:
                print(f"[A* 3D] Iteration {iteration}, open set size: {len(open_set)}")

            _, cost, current = heapq.heappop(open_set)

            # Check if we reached the goal
            if current == goal_grid:
                print(f"[A* 3D] Path found after {iteration} iterations!")
                break

            # Explore neighbors
            for neighbor, move_cost in self.get_neighbors(current):
                new_cost = cost + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        else:
            if iteration >= max_iterations:
                print(f"[A* 3D] Max iterations ({max_iterations}) reached!")
            else:
                print(f"[A* 3D] No path found after {iteration} iterations!")
            return []

        # Reconstruct path
        path_grid = []
        current = goal_grid
        while current is not None:
            path_grid.append(current)
            current = came_from.get(current)
        path_grid.reverse()

        # Convert to world coordinates
        path_world = [self.grid_to_world(grid_pos) for grid_pos in path_grid]

        print(f"[A* 3D] Path reconstructed with {len(path_world)} waypoints")
        return path_world

def create_astar_3d_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float = 0.01,
                              safety_margin: float = 0.02,
                              use_3d: bool = True) -> np.ndarray:
    """
    High-level interface to create A* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries (format as specified)
        grid_resolution: Grid cell size in meters
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create A* planner
    planner = AStar3D(grid_config, obstacle_checker, use_3d=use_3d)

    # Plan path
    path = planner.plan_path(start_pos, goal_pos)

    if not path:
        raise RuntimeError("A* 3D failed to find a path")

    # Convert to NumPy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[A* 3D] Created trajectory with {len(path)} waypoints")
    print(f"[A* 3D] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[A* 3D] Grid resolution: {grid_resolution}m, Safety margin: {safety_margin}m")
    print(f"[A* 3D] Workspace bounds: {grid_config.bounds_min} to {grid_config.bounds_max}")

    return path_array