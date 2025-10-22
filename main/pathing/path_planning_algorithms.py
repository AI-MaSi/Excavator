"""
Enhanced 3D A* Path Planning Algorithm with Obstacle Avoidance
==============================================================

This module provides a 3D A* pathfinding algorithm that can handle obstacle avoidance
for robotic path planning applications. It supports both 2D (X-Z plane) and full 3D
path planning with configurable grid resolution and obstacle checking.

Features:
- Full 3D pathfinding capability
- Obstacle avoidance using bounding box collision detection
- Configurable grid resolution for different precision needs
- Support for rotated obstacles (quaternion-based)
- Efficient neighbor generation with 26-connectivity in 3D
- Path smoothing and interpolation capabilities
- Compatible with Isaac Sim obstacle format

Author: AI Assistant
Date: 2025
"""

import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
from typing import Set
import time

@dataclass
class GridConfig:
    """Configuration for the A* grid."""
    resolution: float = 0.01  # Grid cell size in meters
    bounds_min: Tuple[float, float, float] = (0.0, -1, 0.1)  # (x_min, y_min, z_min)
    bounds_max: Tuple[float, float, float] = (1.0, 1, 2.0)   # (x_max, y_max, z_max)
    safety_margin: float = 0.02  # Additional clearance around obstacles


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
        # CHANGED: Use tolerance-based comparison instead of exact match
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
        
        # if self.use_3d:
        #     # 6-connectivity only - orthogonal directions (no diagonals)
        #     directions = [
        #         (1, 0, 0), (-1, 0, 0),    # Forward/backward
        #         (0, 1, 0), (0, -1, 0),    # Left/right  
        #         (0, 0, 1), (0, 0, -1)     # Up/down
        #     ]
        # else:
        #     # 4-connectivity in X-Z plane (no diagonals)
        #     directions = [
        #         (1, 0, 0), (-1, 0, 0),    # Forward/backward
        #         (0, 0, 1), (0, 0, -1)     # Up/down
        #     ]
        
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
            print(f"[A*] Start position {start} is in collision!")
            return []
        
        if not self.obstacle_checker.is_point_collision_free(goal):
            print(f"[A*] Goal position {goal} is in collision!")
            return []
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start_grid, goal_grid), 0, start_grid))
        
        came_from = {}
        cost_so_far = {start_grid: 0}
        
        iteration = 0
        print(f"[A*] Starting search from {start} to {goal}")
        print(f"[A*] Grid start: {start_grid}, Grid goal: {goal_grid}")
        
        while open_set and iteration < max_iterations:
            iteration += 1
            
            if iteration % 5000 == 0:
                print(f"[A*] Iteration {iteration}, open set size: {len(open_set)}")
            
            _, cost, current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current == goal_grid:
                print(f"[A*] Path found after {iteration} iterations!")
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
                print(f"[A*] Max iterations ({max_iterations}) reached!")
            else:
                print(f"[A*] No path found after {iteration} iterations!")
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
        
        print(f"[A*] Path reconstructed with {len(path_world)} waypoints")
        return path_world
    
def create_astar_3d_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float,
                              safety_margin: float,
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
        NumPy array of shape [N, 3] containing path waypoints (float32)
    """
    # Set up grid bounds based on typical workspace
    if len(obstacle_data) > 0:
        # Determine bounds from obstacles and start/goal
        all_positions = [obs["pos"] for obs in obstacle_data] + [list(start_pos), list(goal_pos)]
        all_positions = np.array(all_positions)
        
        bounds_min = np.min(all_positions, axis=0) - 0.1
        bounds_max = np.max(all_positions, axis=0) + 0.1
        
        # Ensure minimum workspace size
        bounds_min = np.minimum(bounds_min, [0.34, -0.36, 0.0])
        bounds_max = np.maximum(bounds_max, [0.85, 0.36, 0.78])
    else:
        # Default bounds
        bounds_min = (0.34, -0.36, 0.0)
        bounds_max = (0.85, 0.36, 0.78)
    
    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
        safety_margin=safety_margin
    )
    
    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)
    
    # Create A* planner
    planner = AStar3D(grid_config, obstacle_checker, use_3d=use_3d)
    
    # Plan path
    path = planner.plan_path(start_pos, goal_pos)
    
    if not path:
        raise RuntimeError("A* failed to find a path")

    # Convert to NumPy array (float32 for numba compatibility)
    path_array = np.array(path, dtype=np.float32)

    print(f"[A*] Created trajectory with {len(path)} waypoints")
    print(f"[A*] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[A*] Grid resolution: {grid_resolution}m, Safety margin: {safety_margin}m")
    print(f"[A*] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array

"""
RRT* and RRT Algorithm Implementation for 3D Path Planning
=================================================
"""

@dataclass
class RRTNode:
    """Node for RRT* tree."""
    position: Tuple[float, float, float]
    parent: Optional['RRTNode'] = None
    children: List['RRTNode'] = None
    cost: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class RRTStar:
    """RRT* (Rapidly-exploring Random Tree Star) pathfinding algorithm with obstacle avoidance."""
    
    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker, 
                 use_3d: bool = True):
        """
        Initialize RRT* planner.
        
        Args:
            grid_config: Grid configuration for bounds and resolution
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d
        
        # RRT* specific parameters
        self.max_step_size = 0.05  # Maximum distance for extending tree
        self.goal_bias = 0.1  # Probability of sampling toward goal
        self.rewire_radius = 0.08  # Radius for rewiring optimization
        self.goal_tolerance = 0.02  # Distance to consider goal reached
        
        # Early termination parameters
        self.max_acceptable_cost = None  # Direct cost threshold in meters
        self.cost_improvement_patience = 5000  # Iterations to wait for cost improvement
        self.minimum_iterations = 1000  # Minimum iterations before early termination
        
    def should_terminate_early(self, goal_node: Optional[RRTNode], iteration: int, 
                          last_improvement_iteration: int) -> bool:
        """
        Check if we should terminate early based on cost threshold.
        
        Args:
            goal_node: Current best goal node (None if no path found yet)
            iteration: Current iteration number
            last_improvement_iteration: Iteration when cost was last improved
            
        Returns:
            True if should terminate early, False otherwise
        """
        # Don't terminate before minimum iterations
        if iteration < self.minimum_iterations:
            return False
            
        # No goal found yet
        if goal_node is None:
            return False
            
        # Check if we have an acceptable cost
        if (self.max_acceptable_cost is not None and 
            goal_node.cost <= self.max_acceptable_cost):
            print(f"[RRT*] Early termination: Acceptable cost {goal_node.cost:.3f}m <= {self.max_acceptable_cost:.3f}m")
            return True
            
        # Check if we haven't improved in a while
        if iteration - last_improvement_iteration > self.cost_improvement_patience:
            print(f"[RRT*] Early termination: No improvement for {self.cost_improvement_patience} iterations")
            return True
            
        return False    
        
    def sample_random_point(self, goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Sample a random point in the configuration space."""
        # Goal biasing: occasionally sample near the goal
        if random.random() < self.goal_bias:
            # Sample in small sphere around goal
            offset_radius = 0.05
            if self.use_3d:
                offset = np.array([
                    random.uniform(-offset_radius, offset_radius),
                    random.uniform(-offset_radius, offset_radius),
                    random.uniform(-offset_radius, offset_radius)
                ])
            else:
                offset = np.array([
                    random.uniform(-offset_radius, offset_radius),
                    0.0,  # Keep Y fixed for 2D planning
                    random.uniform(-offset_radius, offset_radius)
                ])
            
            sampled_point = np.array(goal) + offset
            # Clamp to bounds
            sampled_point = np.clip(
                sampled_point,
                self.grid_config.bounds_min,
                self.grid_config.bounds_max
            )
            return tuple(sampled_point)
        
        # Regular uniform sampling
        if self.use_3d:
            return (
                random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                random.uniform(self.grid_config.bounds_min[1], self.grid_config.bounds_max[1]),
                random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
            )
        else:
            # 2D planning in X-Z plane
            y_fixed = (self.grid_config.bounds_min[1] + self.grid_config.bounds_max[1]) / 2
            return (
                random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                y_fixed,
                random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
            )
    
    def find_nearest_node(self, tree: List[RRTNode], point: Tuple[float, float, float]) -> RRTNode:
        """Find the nearest node in the tree to the given point."""
        min_distance = float('inf')
        nearest_node = None
        
        point_array = np.array(point)
        for node in tree:
            node_array = np.array(node.position)
            distance = np.linalg.norm(point_array - node_array)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def steer(self, from_pos: Tuple[float, float, float], 
              to_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Steer from one position toward another, limited by max step size."""
        from_array = np.array(from_pos)
        to_array = np.array(to_pos)
        
        direction = to_array - from_array
        distance = np.linalg.norm(direction)
        
        if distance <= self.max_step_size:
            return to_pos
        
        # Limit to max step size
        unit_direction = direction / distance
        new_position = from_array + unit_direction * self.max_step_size
        
        return tuple(new_position)
    
    def find_near_nodes(self, tree: List[RRTNode], point: Tuple[float, float, float], 
                       radius: float) -> List[RRTNode]:
        """Find all nodes within a given radius of the point."""
        near_nodes = []
        point_array = np.array(point)
        
        for node in tree:
            node_array = np.array(node.position)
            if np.linalg.norm(point_array - node_array) <= radius:
                near_nodes.append(node)
        
        return near_nodes
    
    def calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def rewire_tree(self, tree: List[RRTNode], new_node: RRTNode, near_nodes: List[RRTNode]):
        """Rewire the tree to optimize paths through the new node."""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            
            # Calculate potential new cost through new_node
            potential_cost = new_node.cost + self.calculate_distance(new_node.position, near_node.position)
            
            # If this path is better and collision-free, rewire
            if (potential_cost < near_node.cost and 
                self.obstacle_checker.is_line_collision_free(new_node.position, near_node.position)):
                
                # Remove old parent connection
                if near_node.parent:
                    near_node.parent.children.remove(near_node)
                
                # Establish new parent connection
                near_node.parent = new_node
                new_node.children.append(near_node)
                
                # Update cost and propagate to descendants
                old_cost = near_node.cost
                near_node.cost = potential_cost
                self._propagate_cost_update(near_node, near_node.cost - old_cost)
    
    def _propagate_cost_update(self, node: RRTNode, cost_delta: float):
        """Recursively update costs of all descendants."""
        for child in node.children:
            child.cost += cost_delta
            self._propagate_cost_update(child, cost_delta)
    
    def choose_parent(self, near_nodes: List[RRTNode], 
                     new_position: Tuple[float, float, float]) -> Tuple[RRTNode, float]:
        """Choose the best parent from near nodes to minimize cost."""
        best_parent = None
        best_cost = float('inf')
        
        for node in near_nodes:
            potential_cost = node.cost + self.calculate_distance(node.position, new_position)
            
            if (potential_cost < best_cost and 
                self.obstacle_checker.is_line_collision_free(node.position, new_position)):
                best_parent = node
                best_cost = potential_cost
        
        return best_parent, best_cost
    
    def extract_path(self, goal_node: RRTNode) -> List[Tuple[float, float, float]]:
        """Extract path from start to goal by following parent pointers."""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def plan_path(self, start: Tuple[float, float, float], 
                  goal: Tuple[float, float, float],
                  max_iterations: int = 5000,
                  max_acceptable_cost: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """
        Plan a path from start to goal using RRT*.
        
        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            max_iterations: Maximum planning iterations
            
        Returns:
            List of waypoints in world coordinates, or empty list if no path found
        """
        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            print(f"[RRT*] Start position {start} is in collision!")
            return []
        
        if not self.obstacle_checker.is_point_collision_free(goal):
            print(f"[RRT*] Goal position {goal} is in collision!")
            return []
        
        # Set cost threshold
        self.max_acceptable_cost = max_acceptable_cost

        # Calculate straight-line distance for reference
        straight_line_distance = self.calculate_distance(start, goal)
        
        # Initialize tree with start node
        start_node = RRTNode(position=start, cost=0.0)
        tree = [start_node]
        goal_node = None
        
        last_improvement_iteration = 0
        
        print(f"[RRT*] Starting planning from {start} to {goal}")
        print(f"[RRT*] Straight-line distance: {straight_line_distance:.3f}m")
        print(f"[RRT*] Max acceptable cost: {max_acceptable_cost}m" if max_acceptable_cost else "[RRT*] No cost limit set")
        print(f"[RRT*] Max iterations: {max_iterations}")
        
        for iteration in range(max_iterations):
            if iteration % 500 == 0 and iteration > 0:
                print(f"[RRT*] Iteration {iteration}, tree size: {len(tree)}")
            
            # Sample random point
            rand_point = self.sample_random_point(goal)
            
            # Find nearest node
            nearest_node = self.find_nearest_node(tree, rand_point)
            
            # Steer toward random point
            new_position = self.steer(nearest_node.position, rand_point)
            
            # Check if new position is collision-free
            if not self.obstacle_checker.is_point_collision_free(new_position):
                continue
            
            # Check if path to new position is collision-free
            if not self.obstacle_checker.is_line_collision_free(nearest_node.position, new_position):
                continue
            
            # Find near nodes for optimization
            near_nodes = self.find_near_nodes(tree, new_position, self.rewire_radius)
            
            # Choose best parent (RRT* optimization)
            best_parent, best_cost = self.choose_parent(near_nodes, new_position)
            
            if best_parent is None:
                # Fallback to nearest node
                best_parent = nearest_node
                best_cost = nearest_node.cost + self.calculate_distance(nearest_node.position, new_position)
            
            # Create new node
            new_node = RRTNode(position=new_position, parent=best_parent, cost=best_cost)
            best_parent.children.append(new_node)
            tree.append(new_node)
            
            # Rewire tree (RRT* optimization)
            self.rewire_tree(tree, new_node, near_nodes)
            
            # Check if we reached the goal
            # if self.calculate_distance(new_position, goal) <= self.goal_tolerance:
            #     if goal_node is None or new_node.cost < goal_node.cost:
            #         goal_node = new_node
            #         print(f"[RRT*] Goal reached at iteration {iteration}, cost: {goal_node.cost:.3f}")
            if self.calculate_distance(new_position, goal) <= self.goal_tolerance:
                if goal_node is None or new_node.cost < goal_node.cost:
                    old_cost = goal_node.cost if goal_node else float('inf')
                    goal_node = new_node
                    last_improvement_iteration = iteration
                    print(f"[RRT*] Goal reached at iteration {iteration}, cost: {goal_node.cost:.3f}m (improved from {old_cost:.3f}m)")

            # Check for early termination
            if self.should_terminate_early(goal_node, iteration, last_improvement_iteration):
                print(f"[RRT*] Early termination at iteration {iteration}")
                break
        
        if goal_node is None:
            print(f"[RRT*] No path found after {max_iterations} iterations!")
            return []
        
        # Extract and return path
        path = self.extract_path(goal_node)
        print(f"[RRT*] Path found with {len(path)} waypoints, total cost: {goal_node.cost:.3f}")
        
        return path

class RRT(RRTStar):
    def rewire_tree(self, tree, new_node, near_nodes):
        pass  # No rewiring in basic RRT
    
    def choose_parent(self, near_nodes, new_position):
        # Just use nearest node, no cost optimization
        return near_nodes[0] if near_nodes else None, 0.0

def create_rrt_star_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float,
                              safety_margin: float,
                              max_iterations: int,
                              use_3d: bool = True,
                              max_acceptable_cost: Optional[float] = None,
                              # Optional RRT* tuning
                              max_step_size: Optional[float] = None,
                              goal_bias: Optional[float] = None,
                              rewire_radius: Optional[float] = None,
                              goal_tolerance: Optional[float] = None,
                              minimum_iterations: Optional[int] = None,
                              cost_improvement_patience: Optional[int] = None) -> np.ndarray:
    """
    High-level interface to create RRT* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        max_iterations: Maximum RRT* iterations
        use_3d: Use full 3D planning or just X-Z plane
        max_acceptable_cost: Optional cost threshold for early termination

    Returns:
        NumPy array of shape [N, 3] containing path waypoints (float32)
    """
    # Set up grid bounds based on obstacles and start/goal
    if len(obstacle_data) > 0:
        all_positions = [obs["pos"] for obs in obstacle_data] + [list(start_pos), list(goal_pos)]
        all_positions = np.array(all_positions)
        
        bounds_min = np.min(all_positions, axis=0) - 0.1
        bounds_max = np.max(all_positions, axis=0) + 0.1
        
        # Ensure minimum workspace size
        bounds_min = np.minimum(bounds_min, [0.34, -0.36, 0.0])
        bounds_max = np.maximum(bounds_max, [0.85, 0.36, 0.78])
    else:
        # Default bounds
        bounds_min = (0.34, -0.36, 0.0)
        bounds_max = (0.85, 0.36, 0.78)
    
    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
        safety_margin=safety_margin
    )
    
    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)
    
    # Create RRT* planner
    planner = RRTStar(grid_config, obstacle_checker, use_3d=use_3d)
    # Apply optional tuning parameters if provided
    if max_step_size is not None:
        planner.max_step_size = max_step_size
    if goal_bias is not None:
        planner.goal_bias = goal_bias
    if rewire_radius is not None:
        planner.rewire_radius = rewire_radius
    if goal_tolerance is not None:
        planner.goal_tolerance = goal_tolerance
    if minimum_iterations is not None:
        planner.minimum_iterations = minimum_iterations
    if cost_improvement_patience is not None:
        planner.cost_improvement_patience = cost_improvement_patience

    # Plan path
    path = planner.plan_path(start_pos, goal_pos, max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT* failed to find a path")

    # Convert to NumPy array (float32 for numba compatibility)
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT*] Created trajectory with {len(path)} waypoints")
    print(f"[RRT*] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT*] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT*] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array

def create_rrt_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float,
                              safety_margin: float,
                              max_iterations: int,
                              use_3d: bool = True,
                              max_acceptable_cost: Optional[float] = None,
                              # Optional RRT tuning
                              max_step_size: Optional[float] = None,
                              goal_bias: Optional[float] = None,
                              rewire_radius: Optional[float] = None,
                              goal_tolerance: Optional[float] = None,
                              minimum_iterations: Optional[int] = None,
                              cost_improvement_patience: Optional[int] = None) -> np.ndarray:
    """
    High-level interface to create RRT trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        max_iterations: Maximum RRT iterations
        use_3d: Use full 3D planning or just X-Z plane
        max_acceptable_cost: Optional cost threshold for early termination

    Returns:
        NumPy array of shape [N, 3] containing path waypoints (float32)
    """
    # Set up grid bounds based on obstacles and start/goal
    if len(obstacle_data) > 0:
        all_positions = [obs["pos"] for obs in obstacle_data] + [list(start_pos), list(goal_pos)]
        all_positions = np.array(all_positions)
        
        bounds_min = np.min(all_positions, axis=0) - 0.1
        bounds_max = np.max(all_positions, axis=0) + 0.1
        
        # Ensure minimum workspace size
        bounds_min = np.minimum(bounds_min, [0.34, -0.36, 0.0])
        bounds_max = np.maximum(bounds_max, [0.85, 0.36, 0.78])
    else:
        # Default bounds
        bounds_min = (0.34, -0.36, 0.0)
        bounds_max = (0.85, 0.36, 0.78)
    
    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
        safety_margin=safety_margin
    )
    
    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)
    
    # Create RRT planner
    planner = RRT(grid_config, obstacle_checker, use_3d=use_3d)
    # Apply optional tuning parameters if provided
    if max_step_size is not None:
        planner.max_step_size = max_step_size
    if goal_bias is not None:
        planner.goal_bias = goal_bias
    if rewire_radius is not None:
        planner.rewire_radius = rewire_radius
    if goal_tolerance is not None:
        planner.goal_tolerance = goal_tolerance
    if minimum_iterations is not None:
        planner.minimum_iterations = minimum_iterations
    if cost_improvement_patience is not None:
        planner.cost_improvement_patience = cost_improvement_patience

    # Plan path
    path = planner.plan_path(start_pos, goal_pos, max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT failed to find a path")

    # Convert to NumPy array (float32 for numba compatibility)
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT] Created trajectory with {len(path)} waypoints")
    print(f"[RRT] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


"""
PRM Algorithm Implementation for 3D Path Planning
=================================================
"""

class PRMNode:
    """Node in the PRM roadmap graph."""
    
    def __init__(self, node_id: int, position: Tuple[float, float, float]):
        self.id = node_id
        self.position = position
        self.neighbors: Set[int] = set()
        self.distances: Dict[int, float] = {}  # Distance to each neighbor
    
    def add_neighbor(self, neighbor_id: int, distance: float):
        """Add a bidirectional connection to another node."""
        self.neighbors.add(neighbor_id)
        self.distances[neighbor_id] = distance


class PRMRoadmap:
    """Probabilistic Roadmap graph structure."""
    
    def __init__(self):
        self.nodes: Dict[int, PRMNode] = {}
        self.next_node_id = 0
        
    def add_node(self, position: Tuple[float, float, float]) -> int:
        """Add a new node to the roadmap."""
        node_id = self.next_node_id
        self.nodes[node_id] = PRMNode(node_id, position)
        self.next_node_id += 1
        return node_id
    
    def add_edge(self, node1_id: int, node2_id: int, distance: float):
        """Add bidirectional edge between two nodes."""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id].add_neighbor(node2_id, distance)
            self.nodes[node2_id].add_neighbor(node1_id, distance)
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Get neighbors of a node with their distances."""
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        return [(neighbor_id, node.distances[neighbor_id]) 
                for neighbor_id in node.neighbors]
    
    def dijkstra(self, start_id: int, goal_id: int) -> List[int]:
        """Find shortest path between two nodes using Dijkstra's algorithm."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return []
        
        # Priority queue: (distance, node_id)
        pq = [(0.0, start_id)]
        distances = {start_id: 0.0}
        previous = {}
        visited = set()
        
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            if current_id == goal_id:
                # Reconstruct path
                path = []
                node_id = goal_id
                while node_id is not None:
                    path.append(node_id)
                    node_id = previous.get(node_id)
                path.reverse()
                return path
            
            # Check neighbors
            for neighbor_id, edge_distance in self.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue
                
                new_distance = current_dist + edge_distance
                
                if neighbor_id not in distances or new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_distance, neighbor_id))
        
        return []  # No path found


class PRM:
    """Probabilistic Roadmap pathfinding algorithm with obstacle avoidance."""
    
    def __init__(self, grid_config: GridConfig, obstacle_checker: ObstacleChecker, 
                 use_3d: bool = True):
        """
        Initialize PRM planner.
        
        Args:
            grid_config: Grid configuration for workspace bounds
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d
        
        # PRM-specific parameters
        self.num_samples = 1000  # Number of random samples for roadmap
        self.connection_radius = 0.12  # Max distance for connecting nodes
        self.max_connections_per_node = 15  # Limit connections for efficiency
        
        # Roadmap storage
        self.roadmap = PRMRoadmap()
        self.roadmap_built = False
        self.construction_time = 0.0
        
    def sample_random_point(self) -> Tuple[float, float, float]:
        """Sample a random collision-free point in the workspace."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            if self.use_3d:
                point = (
                    random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                    random.uniform(self.grid_config.bounds_min[1], self.grid_config.bounds_max[1]),
                    random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
                )
            else:
                # 2D planning in X-Z plane
                y_fixed = (self.grid_config.bounds_min[1] + self.grid_config.bounds_max[1]) / 2
                point = (
                    random.uniform(self.grid_config.bounds_min[0], self.grid_config.bounds_max[0]),
                    y_fixed,
                    random.uniform(self.grid_config.bounds_min[2], self.grid_config.bounds_max[2])
                )
            
            if self.obstacle_checker.is_point_collision_free(point):
                return point
        
        # If we can't find a collision-free point, return a corner of workspace
        return self.grid_config.bounds_min
    
    def calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def find_nearby_nodes(self, position: Tuple[float, float, float], 
                         max_distance: float) -> List[Tuple[int, float]]:
        """Find all nodes within max_distance of the given position."""
        nearby_nodes = []
        
        for node_id, node in self.roadmap.nodes.items():
            distance = self.calculate_distance(position, node.position)
            if distance <= max_distance:
                nearby_nodes.append((node_id, distance))
        
        # Sort by distance and limit connections
        nearby_nodes.sort(key=lambda x: x[1])
        return nearby_nodes[:self.max_connections_per_node]
    
    def construct_roadmap(self, verbose: bool = True):
        """Construct the PRM roadmap through sampling and connection."""
        if self.roadmap_built:
            return
            
        start_time = time.time()
        
        if verbose:
            print(f"[PRM] Constructing roadmap with {self.num_samples} samples...")
        
        # Phase 1: Sampling
        valid_samples = 0
        for i in range(self.num_samples * 3):  # Allow more attempts
            if valid_samples >= self.num_samples:
                break
                
            sample_point = self.sample_random_point()
            if self.obstacle_checker.is_point_collision_free(sample_point):
                self.roadmap.add_node(sample_point)
                valid_samples += 1
                
                if verbose and valid_samples % 200 == 0:
                    print(f"[PRM] Sampled {valid_samples}/{self.num_samples} collision-free points")
        
        if verbose:
            print(f"[PRM] Successfully sampled {valid_samples} collision-free points")
        
        # Phase 2: Connection
        connections_made = 0
        for node_id, node in self.roadmap.nodes.items():
            nearby_nodes = self.find_nearby_nodes(node.position, self.connection_radius)
            
            for nearby_id, distance in nearby_nodes:
                if nearby_id != node_id and nearby_id not in node.neighbors:
                    # Check if connection is collision-free
                    if self.obstacle_checker.is_line_collision_free(node.position, 
                                                                  self.roadmap.nodes[nearby_id].position):
                        self.roadmap.add_edge(node_id, nearby_id, distance)
                        connections_made += 1
            
            if verbose and node_id % 100 == 0:
                print(f"[PRM] Connected {node_id}/{len(self.roadmap.nodes)} nodes")
        
        self.construction_time = time.time() - start_time
        self.roadmap_built = True
        
        if verbose:
            print(f"[PRM] Roadmap construction complete!")
            print(f"[PRM] {len(self.roadmap.nodes)} nodes, {connections_made//2} edges")
            print(f"[PRM] Construction time: {self.construction_time:.2f}s")
            print(f"[PRM] Average connections per node: {connections_made/len(self.roadmap.nodes):.1f}")
    
    def add_temporary_nodes(self, start: Tuple[float, float, float], 
                           goal: Tuple[float, float, float]) -> Tuple[int, int]:
        """Add start and goal as temporary nodes to the roadmap."""
        start_id = self.roadmap.add_node(start)
        goal_id = self.roadmap.add_node(goal)
        
        # Connect start to nearby nodes
        nearby_to_start = self.find_nearby_nodes(start, self.connection_radius)
        for node_id, distance in nearby_to_start:
            if node_id != start_id:
                if self.obstacle_checker.is_line_collision_free(start, 
                                                               self.roadmap.nodes[node_id].position):
                    self.roadmap.add_edge(start_id, node_id, distance)
        
        # Connect goal to nearby nodes
        nearby_to_goal = self.find_nearby_nodes(goal, self.connection_radius)
        for node_id, distance in nearby_to_goal:
            if node_id != goal_id:
                if self.obstacle_checker.is_line_collision_free(goal, 
                                                               self.roadmap.nodes[node_id].position):
                    self.roadmap.add_edge(goal_id, node_id, distance)
        
        return start_id, goal_id
    
    def remove_temporary_nodes(self, start_id: int, goal_id: int):
        """Remove temporary start and goal nodes from roadmap."""
        # Remove connections first
        if start_id in self.roadmap.nodes:
            for neighbor_id in list(self.roadmap.nodes[start_id].neighbors):
                if neighbor_id in self.roadmap.nodes:
                    self.roadmap.nodes[neighbor_id].neighbors.discard(start_id)
                    if start_id in self.roadmap.nodes[neighbor_id].distances:
                        del self.roadmap.nodes[neighbor_id].distances[start_id]
            del self.roadmap.nodes[start_id]
        
        if goal_id in self.roadmap.nodes:
            for neighbor_id in list(self.roadmap.nodes[goal_id].neighbors):
                if neighbor_id in self.roadmap.nodes:
                    self.roadmap.nodes[neighbor_id].neighbors.discard(goal_id)
                    if goal_id in self.roadmap.nodes[neighbor_id].distances:
                        del self.roadmap.nodes[neighbor_id].distances[goal_id]
            del self.roadmap.nodes[goal_id]
    
    def plan_path(self, start: Tuple[float, float, float], 
                  goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Plan a path from start to goal using PRM.
        
        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            
        Returns:
            List of waypoints in world coordinates, or empty list if no path found
        """
        # Check if start and goal are valid
        if not self.obstacle_checker.is_point_collision_free(start):
            print(f"[PRM] Start position {start} is in collision!")
            return []
        
        if not self.obstacle_checker.is_point_collision_free(goal):
            print(f"[PRM] Goal position {goal} is in collision!")
            return []
        
        # Build roadmap if not already built
        if not self.roadmap_built:
            self.construct_roadmap()
        
        print(f"[PRM] Planning path from {start} to {goal}")
        
        # Add temporary start and goal nodes
        start_id, goal_id = self.add_temporary_nodes(start, goal)
        
        # Find shortest path in roadmap
        path_ids = self.roadmap.dijkstra(start_id, goal_id)
        
        if not path_ids:
            print(f"[PRM] No path found in roadmap!")
            self.remove_temporary_nodes(start_id, goal_id)
            return []
        
        # Convert path IDs to world coordinates
        path_world = []
        total_distance = 0.0
        
        for i, node_id in enumerate(path_ids):
            if node_id in self.roadmap.nodes:
                position = self.roadmap.nodes[node_id].position
                path_world.append(position)
                
                if i > 0:
                    prev_pos = self.roadmap.nodes[path_ids[i-1]].position
                    segment_distance = self.calculate_distance(position, prev_pos)
                    total_distance += segment_distance
        
        # Clean up temporary nodes
        self.remove_temporary_nodes(start_id, goal_id)
        
        print(f"[PRM] Path found with {len(path_world)} waypoints")
        print(f"[PRM] Total path length: {total_distance:.3f}m")
        print(f"[PRM] Using roadmap with {len(self.roadmap.nodes)} nodes")
        
        return path_world


def create_prm_trajectory(start_pos: Tuple[float, float, float],
                         goal_pos: Tuple[float, float, float],
                         obstacle_data: List[Dict[str, Any]],
                         grid_resolution: float,
                         safety_margin: float,
                         num_samples: int,
                         connection_radius: float,
                         use_3d: bool = True) -> np.ndarray:
    """
    High-level interface to create PRM trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (for bounds)
        safety_margin: Additional clearance around obstacles
        num_samples: Number of samples for roadmap construction
        connection_radius: Maximum distance for connecting roadmap nodes
        use_3d: Use full 3D planning or just X-Z plane

    Returns:
        NumPy array of shape [N, 3] containing path waypoints (float32)
    """
    # Set up grid bounds based on obstacles and start/goal
    if len(obstacle_data) > 0:
        all_positions = [obs["pos"] for obs in obstacle_data] + [list(start_pos), list(goal_pos)]
        all_positions = np.array(all_positions)
        
        bounds_min = np.min(all_positions, axis=0) - 0.1
        bounds_max = np.max(all_positions, axis=0) + 0.1
        
        # Ensure minimum workspace size
        bounds_min = np.minimum(bounds_min, [0.34, -0.36, 0.0])
        bounds_max = np.maximum(bounds_max, [0.85, 0.36, 0.78])
    else:
        # Default bounds
        bounds_min = (0.34, -0.36, 0.0)
        bounds_max = (0.85, 0.36, 0.78)
    
    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
        safety_margin=safety_margin
    )
    
    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)
    
    # Create PRM planner
    planner = PRM(grid_config, obstacle_checker, use_3d=use_3d)
    planner.num_samples = num_samples
    planner.connection_radius = connection_radius
    
    # Plan path
    path = planner.plan_path(start_pos, goal_pos)

    if not path:
        raise RuntimeError("PRM failed to find a path")

    # Convert to NumPy array (float32 for numba compatibility)
    path_array = np.array(path, dtype=np.float32)

    print(f"[PRM] Created trajectory with {len(path)} waypoints")
    print(f"[PRM] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[PRM] Roadmap: {num_samples} samples, {connection_radius:.3f}m connection radius")
    print(f"[PRM] Safety margin: {safety_margin}m")
    print(f"[PRM] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array
