"""
Path Planning Algorithms - NumPy Implementation
================================================

Unified path planning algorithms for robotics applications.
All algorithms use NumPy arrays only (no PyTorch dependencies).

Includes:
- A* (A-star) pathfinding with 3D/2D support
- RRT (Rapidly-exploring Random Tree)
- RRT* (RRT with rewiring optimization)
- PRM (Probabilistic Roadmap)

All algorithms:
- Accept tuning parameters from configuration
- Use shared ObstacleChecker for consistency
- Return NumPy arrays (float32, shape [N, 3])
- Support both 3D and planar (X-Z) planning

Author: Refactored for NumPy-only usage
Date: 2025
"""

import heapq
import math
import numpy as np
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass

# Import utilities from path_utils
from .path_utils import (
    GridConfig,
    ObstacleChecker,
    calculate_workspace_bounds,
    quaternion_conjugate,
    quaternion_multiply,
    rotation_matrix_to_quaternion,
    basis_start_goal_plane,
    normalize_vector,
    standardize_path,
)


# ============================================================================
# Parameter Dataclasses (algorithm-specific + standardizer)
# ============================================================================

@dataclass
class AStarParams:
    max_iterations: int = 200000


@dataclass
class RRTParams:
    max_iterations: int = 10000
    max_acceptable_cost: Optional[float] = None
    max_step_size: float = 0.05
    goal_bias: float = 0.1
    goal_tolerance: float = 0.02


@dataclass
class RRTStarParams(RRTParams):
    rewire_radius: float = 0.08
    minimum_iterations: int = 1000
    cost_improvement_patience: int = 5000


@dataclass
class PRMParams:
    num_samples: int = 1000
    connection_radius: float = 0.12
    max_connections_per_node: int = 15


@dataclass
class StandardizerParams:
    speed_mps: float = 0.10
    dt: float = 0.20
    max_points: int = 30
    smoothing: Optional[Dict[str, float]] = None
    return_poses: bool = True


# ============================================================================
# A* (A-Star) Pathfinding Algorithm
# ============================================================================

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
        world_curr = self.grid_to_world(node)

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

            # Corner-cutting prevention for 2D (planar) mode
            if not self.use_3d and dx != 0 and dz != 0:
                # both orthogonal steps must be free
                n1 = (new_x, y, z)           # step in X only
                n2 = (x, y, new_z)           # step in Z only
                w1 = self.grid_to_world(n1)
                w2 = self.grid_to_world(n2)
                if (not self.obstacle_checker.is_point_collision_free(w1) or
                    not self.obstacle_checker.is_point_collision_free(w2)):
                    continue

            # Check collision
            if not self.obstacle_checker.is_point_collision_free(world_pos):
                continue

            # --- Edge collision sampling to prevent skipping through thin geometry ---
            seg_len = math.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_config.resolution
            # sample ~every 0.5 * resolution along the short edge (min 5)
            num_samples = max(5, int(seg_len / (self.grid_config.resolution * 0.5)))
            if not self.obstacle_checker.is_line_collision_free(world_curr, world_pos, num_samples=num_samples):
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
        print(f"[A*] Max iterations: {max_iterations}")

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


# ============================================================================
# RRT* (Rapidly-exploring Random Tree Star) Algorithm
# ============================================================================

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
                 use_3d: bool = True,
                 max_step_size: float = 0.05,
                 goal_bias: float = 0.1,
                 rewire_radius: float = 0.08,
                 goal_tolerance: float = 0.02,
                 minimum_iterations: int = 1000,
                 cost_improvement_patience: int = 5000):
        """
        Initialize RRT* planner.

        Args:
            grid_config: Grid configuration for bounds and resolution
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            max_step_size: Maximum distance for extending tree
            goal_bias: Probability of sampling toward goal (0.0-1.0)
            rewire_radius: Radius for rewiring optimization
            goal_tolerance: Distance to consider goal reached
            minimum_iterations: Minimum iterations before early termination
            cost_improvement_patience: Iterations to wait for cost improvement before terminating
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d

        # RRT* specific parameters (now configurable!)
        self.max_step_size = max_step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.goal_tolerance = goal_tolerance

        # Early termination parameters
        self.max_acceptable_cost = None  # Set during plan_path call
        self.cost_improvement_patience = cost_improvement_patience
        self.minimum_iterations = minimum_iterations

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
            max_acceptable_cost: Early termination cost threshold (meters)

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
        print(f"[RRT*] Parameters: step={self.max_step_size}, bias={self.goal_bias}, rewire={self.rewire_radius}")

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
        print(f"[RRT*] Path found with {len(path)} waypoints, total cost: {goal_node.cost:.3f}m")

        return path


# ============================================================================
# RRT (Basic Rapidly-exploring Random Tree) Algorithm
# ============================================================================

class RRT(RRTStar):
    """Basic RRT algorithm (no rewiring optimization)."""

    def rewire_tree(self, tree, new_node, near_nodes):
        """No rewiring in basic RRT."""
        pass

    def choose_parent(self, near_nodes, new_position):
        """Just use nearest node, no cost optimization."""
        return near_nodes[0] if near_nodes else None, 0.0


# ============================================================================
# PRM (Probabilistic Roadmap) Algorithm
# ============================================================================

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
                 use_3d: bool = True,
                 num_samples: int = 1000,
                 connection_radius: float = 0.12,
                 max_connections_per_node: int = 15):
        """
        Initialize PRM planner.

        Args:
            grid_config: Grid configuration for workspace bounds
            obstacle_checker: Obstacle collision checker
            use_3d: If True, use full 3D planning. If False, plan in X-Z plane only.
            num_samples: Number of random samples for roadmap
            connection_radius: Max distance for connecting nodes
            max_connections_per_node: Limit connections for efficiency
        """
        self.grid_config = grid_config
        self.obstacle_checker = obstacle_checker
        self.use_3d = use_3d

        # PRM-specific parameters (now configurable!)
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.max_connections_per_node = max_connections_per_node

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
            print(f"[PRM] Parameters: samples={self.num_samples}, radius={self.connection_radius}")

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


# ============================================================================
# High-Level Wrapper Functions
# ============================================================================

def create_astar_3d_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float = 0.01,
                              safety_margin: float = 0.02,
                              use_3d: bool = True,
                              max_iterations: int = 200000) -> np.ndarray:
    """
    High-level interface to create A* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        max_iterations: Maximum A* search iterations

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create A* planner
    planner = AStar3D(grid_config, obstacle_checker, use_3d=use_3d)

    # Plan path
    path = planner.plan_path(start_pos, goal_pos, max_iterations=max_iterations)

    if not path:
        raise RuntimeError("A* failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[A*] Created trajectory with {len(path)} waypoints")
    print(f"[A*] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[A*] Grid resolution: {grid_resolution}m, Safety margin: {safety_margin}m")
    print(f"[A*] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


def create_astar_plane_trajectory(start_pos: Tuple[float, float, float],
                                  goal_pos: Tuple[float, float, float],
                                  obstacle_data: List[Dict[str, Any]],
                                  grid_resolution: float = 0.01,
                                  safety_margin: float = 0.02,
                                  max_iterations: int = 200000,
                                  tool_radius: float = 0.0275) -> np.ndarray:
    """
    A* constrained to the vertical plane defined by start→goal.

    - Rotates the scene into that plane so planning occurs in Xp–Zp (Yp locked to 0).
    - Plans with A* in 2D (use_3d=False) for smoother, more linear slew motion.
    - Transforms the resulting path back to world coordinates.

    Args:
        tool_radius: **A* Plane-specific parameter**. Radius of the tool/end-effector
                     in meters. Obstacles are widened by 2*tool_radius in the X and Z
                     plane directions for collision avoidance. This is unique to the
                     plane algorithm due to its coordinate transformation approach.
                     Default: 0.0275m (27.5mm). Not used by other algorithms.
    """
    # Build plane basis (columns Xp, Yp, Zp) and rotation matrix world↔plane
    s_w = np.asarray(start_pos, dtype=np.float32)
    g_w = np.asarray(goal_pos, dtype=np.float32)
    Xp, Yp, Zp = basis_start_goal_plane(s_w, g_w)  # columns for R_wp = [Xp Yp Zp]
    R_wp = np.stack([Xp, Yp, Zp], axis=1).astype(np.float32)  # maps plane axes to world

    def world_to_plane(p_w: np.ndarray) -> np.ndarray:
        return R_wp.T @ (p_w - s_w)

    def plane_to_world(p_p: np.ndarray) -> np.ndarray:
        return (R_wp @ p_p) + s_w

    # Quaternion for world→plane transform: q_wp corresponds to R_wp
    q_wp = rotation_matrix_to_quaternion(R_wp)
    q_wp_conj = quaternion_conjugate(q_wp)

    # Transform obstacles into plane frame (normalize keys first)
    obstacles_plane: List[Dict[str, Any]] = []
    for obs in obstacle_data or []:
        size_w = np.asarray(obs.get("size", obs.get("dimensions", [0.1, 0.1, 0.1])), dtype=np.float32)
        pos_w = np.asarray(obs.get("pos", obs.get("position", [0.0, 0.0, 0.0])), dtype=np.float32)
        rot_w = np.asarray(obs.get("rot", obs.get("rotation", [1.0, 0.0, 0.0, 0.0])), dtype=np.float32)

        # Widen obstacle by tool_radius in plane X and Z directions
        size_p = size_w.astype(np.float32).copy()
        size_p[0] += 2.0 * tool_radius    # widen along X_p
        size_p[2] += 2.0 * tool_radius    # widen along Z_p
        # Keep a very thin but non-zero Y_p thickness so the 2D checker is stable
        size_p[1] = max(size_p[1], 0.02)

        pos_p = world_to_plane(pos_w)
        rot_p = quaternion_multiply(q_wp_conj, rot_w)

        obstacles_plane.append({
            "size": size_p.astype(np.float32),
            "pos": pos_p.astype(np.float32),
            "rot": rot_p.astype(np.float32),
        })

    # Start/goal in plane frame with Yp locked to 0
    s_p = world_to_plane(s_w); s_p[1] = 0.0
    g_p = world_to_plane(g_w); g_p[1] = 0.0

    # Bounds in plane frame (use shared utility for consistency)
    bounds_min_p, bounds_max_p = calculate_workspace_bounds(obstacles_plane, tuple(s_p), tuple(g_p))

    # Create grid configuration in plane frame
    grid_config_p = GridConfig(
        resolution=float(grid_resolution),
        bounds_min=bounds_min_p,
        bounds_max=bounds_max_p,
        safety_margin=float(safety_margin),
    )

    # Obstacle checker in plane frame
    obstacle_checker_p = ObstacleChecker(obstacles_plane, safety_margin=float(safety_margin))

    # Plan with 2D A* in plane
    planner = AStar3D(grid_config_p, obstacle_checker_p, use_3d=False)
    path_plane = planner.plan_path(tuple(s_p.tolist()), tuple(g_p.tolist()), max_iterations=max_iterations)
    if not path_plane:
        raise RuntimeError("A* Plane: no feasible path found on the start→goal plane")

    # Map path back to world (enforce Yp=0)
    path_world = []
    for p in path_plane:
        pp = np.array([p[0], 0.0, p[2]], dtype=np.float32)
        pw = plane_to_world(pp)
        path_world.append(pw.tolist())

    return np.asarray(path_world, dtype=np.float32)


def create_rrt_star_trajectory(start_pos: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float],
                              obstacle_data: List[Dict[str, Any]],
                              grid_resolution: float = 0.01,
                              safety_margin: float = 0.02,
                              use_3d: bool = True,
                              max_iterations: int = 10000,
                              max_acceptable_cost: Optional[float] = None,
                              max_step_size: float = 0.05,
                              goal_bias: float = 0.1,
                              rewire_radius: float = 0.08,
                              goal_tolerance: float = 0.02,
                              minimum_iterations: int = 1000,
                              cost_improvement_patience: int = 5000) -> np.ndarray:
    """
    High-level interface to create RRT* trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        max_iterations: Maximum RRT* iterations
        max_acceptable_cost: Early termination cost threshold (meters)
        max_step_size: Maximum tree extension distance
        goal_bias: Probability of sampling toward goal (0.0-1.0)
        rewire_radius: Radius for tree rewiring
        goal_tolerance: Distance threshold to reach goal
        minimum_iterations: Minimum iterations before early stop
        cost_improvement_patience: Iterations to wait for improvement

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create RRT* planner with all tuning parameters
    planner = RRTStar(
        grid_config, obstacle_checker, use_3d=use_3d,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        rewire_radius=rewire_radius,
        goal_tolerance=goal_tolerance,
        minimum_iterations=minimum_iterations,
        cost_improvement_patience=cost_improvement_patience
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos,
                             max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT* failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT*] Created trajectory with {len(path)} waypoints")
    print(f"[RRT*] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT*] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT*] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


def create_rrt_trajectory(start_pos: Tuple[float, float, float],
                         goal_pos: Tuple[float, float, float],
                         obstacle_data: List[Dict[str, Any]],
                         grid_resolution: float = 0.01,
                         safety_margin: float = 0.02,
                         use_3d: bool = True,
                         max_iterations: int = 10000,
                         max_acceptable_cost: Optional[float] = None,
                         max_step_size: float = 0.05,
                         goal_bias: float = 0.1,
                         goal_tolerance: float = 0.02) -> np.ndarray:
    """
    High-level interface to create RRT trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (used for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        max_iterations: Maximum RRT iterations
        max_acceptable_cost: Early termination cost threshold (meters)
        max_step_size: Maximum tree extension distance
        goal_bias: Probability of sampling toward goal (0.0-1.0)
        goal_tolerance: Distance threshold to reach goal

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create RRT planner (no rewiring)
    planner = RRT(
        grid_config, obstacle_checker, use_3d=use_3d,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        rewire_radius=0.0,  # Not used in basic RRT
        goal_tolerance=goal_tolerance,
        minimum_iterations=100,  # Lower for basic RRT
        cost_improvement_patience=10000
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos,
                             max_iterations=max_iterations,
                             max_acceptable_cost=max_acceptable_cost)

    if not path:
        raise RuntimeError("RRT failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[RRT] Created trajectory with {len(path)} waypoints")
    print(f"[RRT] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[RRT] Safety margin: {safety_margin}m, Max iterations: {max_iterations}")
    print(f"[RRT] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


def create_prm_trajectory(start_pos: Tuple[float, float, float],
                         goal_pos: Tuple[float, float, float],
                         obstacle_data: List[Dict[str, Any]],
                         grid_resolution: float = 0.01,
                         safety_margin: float = 0.02,
                         use_3d: bool = True,
                         num_samples: int = 1000,
                         connection_radius: float = 0.12,
                         max_connections_per_node: int = 15) -> np.ndarray:
    """
    High-level interface to create PRM trajectory with obstacle avoidance.

    Args:
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacle_data: List of obstacle dictionaries
        grid_resolution: Grid cell size in meters (for bounds)
        safety_margin: Additional clearance around obstacles
        use_3d: Use full 3D planning or just X-Z plane
        num_samples: Number of samples for roadmap construction
        connection_radius: Maximum distance for connecting roadmap nodes
        max_connections_per_node: Limit connections per node

    Returns:
        NumPy array of shape [N, 3] containing path waypoints
    """
    # Calculate workspace bounds
    bounds_min, bounds_max = calculate_workspace_bounds(obstacle_data, start_pos, goal_pos)

    # Create grid configuration
    grid_config = GridConfig(
        resolution=grid_resolution,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_margin=safety_margin
    )

    # Create obstacle checker
    obstacle_checker = ObstacleChecker(obstacle_data, safety_margin)

    # Create PRM planner with all tuning parameters
    planner = PRM(
        grid_config, obstacle_checker, use_3d=use_3d,
        num_samples=num_samples,
        connection_radius=connection_radius,
        max_connections_per_node=max_connections_per_node
    )

    # Plan path
    path = planner.plan_path(start_pos, goal_pos)

    if not path:
        raise RuntimeError("PRM failed to find a path")

    # Convert to numpy array
    path_array = np.array(path, dtype=np.float32)

    print(f"[PRM] Created trajectory with {len(path)} waypoints")
    print(f"[PRM] Start: {start_pos}, Goal: {goal_pos}")
    print(f"[PRM] Roadmap: {num_samples} samples, {connection_radius:.3f}m connection radius")
    print(f"[PRM] Safety margin: {safety_margin}m")
    print(f"[PRM] Workspace bounds: {bounds_min} to {bounds_max}")

    return path_array


# ============================================================================
# Standardized (Constant-Speed) Wrapper Functions
# ============================================================================

def create_astar_3d_trajectory_standardized(start_pos: Tuple[float, float, float],
                                            goal_pos: Tuple[float, float, float],
                                            obstacle_data: List[Dict[str, Any]],
                                            grid_resolution: float = 0.01,
                                            safety_margin: float = 0.02,
                                            use_3d: bool = True,
                                            max_iterations: int = 200000,
                                            *,
                                            params: Optional['AStarParams'] = None,
                                            standardizer: Optional['StandardizerParams'] = None,
                                            speed_mps: float = 0.10,
                                            dt: float = 0.20,
                                            max_points: int = 30,
                                            smoothing: Optional[Dict[str, float]] = None,
                                            return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Create an A* path and standardize it to constant-speed execution.

    Returns a dictionary from standardize_path with positions/poses and times.
    """
    # Resolve algorithm parameters (kwargs > params > defaults)
    if params is not None and max_iterations is None:
        max_iterations = params.max_iterations

    raw = create_astar_3d_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        max_iterations=max_iterations,
    )
    # Resolve standardizer parameters (kwargs > object > defaults)
    if standardizer is not None:
        if speed_mps is None: speed_mps = standardizer.speed_mps
        if dt is None: dt = standardizer.dt
        if max_points is None: max_points = standardizer.max_points
        if smoothing is None: smoothing = standardizer.smoothing
        if return_poses is None: return_poses = standardizer.return_poses

    return standardize_path(raw, speed_mps=speed_mps, dt=dt,
                            max_points=max_points, smoothing=smoothing,
                            return_poses=return_poses)


def create_astar_plane_trajectory_standardized(start_pos: Tuple[float, float, float],
                                               goal_pos: Tuple[float, float, float],
                                               obstacle_data: List[Dict[str, Any]],
                                               grid_resolution: float = 0.01,
                                               safety_margin: float = 0.02,
                                               max_iterations: int = 200000,
                                               tool_radius: float = 0.0275,
                                               *,
                                               params: Optional['AStarParams'] = None,
                                               standardizer: Optional['StandardizerParams'] = None,
                                               speed_mps: float = 0.10,
                                               dt: float = 0.20,
                                               max_points: int = 30,
                                               smoothing: Optional[Dict[str, float]] = None,
                                               return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Planar (X-Z) A* standardized to constant-speed execution.

    Signature mirrors create_astar_3d_trajectory_standardized, with the addition
    of the A* Plane-specific tool_radius parameter.

    Args:
        tool_radius: **A* Plane-specific parameter**. Radius of the tool/end-effector
                     in meters. Obstacles are widened by 2*tool_radius in the X and Z
                     plane directions for collision avoidance. This is unique to the
                     plane algorithm due to its coordinate transformation approach.
                     Default: 0.0275m (27.5mm). Not used by other algorithms.
    """
    # Resolve algorithm parameters (kwargs > params > defaults)
    if params is not None and max_iterations is None:
        max_iterations = params.max_iterations

    raw = create_astar_plane_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        max_iterations=max_iterations,
        tool_radius=tool_radius,
    )

    # Resolve standardizer parameters (kwargs > object > defaults)
    if standardizer is not None:
        if speed_mps is None: speed_mps = standardizer.speed_mps
        if dt is None: dt = standardizer.dt
        if max_points is None: max_points = standardizer.max_points
        if smoothing is None: smoothing = standardizer.smoothing
        if return_poses is None: return_poses = standardizer.return_poses

    return standardize_path(raw, speed_mps=speed_mps, dt=dt,
                            max_points=max_points, smoothing=smoothing,
                            return_poses=return_poses)


# ---------------------------------------------------------------------------
# Back-compat shim for older sim code expecting a plane-specific A*
# ---------------------------------------------------------------------------
def create_astar_on_start_goal_plane(start_pos: Tuple[float, float, float],
                                     goal_pos: Tuple[float, float, float],
                                     wall_obstacle: Dict[str, Any],
                                     grid_resolution: float = 0.01,
                                     safety_margin: float = 0.02) -> np.ndarray:
    """
    Compatibility wrapper for older sim/run_sim.py that used a plane-constrained A*.

    Internally calls the planar A* wrapper with obstacle_data=[wall_obstacle].
    Returns a NumPy array of shape [N, 3].
    """
    obstacle_data = [wall_obstacle] if wall_obstacle is not None else []
    return create_astar_plane_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        max_iterations=200000,
    )


def create_rrt_star_trajectory_standardized(start_pos: Tuple[float, float, float],
                                            goal_pos: Tuple[float, float, float],
                                            obstacle_data: List[Dict[str, Any]],
                                            grid_resolution: float = 0.01,
                                            safety_margin: float = 0.02,
                                            use_3d: bool = True,
                                            max_iterations: int = 10000,
                                            max_acceptable_cost: Optional[float] = None,
                                            max_step_size: float = 0.05,
                                            goal_bias: float = 0.1,
                                            rewire_radius: float = 0.08,
                                            goal_tolerance: float = 0.02,
                                            minimum_iterations: int = 1000,
                                            cost_improvement_patience: int = 5000,
                                            *,
                                            params: Optional['RRTStarParams'] = None,
                                            standardizer: Optional['StandardizerParams'] = None,
                                            speed_mps: float = 0.10,
                                            dt: float = 0.20,
                                            max_points: int = 30,
                                            smoothing: Optional[Dict[str, float]] = None,
                                            return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Create an RRT* path and standardize it to constant-speed execution."""
    # Resolve algorithm parameters (kwargs > params > defaults)
    if params is not None:
        if max_iterations is None: max_iterations = params.max_iterations
        if max_acceptable_cost is None: max_acceptable_cost = params.max_acceptable_cost
        if max_step_size is None: max_step_size = params.max_step_size
        if goal_bias is None: goal_bias = params.goal_bias
        if goal_tolerance is None: goal_tolerance = params.goal_tolerance
        if rewire_radius is None: rewire_radius = params.rewire_radius
        if minimum_iterations is None: minimum_iterations = params.minimum_iterations
        if cost_improvement_patience is None: cost_improvement_patience = params.cost_improvement_patience

    raw = create_rrt_star_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        max_iterations=max_iterations,
        max_acceptable_cost=max_acceptable_cost,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        rewire_radius=rewire_radius,
        goal_tolerance=goal_tolerance,
        minimum_iterations=minimum_iterations,
        cost_improvement_patience=cost_improvement_patience,
    )
    # Resolve standardizer parameters
    if standardizer is not None:
        if speed_mps is None: speed_mps = standardizer.speed_mps
        if dt is None: dt = standardizer.dt
        if max_points is None: max_points = standardizer.max_points
        if smoothing is None: smoothing = standardizer.smoothing
        if return_poses is None: return_poses = standardizer.return_poses

    return standardize_path(raw, speed_mps=speed_mps, dt=dt,
                            max_points=max_points, smoothing=smoothing,
                            return_poses=return_poses)


def create_rrt_trajectory_standardized(start_pos: Tuple[float, float, float],
                                       goal_pos: Tuple[float, float, float],
                                       obstacle_data: List[Dict[str, Any]],
                                       grid_resolution: float = 0.01,
                                       safety_margin: float = 0.02,
                                       use_3d: bool = True,
                                       max_iterations: int = 10000,
                                       max_acceptable_cost: Optional[float] = None,
                                       max_step_size: float = 0.05,
                                       goal_bias: float = 0.1,
                                       goal_tolerance: float = 0.02,
                                       *,
                                       params: Optional['RRTParams'] = None,
                                       standardizer: Optional['StandardizerParams'] = None,
                                       speed_mps: float = 0.10,
                                       dt: float = 0.20,
                                       max_points: int = 30,
                                       smoothing: Optional[Dict[str, float]] = None,
                                       return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Create an RRT path and standardize it to constant-speed execution."""
    # Resolve algorithm parameters (kwargs > params > defaults)
    if params is not None:
        if max_iterations is None: max_iterations = params.max_iterations
        if max_acceptable_cost is None: max_acceptable_cost = params.max_acceptable_cost
        if max_step_size is None: max_step_size = params.max_step_size
        if goal_bias is None: goal_bias = params.goal_bias
        if goal_tolerance is None: goal_tolerance = params.goal_tolerance

    raw = create_rrt_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        max_iterations=max_iterations,
        max_acceptable_cost=max_acceptable_cost,
        max_step_size=max_step_size,
        goal_bias=goal_bias,
        goal_tolerance=goal_tolerance,
    )
    # Resolve standardizer parameters
    if standardizer is not None:
        if speed_mps is None: speed_mps = standardizer.speed_mps
        if dt is None: dt = standardizer.dt
        if max_points is None: max_points = standardizer.max_points
        if smoothing is None: smoothing = standardizer.smoothing
        if return_poses is None: return_poses = standardizer.return_poses

    return standardize_path(raw, speed_mps=speed_mps, dt=dt,
                            max_points=max_points, smoothing=smoothing,
                            return_poses=return_poses)


def create_prm_trajectory_standardized(start_pos: Tuple[float, float, float],
                                       goal_pos: Tuple[float, float, float],
                                       obstacle_data: List[Dict[str, Any]],
                                       grid_resolution: float = 0.01,
                                       safety_margin: float = 0.02,
                                       use_3d: bool = True,
                                       num_samples: int = 1000,
                                       connection_radius: float = 0.12,
                                       max_connections_per_node: int = 15,
                                       *,
                                       params: Optional['PRMParams'] = None,
                                       standardizer: Optional['StandardizerParams'] = None,
                                       speed_mps: float = 0.10,
                                       dt: float = 0.20,
                                       max_points: int = 30,
                                       smoothing: Optional[Dict[str, float]] = None,
                                       return_poses: bool = True) -> Dict[str, np.ndarray]:
    """Create a PRM path and standardize it to constant-speed execution."""
    # Resolve algorithm parameters (kwargs > params > defaults)
    if params is not None:
        if num_samples is None: num_samples = params.num_samples
        if connection_radius is None: connection_radius = params.connection_radius
        if max_connections_per_node is None: max_connections_per_node = params.max_connections_per_node

    raw = create_prm_trajectory(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacle_data=obstacle_data,
        grid_resolution=grid_resolution,
        safety_margin=safety_margin,
        use_3d=use_3d,
        num_samples=num_samples,
        connection_radius=connection_radius,
        max_connections_per_node=max_connections_per_node,
    )
    # Resolve standardizer parameters
    if standardizer is not None:
        if speed_mps is None: speed_mps = standardizer.speed_mps
        if dt is None: dt = standardizer.dt
        if max_points is None: max_points = standardizer.max_points
        if smoothing is None: smoothing = standardizer.smoothing
        if return_poses is None: return_poses = standardizer.return_poses

    return standardize_path(raw, speed_mps=speed_mps, dt=dt,
                            max_points=max_points, smoothing=smoothing,
                            return_poses=return_poses)
