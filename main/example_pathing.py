#!/usr/bin/env python3
"""
Path Planning Example with Multiple Algorithms

Usage:
    python example_pathing.py --algo a_star --debug --log

Arguments:
    --algo [name]  : Path planning algorithm to use (default: a_star)
                     Options: a_star, rrt, rrt_star, prm
    --debug        : Enable performance metrics and detailed logging
    --log          : Enable trajectory data logging to CSV files
"""
# NOTE: This script is intended for real hardware only (not simulation).
# TODO: If a sim mode is needed in future, gate hardware deps/creation behind a flag.
# import math  # unused
import time
import os
import traceback
import numpy as np
import sys  # unused
import argparse
import pandas as pd
from modules.hardware_interface import HardwareInterface # pwm + imus
from modules.excavator_controller import ExcavatorController, ControllerConfig # ik + pid
from pathing.path_planning_algorithms import (
    create_astar_3d_trajectory, create_rrt_trajectory,
    create_rrt_star_trajectory, create_prm_trajectory
)
from pathing_config import (
    DEFAULT_CONFIG, DEFAULT_ENV_CONFIG,
)
from pathing.path_utils import (
    interpolate_path,
    calculate_path_length,
    interpolate_along_path,
    calculate_execution_time,
    print_path_info,
)
import json  # unused
import csv   # unused
from datetime import datetime
from typing import List, Dict, Any
from modules.quaternion_math import quat_from_y_deg


class DataLogger:
    """data logging"""

    def __init__(self, algorithm_name: str, base_log_dir: str = "data"):
        # Create algorithm-specific subdirectory (matches simulation)
        algo_folders = []
        if os.path.exists(base_log_dir):
            algo_folders = [f for f in os.listdir(base_log_dir) if f.startswith(f"{algorithm_name}_")]

        if algo_folders:
            existing_nums = [int(f.split("_")[-1]) for f in algo_folders if f.split("_")[-1].isdigit()]
            folder_num = max(existing_nums) + 1 if existing_nums else 1
        else:
            folder_num = 1

        # Create algorithm-specific folder: data/a_star_1/, data/rrt_1/, etc.
        self.log_dir = os.path.join(base_log_dir, f"{algorithm_name}_{folder_num}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.algorithm_name = algorithm_name
        self.trajectory_log = []  # Store simulation-compatible format
        self.trajectory_counter = 0
        
        # Trajectory tracking state
        self.current_waypoint_index = 0
        self.current_progress = 0.0
        self.execution_start_time = None
        self.calculation_time = 0.0
        self.execution_time = 0.0
        self.total_distance_planned = 0.0
        self.total_distance_executed = 0.0
        self.max_tracking_error = 0.0
        self.avg_tracking_error = 0.0
        self.original_trajectory_points = 0
        self.interpolated_trajectory_points = 0
        self.prev_pos = None

        print(f"[Data Logger] Initialized for algorithm: {algorithm_name}")
        print(f"[Data Logger] Logging to: {self.log_dir}")
    
    def start_trajectory_tracking(self, original_waypoints: int, interpolated_waypoints: int):
        """Start tracking a new trajectory execution (matches simulation)."""
        self.trajectory_log.clear()
        self.execution_start_time = time.time()
        self.max_tracking_error = 0.0
        self.total_distance_executed = 0.0
        self.original_trajectory_points = original_waypoints
        self.interpolated_trajectory_points = interpolated_waypoints
        self.prev_pos = None
        print(f"[Data Logger] Started trajectory tracking: {original_waypoints} original, {interpolated_waypoints} interpolated waypoints")
    
    def log_trajectory_step(self, planned_pos: np.ndarray, actual_pos: np.ndarray, 
                           planned_quat: np.ndarray, actual_quat: np.ndarray, joint_angles: np.ndarray,
                           waypoint_idx: int, progress: float):
        """Log a single trajectory step with orientations and joint angles - LIGHTWEIGHT."""
        # Store raw data only - calculations done in save_trajectory_log() for performance
        self.trajectory_log.append([
            # Position data
            float(planned_pos[0]),  # x_g
            float(planned_pos[1]),  # y_g  
            float(planned_pos[2]),  # z_g
            float(actual_pos[0]),   # x_e
            float(actual_pos[1]),   # y_e
            float(actual_pos[2]),   # z_e
            # Planned orientation (quaternion: w, x, y, z)
            float(planned_quat[0]), # quat_g_w
            float(planned_quat[1]), # quat_g_x
            float(planned_quat[2]), # quat_g_y
            float(planned_quat[3]), # quat_g_z
            # Actual orientation (quaternion: w, x, y, z)
            float(actual_quat[0]),  # quat_e_w
            float(actual_quat[1]),  # quat_e_x
            float(actual_quat[2]),  # quat_e_y
            float(actual_quat[3]),  # quat_e_z
            # Joint angles (5 joints)
            float(joint_angles[0]), # joint_1 (revolute_cabin)
            float(joint_angles[1]), # joint_2 (revolute_lift)
            float(joint_angles[2]), # joint_3 (revolute_tilt)
            float(joint_angles[3]), # joint_4 (revolute_scoop)
            float(joint_angles[4]), # joint_5 (revolute_gripper)
            # Tracking data
            int(waypoint_idx),      # waypoint_idx
            float(progress)         # progress (0.0-1.0)
        ])
    
    def save_trajectory_log(self, algorithm_name: str = "a_star"):
        """Save trajectory log to CSV file - ALL CALCULATIONS DONE HERE (not during real-time)."""
        if not self.trajectory_log:
            print("[Data Logger] No trajectory data to save")
            return
    
        self.trajectory_counter += 1
        self.execution_time = time.time() - self.execution_start_time if self.execution_start_time else 0.0
        simulation_steps_logged = len(self.trajectory_log)
        
        # POST-PROCESS: Calculate all metrics from logged data (for performance)
        errors = [np.linalg.norm(np.array(log[3:6]) - np.array(log[0:3])) for log in self.trajectory_log]
        self.avg_tracking_error = np.mean(errors) if errors else 0.0
        self.max_tracking_error = np.max(errors) if errors else 0.0
        
        # Calculate executed distance from logged positions
        actual_positions = [np.array(log[3:6]) for log in self.trajectory_log]
        self.total_distance_executed = 0.0
        for i in range(1, len(actual_positions)):
            self.total_distance_executed += np.linalg.norm(actual_positions[i] - actual_positions[i-1])
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save trajectory data with waypoint tracking, orientations, and joint angles (matches simulation exactly)
        columns = ['x_g', 'y_g', 'z_g', 'x_e', 'y_e', 'z_e',
                   'quat_g_w', 'quat_g_x', 'quat_g_y', 'quat_g_z',
                   'quat_e_w', 'quat_e_x', 'quat_e_y', 'quat_e_z',
                   'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
                   'waypoint_idx', 'progress']
        df = pd.DataFrame(self.trajectory_log, columns=columns)
        csv_path = os.path.join(self.log_dir, f"{algorithm_name}_{self.trajectory_counter}.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate efficiency ratio
        efficiency_ratio = self.total_distance_executed / self.total_distance_planned if self.total_distance_planned > 0 else 1.0
        
        # Save metrics (matches simulation format exactly)
        metrics = {
            'trajectory_id': self.trajectory_counter,
            'algorithm': algorithm_name,
            'calculation_time_s': self.calculation_time,
            'execution_time_s': self.execution_time - self.calculation_time,
            'original_waypoints': self.original_trajectory_points,
            'interpolated_waypoints': self.interpolated_trajectory_points,
            'simulation_steps': simulation_steps_logged,
            'planned_distance_m': self.total_distance_planned,
            'executed_distance_m': self.total_distance_executed,
            'max_tracking_error_m': self.max_tracking_error,
            'avg_tracking_error_m': self.avg_tracking_error,
            'efficiency_ratio': efficiency_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_path = os.path.join(self.log_dir, "metrics.csv")
        metrics_df = pd.DataFrame([metrics])
        
        # Append to existing metrics file or create new one
        if os.path.exists(metrics_path):
            existing_metrics = pd.read_csv(metrics_path)
            combined_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)
            combined_metrics.to_csv(metrics_path, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)
        
        print(f"[Data Logger] Saved trajectory {self.trajectory_counter} to {csv_path}")
        print(f"[Data Logger] Steps: {simulation_steps_logged}, Avg Error: {self.avg_tracking_error:.6f}m")
        print(f"[Data Logger] Distance: planned={self.total_distance_planned:.4f}m, executed={self.total_distance_executed:.4f}m")
    
    def set_calculation_time(self, calc_time: float):
        """Set A* calculation time for metrics."""
        self.calculation_time = calc_time
    
    def set_planned_distance(self, distance: float):
        """Set total planned distance for metrics."""
        self.total_distance_planned = distance


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Path Planning Example with Multiple Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--algo',
        type=str,
        default='a_star',
        choices=['a_star', 'rrt', 'rrt_star', 'prm'],
        help='Path planning algorithm to use (default: a_star)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable performance metrics and detailed logging'
    )

    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable trajectory data logging to CSV files'
    )

    return parser.parse_args()


def create_trajectory_with_algorithm(algo_name: str, start_pos: tuple, goal_pos: tuple,
                                     obstacles: List[Dict[str, Any]], path_config) -> np.ndarray:
    """
    Create trajectory using specified algorithm.

    Args:
        algo_name: Algorithm name ('a_star', 'rrt', 'rrt_star', 'prm')
        start_pos: Start position (x, y, z)
        goal_pos: Goal position (x, y, z)
        obstacles: List of obstacle data
        path_config: Path configuration object

    Returns:
        NumPy array of waypoints
    """
    if algo_name == 'a_star':
        return create_astar_3d_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            use_3d=path_config.use_3d
        )
    elif algo_name == 'rrt':
        return create_rrt_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            max_iterations=path_config.max_iterations,
            use_3d=path_config.use_3d,
            max_acceptable_cost=path_config.max_acceptable_cost,
            max_step_size=path_config.rrt_max_step_size,
            goal_bias=path_config.rrt_goal_bias,
            rewire_radius=path_config.rrt_rewire_radius,
            goal_tolerance=path_config.rrt_goal_tolerance,
            minimum_iterations=path_config.rrt_minimum_iterations,
            cost_improvement_patience=path_config.rrt_cost_improvement_patience
        )
    elif algo_name == 'rrt_star':
        return create_rrt_star_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            max_iterations=path_config.max_iterations,
            use_3d=path_config.use_3d,
            max_acceptable_cost=path_config.max_acceptable_cost,
            max_step_size=path_config.rrt_max_step_size,
            goal_bias=path_config.rrt_goal_bias,
            rewire_radius=path_config.rrt_rewire_radius,
            goal_tolerance=path_config.rrt_goal_tolerance,
            minimum_iterations=path_config.rrt_minimum_iterations,
            cost_improvement_patience=path_config.rrt_cost_improvement_patience
        )
    elif algo_name == 'prm':
        return create_prm_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            num_samples=path_config.num_samples,
            connection_radius=path_config.connection_radius,
            use_3d=path_config.use_3d
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def define_static_walls() -> List[Dict[str, Any]]:
    """Define static wall obstacles using configuration from pathing_config.py."""
    obstacles = []

    # Wall configuration from pathing_config.py
    wall = {
        "size": np.array(DEFAULT_ENV_CONFIG.wall_size),
        "pos": np.array(DEFAULT_ENV_CONFIG.wall_pos),
        "rot": np.array(DEFAULT_ENV_CONFIG.wall_rot)
    }
    obstacles.append(wall)

    print(f"[Obstacles] Defined {len(obstacles)} static walls:")
    for i, obs in enumerate(obstacles, 1):
        print(f"  Wall {i}: size={obs['size']}, pos={obs['pos']}")

    return obstacles


def execute_continuous_path(controller: ExcavatorController, path: np.ndarray,
                           target_y_rotation: float, speed_mps: float,
                           interpolation_factor: int, update_frequency: float,
                            data_logger: DataLogger = None,
                            enable_debug: bool = False
                            ) -> bool:
    """
    Execute planned path with continuous smooth motion and configurable speed.

    Args:
        controller: Excavator controller
        path: Planned waypoints from path planning algorithm
        target_y_rotation: Target Y rotation in degrees
        speed_mps: Speed in meters per second (default 0.05 m/s = 5cm/s)
        interpolation_factor: Points to add between each waypoint
        update_frequency: target loop rate (Hz)
        data_logger: Optional data logger
        enable_debug: Enable performance metrics display
    """
    print(f"[Continuous Path] Smoothing {len(path)} waypoints...")

    # Create smooth interpolated path
    smooth_path = interpolate_path(path, interpolation_factor)
    print(f"[Continuous Path] Interpolated to {len(smooth_path)} smooth waypoints")

    # Calculate timing based on speed
    total_time = calculate_execution_time(smooth_path, speed_mps)

    print_path_info(smooth_path, type('Config', (), {'speed_mps': speed_mps})(), "Continuous Path")

    # Execute path with continuous motion
    start_time = time.perf_counter()
    last_debug_print = 0.0
    #path_start_time = start_time
    if update_frequency is None:
        update_frequency = DEFAULT_CONFIG.update_frequency  # Use config default
    # Initialize drift-corrected scheduler variables
    t0 = time.perf_counter()
    iteration = 0

    while True:
        elapsed = time.perf_counter() - start_time
        progress = min(elapsed / total_time, 1.0)
        
        # Get current target position along smooth path
        current_target = interpolate_along_path(smooth_path, progress)
        
        # Continuously update controller target (no stopping at waypoints)
        controller.give_pose(current_target, target_y_rotation)
        
        # Data logging (if enabled) - matches simulation format
        if data_logger is not None:
            current_pos, current_rot_y = controller.get_pose()
            hardware_joints = controller.get_joint_angles()  # Get available joint angles from hardware
            
            # Pad joint angles to match simulation format (5 joints total)
            # Hardware currently provides fewer joints, so zero out missing ones
            joint_angles = np.zeros(5)
            joint_angles[:len(hardware_joints)] = hardware_joints  # Copy available joints
            # joint_angles[3] = 0.0  # revolute_scoop (gripper) - not available yet
            # joint_angles[4] = 0.0  # revolute_gripper (claw) - not available yet
            
            # Calculate waypoint index (approximate based on progress and path length)
            waypoint_idx = int(progress * len(path)) if len(path) > 0 else 0
            waypoint_idx = min(waypoint_idx, len(path) - 1)  # Clamp to valid range
            
            # Convert orientations to quaternions for logging
            # For now, using simplified orientation (Y rotation only -> quaternion)
            planned_quat = quat_from_y_deg(float(target_y_rotation))
            actual_quat = quat_from_y_deg(float(current_rot_y))
            
            data_logger.log_trajectory_step(
                planned_pos=current_target,
                actual_pos=current_pos,
                planned_quat=planned_quat,
                actual_quat=actual_quat,
                joint_angles=joint_angles,
                waypoint_idx=waypoint_idx,
                progress=progress
            )
        
        # Progress feedback every 2 seconds
        if elapsed - last_debug_print >= 2.0:
            print(f"[Continuous Path] Progress: {progress*100:.1f}% ({elapsed:.1f}s/{total_time:.1f}s)")

            # Display performance metrics if debug mode is enabled
            if enable_debug:
                current_pos, current_rot_y = controller.get_pose()
                joint_angles = controller.get_joint_angles()
                hw_status = controller.get_hardware_status()
                perf_stats = controller.get_performance_stats()

                # Calculate waypoint index
                waypoint_idx = int(progress * len(path)) if len(path) > 0 else 0
                waypoint_idx = min(waypoint_idx, len(path) - 1)

                # Calculate individual axis errors
                pos_error = current_pos - current_target
                distance_to_target = np.linalg.norm(pos_error)
                rot_error = abs(current_rot_y - target_y_rotation)

                print(f"\n[DEBUG] STATUS @{elapsed:.1f}s:")
                print(f"  Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                print(f"  Target: [{current_target[0]:.3f}, {current_target[1]:.3f}, {current_target[2]:.3f}]")
                print(f"  Error: [X={pos_error[0]:.4f}, Y={pos_error[1]:.4f}, Z={pos_error[2]:.4f}]m, Total={distance_to_target:.4f}m")
                print(f"  Rotation Y: {current_rot_y:.1f} deg (error: {rot_error:.1f} deg)")
                print(f"  Joints: [Slew={joint_angles[0]:.1f}, Boom={joint_angles[1]:.1f}, "
                      f"Arm={joint_angles[2]:.1f}, Bucket={joint_angles[3]:.1f}]")
                print(f"  Slew Encoder: {hw_status.get('slew_angle', 0.0):.3f} rad "
                      f"({np.degrees(hw_status.get('slew_angle', 0.0)):.1f} deg)")
                print(f"\n[DEBUG] CONTROL LOOP TIMING (target: {update_frequency:.1f}Hz, {1000.0/update_frequency:.2f}ms period):")
                print(f"  Rate: {perf_stats['actual_hz']:.2f}Hz | "
                      f"Period: {perf_stats['avg_loop_time_ms']:.2f}ms")
                print(f"  Compute: {perf_stats['avg_compute_time_ms']:.2f} ms [{perf_stats['min_compute_time_ms']:.2f}-{perf_stats['max_compute_time_ms']:.2f} ms], std={perf_stats['std_compute_time_ms']:.2f} ms")
                #
                #
                print(f"  Headroom: {perf_stats['avg_headroom_ms']:.2f}ms "
                      f"({perf_stats['avg_headroom_ms']/(1000.0/update_frequency)*100:.0f}% available) | "
                      f"CPU: {perf_stats['cpu_usage_pct']:.1f}%")
                print(f"  Overruns: {perf_stats['violation_pct']:.1f}% | "
                      f"Samples: {perf_stats['sample_count']}")
                print("=" * 80)

                # Reset performance stats for next window
                controller.reset_performance_stats()

            last_debug_print = elapsed

        # Check if we've reached the target position and rotation (like example.py does)
        current_pos, current_rot_y = controller.get_pose()
        final_pos_error = current_pos - path[-1]
        distance_to_final_target = np.linalg.norm(final_pos_error)
        rot_error = abs(current_rot_y - target_y_rotation)

        # Check if target reached with proper position AND rotation tolerance
        position_reached = distance_to_final_target < DEFAULT_CONFIG.final_target_tolerance
        rotation_reached = rot_error < np.degrees(DEFAULT_CONFIG.orientation_tolerance)

        # In position-only mode, ignore rotation check (controller.ik_config.command_type == "position")
        if hasattr(controller, 'ik_config') and controller.ik_config.command_type == "position":
            target_fully_reached = position_reached  # Position-only mode: ignore rotation
        else:
            target_fully_reached = position_reached and rotation_reached  # Pose mode: require both

        # Complete when target is reached (position AND rotation)
        # Also check the path progress is 100% to avoid early completion
        if target_fully_reached and progress >= 1.0:
            print(f"[Continuous Path] Target reached!")
            print(f"  Position error: [X={final_pos_error[0]:.4f}, Y={final_pos_error[1]:.4f}, Z={final_pos_error[2]:.4f}]m, Total={distance_to_final_target:.4f}m")
            print(f"  Rotation error: {rot_error:.1f} deg")
            print(f"[Continuous Path] Completed in {elapsed:.1f}s (estimated: {total_time:.1f}s)")
            return True

        # After estimated time, show extended waiting status
        if progress >= 1.0:
            # Show waiting status every 2 seconds after estimated completion time
            if int(elapsed) % 2 == 0 and elapsed > 0 and (elapsed - int(elapsed)) < (1.0 / update_frequency):
                extra_time = elapsed - total_time
                print(f"[Continuous Path] Waiting for target (+{extra_time:.1f}s):")
                print(f"  Error: [X={final_pos_error[0]:.4f}, Y={final_pos_error[1]:.4f}, Z={final_pos_error[2]:.4f}]m, Total={distance_to_final_target:.4f}m, Rot={rot_error:.1f} deg")
        
        # Sleep for smooth updates at configured frequency (drift-corrected)
        iteration += 1
        next_tick = t0 + iteration * (1.0 / update_frequency if update_frequency and update_frequency > 0 else 0.01)
        now2 = time.perf_counter()
        sleep_time = next_tick - now2
        if sleep_time > 0:
            time.sleep(sleep_time)



def main():
    """Main path planning demonstration."""

    # Parse command line arguments
    args = parse_arguments()

    # ==================== CONFIGURATION FROM pathing_config.py ====================
    # Load all configuration from pathing_config.py for consistency with simulation
    env_config = DEFAULT_ENV_CONFIG
    path_config = DEFAULT_CONFIG

    # Points A and B with their target rotations (from env config)
    point_a = np.array(env_config.point_a_pos)
    point_b = np.array(env_config.point_b_pos)

    point_a_rotation_deg = env_config.point_a_rotation_deg
    point_b_rotation_deg = env_config.point_b_rotation_deg

    # Motion control settings (from path config)
    excavator_speed_mps = path_config.speed_mps
    interpolation_density = path_config.interpolation_factor
    # ===============================================================================

    # Display configuration
    print("=" * 60)
    print(f"CONFIGURATION:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Debug mode (performance metrics): {'ENABLED' if args.debug else 'DISABLED'}")
    print(f"  Data logging: {'ENABLED' if args.log else 'DISABLED'}")
    print("=" * 60)

    # Initialize data logger only if logging is enabled (with algorithm-specific subdirectory)
    data_logger = DataLogger(algorithm_name=args.algo) if args.log else None
    
    print(f"Point A: {point_a} (rotation: {point_a_rotation_deg} deg)")
    print(f"Point B: {point_b} (rotation: {point_b_rotation_deg} deg)")
    
    # Define static wall obstacles
    obstacles = define_static_walls()
    
    # Create hardware interface (real hardware only)
    print("Initializing hardware...")
    hardware = HardwareInterface()

    # Create controller with config from pathing_config.py
    config = ControllerConfig(control_frequency=path_config.update_frequency)

    print("Initializing controller (includes numba warmup)...")
    controller = ExcavatorController(hardware, config, enable_perf_tracking=args.debug)
    
    # Start the background control loop
    controller.start()
    
    print(f"Starting {args.algo} path planning demonstration in 3 seconds...")
    time.sleep(3)
    
    try:
        # Continuous loop between points A and B
        cycle_count = 0
        current_start = point_a
        current_goal = point_b
        current_rotation = point_b_rotation_deg
        
        while True:  # Continuous loop
            cycle_count += 1
            direction = "A->B" if np.array_equal(current_start, point_a) else "B->A"
            print(f"\n{'='*60}")
            print(f"CYCLE #{cycle_count} - {direction}")
            print(f"{'='*60}")
            
            # Plan path with selected algorithm (controller already paused from previous cycle)
            print(f"Planning {args.algo} path from {current_start} to {current_goal}...")
            calc_start = time.time()
            path = create_trajectory_with_algorithm(
                algo_name=args.algo,
                start_pos=tuple(current_start),
                goal_pos=tuple(current_goal),
                obstacles=obstacles,
                path_config=path_config
            )

            print(f"{args.algo} found path with {len(path)} waypoints")

            # Set up data logging for this trajectory (if logging enabled)
            if data_logger is not None:
                # Calculate interpolated waypoints for metrics
                # Use 4x interpolation for non-A* algorithms (matches simulation)
                interp_factor = interpolation_density if args.algo == 'a_star' else 4 * interpolation_density
                smooth_path = interpolate_path(path, interp_factor)
                planned_distance = calculate_path_length(smooth_path)

                # Start trajectory tracking (matches simulation)
                data_logger.start_trajectory_tracking(
                    original_waypoints=len(path),
                    interpolated_waypoints=len(smooth_path)
                )
                data_logger.set_planned_distance(planned_distance)
                data_logger.set_calculation_time(time.time() - calc_start)

            # Resume controller and send 0.0 commands for 1 second to wake up PWM
            print("Resuming controller and waking up PWM controller...")
            controller.resume()

            # Send 0.0 commands for 1 second to ensure PWM is fully awake
            print("Sending 0.0 commands to PWM for 1 second...")
            wake_start = time.perf_counter()
            while time.perf_counter() - wake_start < 1.0:
                # Controller is already running, just wait for it to stabilize
                time.sleep(0.01)
            print("PWM controller ready!")
            print(f"Executing {direction} path with continuous motion...")
            # Use 4x interpolation for non-A* algorithms (matches simulation)
            exec_interp_factor = interpolation_density if args.algo == 'a_star' else 4 * interpolation_density
            success = execute_continuous_path(
                controller,
                path,
                current_rotation,
                speed_mps=excavator_speed_mps,
                interpolation_factor=exec_interp_factor,
                data_logger=data_logger,
                update_frequency=path_config.update_frequency,
                enable_debug=args.debug
            )
            
            if success:
                print(f"Successfully completed {direction} path!")

                # PAUSE THE MACHINE FIRST (before saving/calculating)
                print("Pausing controller before saving data and calculating next path...")
                controller.pause()  # This calls pwm.reset(reset_pump=False)

                # Save trajectory log immediately after each completion (like simulation)
                if data_logger is not None:
                    data_logger.save_trajectory_log(args.algo)
                    print(f"Trajectory log saved for {direction}")

                # Switch direction for next cycle
                if np.array_equal(current_start, point_a):
                    # Next cycle: B->A
                    current_start = point_b
                    current_goal = point_a
                    current_rotation = point_a_rotation_deg
                else:
                    # Next cycle: A->B
                    current_start = point_a
                    current_goal = point_b
                    current_rotation = point_b_rotation_deg

                #print("Pausing 2 seconds before calculating next path...")
                #time.sleep(2.0)
            else:
                print(f"Failed to complete {direction} path")
                controller.stop()
                time.sleep(10)
                break
    
    except Exception as e:
        print(f"Error during {args.algo} path planning/execution: {e}")
        traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Note: Data logging now happens after each trajectory completion (like simulation)
        if data_logger is None:
            print("[LOGGING] No data logging performed (use '--log' argument to enable)")

        # Clean shutdown
        print("Stopping controller...")
        controller.stop() # this calls pwm.reset internally!!

        # Explicit hardware reset
        print("Resetting hardware...")
        hardware.reset(reset_pump=True)

        # Show final status
        final_pos, final_rot_y = controller.get_pose()
        print(f"Final position: {final_pos}")
        print(f"Final Y rotation: {final_rot_y:.1f} deg")
        print(f"{args.algo} path planning test completed.")


if __name__ == "__main__":
    main()
