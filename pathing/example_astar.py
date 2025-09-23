#!/usr/bin/env python3
"""
path planning example

"""

import time
import os
import traceback
import numpy as np
import sys
import pandas as pd
from modules.hardware_interface import HardwareInterface # pwm + imus
from modules.excavator_controller import ExcavatorController, ControllerConfig # ik + pid
from pathing.pathing_algos import create_astar_3d_trajectory
from pathing_config import (
    DEFAULT_CONFIG, DEFAULT_ENV_CONFIG, 
    interpolate_path, calculate_path_length, 
    interpolate_along_path, calculate_execution_time,
    print_path_info
)
import json
import csv
from datetime import datetime
from typing import List, Dict, Any


class DataLogger:
    """data logging"""
    
    def __init__(self, session_name: str = None, log_dir: str = "data"):
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_name = session_name
        self.log_dir = log_dir
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
        
        print(f"[Data Logger] Initialized session: {session_name}")
        print(f"[Data Logger] Logging to: {log_dir}")
    
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
    
    def save_trajectory_log(self, algorithm_name: str = "astar_3d"):
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
                            ) -> bool:
    """
    Execute A* path with continuous smooth motion and configurable speed.
    
    Args:
        controller: Excavator controller
        path: A* waypoints 
        target_y_rotation: Target Y rotation in degrees
        speed_mps: Speed in meters per second (default 0.05 m/s = 5cm/s)
        interpolation_factor: Points to add between each A* waypoint
        update_frequency: target loop rate (Hz)
        data_logger: Optional data logger
    """
    print(f"[Continuous Path] Smoothing {len(path)} A* waypoints...")
    
    # Create smooth interpolated path
    smooth_path = interpolate_path(path, interpolation_factor)
    print(f"[Continuous Path] Interpolated to {len(smooth_path)} smooth waypoints")
    
    # Calculate timing based on speed
    total_time = calculate_execution_time(smooth_path, speed_mps)
    
    print_path_info(smooth_path, type('Config', (), {'speed_mps': speed_mps})(), "Continuous Path")
    
    # Execute path with continuous motion
    start_time = time.time()
    #path_start_time = start_time
    if update_frequency is None:
        update_frequency = DEFAULT_CONFIG.update_frequency  # Use config default
    
    while True:
        elapsed = time.time() - start_time
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
            import math
            planned_quat = np.array([
                math.cos(target_y_rotation * math.pi / 360.0),  # w (half angle)
                0.0,  # x
                math.sin(target_y_rotation * math.pi / 360.0),  # y (Z-axis rotation)
                0.0   # z
            ])
            actual_quat = np.array([
                math.cos(current_rot_y * math.pi / 360.0),      # w (half angle)
                0.0,  # x
                math.sin(current_rot_y * math.pi / 360.0),      # y (Z-axis rotation)
                0.0   # z
            ])
            
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
        if int(elapsed) % 2 == 0 and elapsed > 0 and (elapsed - int(elapsed)) < (1.0 / update_frequency):
            print(f"[Continuous Path] Progress: {progress*100:.1f}% ({elapsed:.1f}s/{total_time:.1f}s)")
        
        # Check if we've reached the target position and rotation (like example.py does)
        current_pos, current_rot_y = controller.get_pose()
        distance_to_final_target = np.linalg.norm(current_pos - path[-1])
        rot_error = abs(current_rot_y - target_y_rotation)
        
        # Check if target reached with proper position AND rotation tolerance
        position_reached = distance_to_final_target < DEFAULT_CONFIG.final_target_tolerance
        rotation_reached = rot_error < np.degrees(DEFAULT_CONFIG.orientation_tolerance)
        target_fully_reached = position_reached and rotation_reached
        
        # Complete when target is reached (position AND rotation)
        # Also check the path progress is 100% to avoid early completion
        if target_fully_reached and progress >= 1.0:
            print(f"[Continuous Path] ✓ Target reached! Position: {distance_to_final_target:.4f}m, Rotation: {rot_error:.1f}°")
            print(f"[Continuous Path] Completed in {elapsed:.1f}s (estimated: {total_time:.1f}s)")
            return True
        
        # After estimated time, show extended waiting status
        if progress >= 1.0:
            # Show waiting status every 2 seconds after estimated completion time
            if int(elapsed) % 2 == 0 and elapsed > 0 and (elapsed - int(elapsed)) < (1.0 / update_frequency):
                extra_time = elapsed - total_time
                print(f"[Continuous Path] Waiting for target (+{extra_time:.1f}s): dist={distance_to_final_target:.4f}m, rot_err={rot_error:.1f}°")
        
        # Sleep for smooth updates at configured frequency
        # TODO: use precise timing
        time.sleep(1.0 / update_frequency)



def main():
    """Main path planning demonstration."""
    
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
    
    # Check command line arguments for logging mode
    enable_logging = len(sys.argv) > 1 and sys.argv[1].lower() == "log"
    
    if enable_logging:
        print("[LOGGING] Data logging ENABLED - !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("[LOGGING] Data logging DISABLED - use 'python example_astar.py log' to enable")
    
    # Initialize data logger only if logging is enabled
    data_logger = DataLogger() if enable_logging else None
    
    print(f"Point A: {point_a} (rotation: {point_a_rotation_deg}°)")
    print(f"Point B: {point_b} (rotation: {point_b_rotation_deg}°)")
    
    # Define static wall obstacles
    obstacles = define_static_walls()
    
    # Create hardware interface (real hardware only)
    print("Initializing hardware...")
    hardware = HardwareInterface()
    
    # Create controller with config from pathing_config.py
    config = ControllerConfig(control_frequency=path_config.update_frequency)
    
    print("Initializing controller (includes numba warmup)...")
    controller = ExcavatorController(hardware, config)
    
    # Start the background control loop
    controller.start()
    
    print("Starting A* path planning demonstration in 3 seconds...")
    time.sleep(3)
    
    try:
        # Continuous loop between points A and B
        cycle_count = 0
        current_start = point_a
        current_goal = point_b
        current_rotation = point_b_rotation_deg
        
        while True:  # Continuous loop
            cycle_count += 1
            direction = "A→B" if np.array_equal(current_start, point_a) else "B→A"
            print(f"\n{'='*60}")
            print(f"CYCLE #{cycle_count} - {direction}")
            print(f"{'='*60}")
            
            # Pause controller during A* calculation (hardware stays active)
            print("Pausing controller for A* path planning...")
            controller.pause() # this calls pwm.reset(reset_pump=False)!!!
            
            # Plan path with A* (controller paused - no IK/PID/PWM updates)
            print(f"Planning A* path from {current_start} to {current_goal}...")
            path = create_astar_3d_trajectory(
                start_pos=tuple(current_start),
                goal_pos=tuple(current_goal), 
                obstacle_data=obstacles,
                grid_resolution=path_config.grid_resolution,
                safety_margin=path_config.safety_margin,
                use_3d=True
            )
            
            print(f"A* found path with {len(path)} waypoints")
            
            # Set up data logging for this trajectory (if logging enabled)
            if data_logger is not None:
                # Calculate interpolated waypoints for metrics
                smooth_path = interpolate_path(path, interpolation_density)
                planned_distance = calculate_path_length(smooth_path)
                
                # Start trajectory tracking (matches simulation)
                data_logger.start_trajectory_tracking(
                    original_waypoints=len(path),
                    interpolated_waypoints=len(smooth_path)
                )
                data_logger.set_planned_distance(planned_distance)
                # Note: calculation_time will be set to 0 for now (A* timing not measured in real system)
                data_logger.set_calculation_time(0.0)
            
            # Resume controller and execute path
            print("Resuming controller...")
            controller.resume()
            print(f"Executing {direction} path with continuous motion...")
            success = execute_continuous_path(
                controller, 
                path, 
                current_rotation, 
                speed_mps=excavator_speed_mps,
                interpolation_factor=interpolation_density,
                data_logger=data_logger,
                update_frequency=path_config.update_frequency
            )
            
            if success:
                print(f"✓ Successfully completed {direction} path!")
                
                # Save trajectory log immediately after each completion (like simulation)
                if data_logger is not None:
                    data_logger.save_trajectory_log("astar_3d")
                    print(f"✓ Trajectory log saved for {direction}")
                
                # Switch direction for next cycle
                if np.array_equal(current_start, point_a):
                    # Next cycle: B→A
                    current_start = point_b
                    current_goal = point_a
                    current_rotation = point_a_rotation_deg
                else:
                    # Next cycle: A→B
                    current_start = point_a
                    current_goal = point_b
                    current_rotation = point_b_rotation_deg
                
                # Brief pause between cycles
                print("Pausing 2 seconds before next cycle...")
                time.sleep(2.0)
            else:
                print(f"✗ Failed to complete {direction} path")
                controller.stop()
                time.sleep(10)
                break
    
    except Exception as e:
        print(f"Error during A* path planning/execution: {e}")
        traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Note: Data logging now happens after each trajectory completion (like simulation)
        if data_logger is None:
            print("[LOGGING] No data logging performed (use 'log' argument to enable)")
        
        # Clean shutdown
        print("Stopping controller...")
        controller.stop() # this calls pwm.reset internally!!
        
        # Show final status
        final_pos, final_rot_y = controller.get_pose()
        print(f"Final position: {final_pos}")
        print(f"Final Y rotation: {final_rot_y:.1f}°")
        print("A* path planning test completed.")


if __name__ == "__main__":
    main()