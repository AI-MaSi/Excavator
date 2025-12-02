#!/usr/bin/env python3
"""
Hardware path planning runner using normalized planner wrappers.
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from configuration_files.pathing_config import (
    DEFAULT_CONFIG,
    DEFAULT_ENV_CONFIG,
    EnvironmentConfig,
    PathExecutionConfig,
)
from pathing.path_planning_algorithms import (
    AStarParams,
    RRTParams,
    RRTStarParams,
    PRMParams,
    PathPlanningError,
    NoPathFoundError,
    CollisionError,
)
from pathing.normalized_planners import (
    NormalizerParams,
    create_astar_3d_trajectory_normalized,
    create_astar_plane_trajectory_normalized,
    create_rrt_trajectory_normalized,
    create_rrt_star_trajectory_normalized,
    create_prm_trajectory_normalized,
)
from pathing.path_utils import (
    precompute_cumulative_distances,
    interpolate_at_s,
    print_path_info,
)

from modules.hardware_interface import HardwareInterface
from modules.excavator_controller import ExcavatorController
from modules.quaternion_math import quat_from_y_deg


logger = logging.getLogger("run_hw_v2")


def _setup_logging(debug: bool) -> None:
    # Always set root logger to INFO to avoid numba debug spam
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Explicitly disable numba's verbose debug logging
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Make path planning internals chatty when debugging
    if debug:
        logging.getLogger("run_hw_v2").setLevel(logging.DEBUG)
        logging.getLogger("pathing.path_planning_algorithms").setLevel(logging.DEBUG)
        logging.getLogger("pathing.normalized_planners").setLevel(logging.DEBUG)
        logging.getLogger("modules.excavator_controller").setLevel(logging.DEBUG)
        logging.getLogger("modules.hardware_interface").setLevel(logging.DEBUG)


def _build_obstacles_from_env(env_config: EnvironmentConfig) -> List[Dict[str, Any]]:
    """Create obstacle list matching the wall config used in simulation."""
    obstacles: List[Dict[str, Any]] = []
    wall = {
        "size": np.asarray(env_config.wall_size, dtype=np.float32),
        "pos": np.asarray(env_config.wall_pos, dtype=np.float32),
        "rot": np.asarray(env_config.wall_rot, dtype=np.float32),
    }
    obstacles.append(wall)

    logger.info(
        "[Obstacles] Defined %d static walls (IRL/sim matched)", len(obstacles)
    )
    for i, obs in enumerate(obstacles, 1):
        size_mm = obs["size"] * 1000.0
        pos_mm = obs["pos"] * 1000.0
        logger.info("  Wall %d: size=[%.1f, %.1f, %.1f]mm, pos=[%.1f, %.1f, %.1f]mm",
                   i, size_mm[0], size_mm[1], size_mm[2],
                   pos_mm[0], pos_mm[1], pos_mm[2])

    return obstacles


class HardwareDataLogger:
    """Lightweight data logger using the same CSV layout as the simulator."""

    def __init__(self, algorithm_logging_name: str, base_log_dir: str) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, base_log_dir)
        os.makedirs(base_dir, exist_ok=True)

        algo_folders = [
            f
            for f in os.listdir(base_dir)
            if f.startswith(f"{algorithm_logging_name}_")
        ]
        if algo_folders:
            existing_nums = [
                int(f.split("_")[-1])
                for f in algo_folders
                if f.split("_")[-1].isdigit()
            ]
            folder_num = max(existing_nums) + 1 if existing_nums else 1
        else:
            folder_num = 1

        self.log_dir = os.path.join(
            base_dir, f"{algorithm_logging_name}_{folder_num}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.algorithm_logging_name = algorithm_logging_name
        self.trajectory_log: List[List[float]] = []
        self.trajectory_counter: int = 0

        # Tracking metrics
        self.execution_start_time: float = 0.0
        self.calculation_time: float = 0.0
        self.execution_time: float = 0.0
        self.total_distance_planned: float = 0.0
        self.total_distance_executed: float = 0.0
        self.max_tracking_error: float = 0.0
        self.avg_tracking_error: float = 0.0
        self.at_threshold_time: float | None = None

        # For progress fields
        self.current_waypoint_index: int = 0
        self.current_progress: float = 0.0
        self._prev_pos: np.ndarray | None = None

        logger.info(
            "[HW DataLogger] Initialized for algorithm '%s' at %s",
            algorithm_logging_name,
            self.log_dir,
        )

    def start_trajectory_tracking(
        self, calculation_time: float, total_distance_planned: float, path_config: PathExecutionConfig | None = None
    ) -> None:
        self.trajectory_log.clear()
        self.execution_start_time = time.time()
        self.calculation_time = float(calculation_time)
        self.total_distance_planned = float(total_distance_planned)
        self.total_distance_executed = 0.0
        self.max_tracking_error = 0.0
        self.avg_tracking_error = 0.0
        self.at_threshold_time = None
        self._prev_pos = None
        self.path_config = path_config

    def log_step(
        self,
        planned_pos: np.ndarray,
        actual_pos: np.ndarray,
        planned_quat: np.ndarray,
        actual_quat: np.ndarray,
        joint_angles: np.ndarray,
        waypoint_idx: int,
        progress: float,
    ) -> None:
        """Append a single hardware sample to the log."""
        self.current_waypoint_index = int(waypoint_idx)
        self.current_progress = float(progress)

        row = [
            float(planned_pos[0]),
            float(planned_pos[1]),
            float(planned_pos[2]),
            float(actual_pos[0]),
            float(actual_pos[1]),
            float(actual_pos[2]),
            float(planned_quat[0]),
            float(planned_quat[1]),
            float(planned_quat[2]),
            float(planned_quat[3]),
            float(actual_quat[0]),
            float(actual_quat[1]),
            float(actual_quat[2]),
            float(actual_quat[3]),
            float(joint_angles[0]),
            float(joint_angles[1]),
            float(joint_angles[2]),
            float(joint_angles[3]),
            int(self.current_waypoint_index),
            float(self.current_progress),
        ]
        self.trajectory_log.append(row)

        # Update metrics
        error = float(np.linalg.norm(actual_pos - planned_pos))
        if error > self.max_tracking_error:
            self.max_tracking_error = error

        if self._prev_pos is not None:
            self.total_distance_executed += float(
                np.linalg.norm(actual_pos - self._prev_pos)
            )
        self._prev_pos = actual_pos.copy()

    def mark_threshold_reached(self) -> None:
        """Mark the time when the robot first reached within target threshold."""
        if self.at_threshold_time is None:
            self.at_threshold_time = time.time()

    def save(self) -> None:
        """Save trajectory samples + metrics CSVs."""
        if not self.trajectory_log:
            return

        self.trajectory_counter += 1
        self.execution_time = time.time() - self.execution_start_time

        errors = [
            float(
                np.linalg.norm(
                    np.asarray(log[3:6], dtype=np.float32)
                    - np.asarray(log[0:3], dtype=np.float32)
                )
            )
            for log in self.trajectory_log
        ]
        self.avg_tracking_error = float(np.mean(errors)) if errors else 0.0

        # Trajectory CSV (same column order as sim logger)
        columns = [
            "x_g",
            "y_g",
            "z_g",
            "x_e",
            "y_e",
            "z_e",
            "quat_g_w",
            "quat_g_x",
            "quat_g_y",
            "quat_g_z",
            "quat_e_w",
            "quat_e_x",
            "quat_e_y",
            "quat_e_z",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "waypoint_idx",
            "progress",
        ]
        df = pd.DataFrame(self.trajectory_log, columns=columns)
        csv_path = os.path.join(
            self.log_dir,
            f"{self.algorithm_logging_name}_{self.trajectory_counter}.csv",
        )
        df.to_csv(csv_path, index=False)

        # Calculate time to reach threshold (excludes 5s settling time)
        at_threshold_time_s = None
        if self.at_threshold_time is not None:
            at_threshold_time_s = self.at_threshold_time - self.execution_start_time

        metrics = {
            "trajectory_id": self.trajectory_counter,
            "algorithm": self.algorithm_logging_name,
            "calculation_time_s": self.calculation_time,
            "execution_time_s": self.execution_time,
            "at_threshold_time_s": at_threshold_time_s,
            "trajectory_waypoints": len(self.trajectory_log),
            "simulation_steps": len(self.trajectory_log),
            "planned_distance_m": self.total_distance_planned,
            "executed_distance_m": self.total_distance_executed,
            "max_tracking_error_m": self.max_tracking_error,
            "avg_tracking_error_m": self.avg_tracking_error,
            "efficiency_ratio": (
                self.total_distance_executed / self.total_distance_planned
                if self.total_distance_planned > 0.0
                else 0.0
            ),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Add path configuration parameters if available
        if self.path_config is not None:
            # Motion parameters
            metrics["speed_mps"] = self.path_config.speed_mps
            metrics["dt"] = self.path_config.dt
            metrics["update_frequency"] = self.path_config.update_frequency

            # Planning parameters
            metrics["grid_resolution"] = self.path_config.grid_resolution
            metrics["safety_margin"] = self.path_config.safety_margin
            metrics["use_3d"] = self.path_config.use_3d

            # Tolerances
            metrics["final_target_tolerance"] = self.path_config.final_target_tolerance
            metrics["orientation_tolerance"] = self.path_config.orientation_tolerance

            # IK configuration
            metrics["ik_method"] = self.path_config.ik_method
            metrics["ik_command_type"] = self.path_config.ik_command_type
            metrics["ik_use_relative_mode"] = self.path_config.ik_use_relative_mode
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(self.log_dir, "metrics.csv")

        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)

        logger.info(
            "[HW DataLogger] Saved trajectory %d to %s",
            self.trajectory_counter,
            csv_path,
        )


@dataclass
class PlannedTrajectory:
    positions: np.ndarray
    total_distance_m: float
    calculation_time_s: float


def _plan_trajectory(
    algorithm: str,
    start_pos_world: np.ndarray,
    target_pos_world: np.ndarray,
    obstacles: List[Dict[str, Any]],
    path_config: PathExecutionConfig,
    enable_verbose: bool = False,
) -> PlannedTrajectory:
    """Use normalized planner wrappers to create a trajectory."""

    # Build NormalizerParams from path_config
    normalizer_params = NormalizerParams(
        speed_mps=path_config.speed_mps,
        dt=path_config.dt,
        return_poses=path_config.normalizer_return_poses,
        force_goal=path_config.normalizer_force_goal,
    )

    calc_start = time.time()

    if algorithm == "a_star":
        astar_params = AStarParams()
        result = create_astar_3d_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            astar_params=astar_params,
            normalizer_params=normalizer_params,
            verbose=enable_verbose,
        )
        algo_name = "A*"

    elif algorithm == "a_star_plane":
        astar_params = AStarParams()
        result = create_astar_plane_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            astar_params=astar_params,
            normalizer_params=normalizer_params,
            verbose=enable_verbose,
        )
        algo_name = "A* Plane"

    elif algorithm == "rrt":
        rrt_params = RRTParams(max_acceptable_cost=0.768)
        result = create_rrt_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            use_3d=path_config.use_3d,
            rrt_params=rrt_params,
            normalizer_params=normalizer_params,
        )
        algo_name = "RRT"

    elif algorithm == "rrt_star":
        rrt_star_params = RRTStarParams(max_acceptable_cost=0.768)
        result = create_rrt_star_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            use_3d=path_config.use_3d,
            rrt_params=rrt_star_params,
            normalizer_params=normalizer_params,
        )
        algo_name = "RRT*"

    elif algorithm == "prm":
        prm_params = PRMParams()
        result = create_prm_trajectory_normalized(
            start_pos=tuple(start_pos_world),
            goal_pos=tuple(target_pos_world),
            obstacle_data=obstacles,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            use_3d=path_config.use_3d,
            prm_params=prm_params,
            normalizer_params=normalizer_params,
        )
        algo_name = "PRM"

    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")

    calculation_time = time.time() - calc_start

    # Extract positions from normalized result
    exec_positions = np.asarray(result["exec_positions"], dtype=np.float32)
    total_length = float(result["total_length_m"][0])

    # Optionally ensure final waypoint is exactly at target (similar to original)
    if exec_positions.shape[0] > 0:
        last = exec_positions[-1]
        gap = float(np.linalg.norm(last - target_pos_world))
        min_append_threshold = (
            path_config.speed_mps * path_config.dt * 0.5
        )
        if gap > min_append_threshold:
            exec_positions = np.vstack(
                [exec_positions, target_pos_world.astype(np.float32)]
            )
            total_length += gap
            logger.info(
                "[%s] Appended final target (%.2f mm gap)",
                algo_name,
                gap * 1000.0,
            )
        else:
            logger.info(
                "[%s] Path already ends near target (%.2f mm)",
                algo_name,
                gap * 1000.0,
            )

    logger.info(
        "[%s] Path length: %.1f mm (total calculation time: %.2fs)",
        algo_name,
        total_length * 1000.0,
        calculation_time,
    )

    # High-level path info for quick sanity checks
    print_path_info(exec_positions, path_config, algo_name)

    return PlannedTrajectory(
        positions=exec_positions,
        total_distance_m=total_length,
        calculation_time_s=calculation_time,
    )


def _blind_move_to_start(
    controller: ExcavatorController,
    target_pos: np.ndarray,
    target_rot_deg: float,
    path_config: PathExecutionConfig,
    settle_time_s: float = 0.75,
    timeout_s: float = 15.0,
) -> None:
    """Drive directly to the starting pose (Point A) without any path planning."""
    target_pos = np.asarray(target_pos, dtype=np.float32)
    target_rot_deg = float(target_rot_deg)

    pos_tol = float(path_config.final_target_tolerance)
    rot_tol = float(np.degrees(path_config.orientation_tolerance))

    current_pos, current_rot_y = controller.get_pose()
    current_pos = np.asarray(current_pos, dtype=np.float32)
    pos_err = float(np.linalg.norm(current_pos - target_pos))
    rot_err = abs(float(current_rot_y) - target_rot_deg)

    if pos_err <= pos_tol and rot_err <= rot_tol:
        logger.info(
            "Already near Point A (pos err=%.1fmm, rot err=%.1fdeg).",
            pos_err * 1000.0,
            rot_err,
        )
        return

    logger.info("")
    logger.info("Moving to Point A (direct, no path planning)...")
    logger.info(
        "  Current: [%+.1f, %+.1f, %+.1f]mm, rot=%+.1fdeg",
        current_pos[0] * 1000.0,
        current_pos[1] * 1000.0,
        current_pos[2] * 1000.0,
        current_rot_y,
    )
    logger.info(
        "  Target:  [%+.1f, %+.1f, %+.1f]mm, rot=%+.1fdeg",
        target_pos[0] * 1000.0,
        target_pos[1] * 1000.0,
        target_pos[2] * 1000.0,
        target_rot_deg,
    )
    logger.info(
        "  Distance: %.1fmm, Rotation: %.1fdeg",
        pos_err * 1000.0,
        rot_err,
    )

    controller.resume()
    controller.give_pose(target_pos, target_rot_deg)

    start_time = time.time()
    settle_start: float | None = None

    while True:
        current_pos, current_rot_y = controller.get_pose()
        current_pos = np.asarray(current_pos, dtype=np.float32)

        pos_err = float(np.linalg.norm(current_pos - target_pos))
        rot_err = abs(float(current_rot_y) - target_rot_deg)

        if pos_err <= pos_tol and rot_err <= rot_tol:
            if settle_start is None:
                settle_start = time.time()
                logger.info(
                    "  Reached target (pos err=%.1fmm, rot err=%.1fdeg), settling...",
                    pos_err * 1000.0,
                    rot_err,
                )
            elif time.time() - settle_start >= settle_time_s:
                logger.info("Point A reached and settled")
                break
        else:
            settle_start = None

        if time.time() - start_time >= timeout_s:
            logger.warning(
                "[Blind Init] Timed out reaching Point A "
                "(pos_err=%.1fmm, rot_err=%.1fdeg after %.1fs)",
                pos_err * 1000.0,
                rot_err,
                timeout_s,
            )
            break

        time.sleep(0.05)

    controller.pause()


def _execute_on_hardware(
    controller: ExcavatorController,
    trajectory: PlannedTrajectory,
    target_y_rotation_deg: float,
    path_config: PathExecutionConfig,
    data_logger: HardwareDataLogger | None,
    enable_debug: bool,
) -> None:
    """Execute a planned path on the real hardware with continuous motion."""
    path = trajectory.positions
    if path.shape[0] < 2:
        logger.warning("Planned path has fewer than 2 waypoints; skipping.")
        return

    speed_mps = float(path_config.speed_mps)
    if speed_mps <= 0.0:
        raise RuntimeError("PathExecutionConfig.speed_mps must be > 0")

    cum_dist, total_length = precompute_cumulative_distances(path)
    total_time = total_length / speed_mps if speed_mps > 1e-9 else 0.0

    logger.info(
        "[Execute] Total length %.1f mm, estimated time %.1f s",
        total_length * 1000.0,
        total_time,
    )

    if data_logger is not None:
        data_logger.start_trajectory_tracking(
            calculation_time=trajectory.calculation_time_s,
            total_distance_planned=trajectory.total_distance_m,
            path_config=path_config,
        )

    update_frequency = float(path_config.update_frequency)
    if update_frequency <= 0.0:
        update_frequency = 50.0
    loop_period = 1.0 / update_frequency

    t0 = time.perf_counter()
    next_run_time = t0
    last_debug_print = 0.0
    s_cmd = 0.0
    target_reached_time: float | None = None

    while True:
        loop_start_time = time.perf_counter()
        elapsed = loop_start_time - t0

        # Read current state from controller (in base/world frame) for logging/termination checks
        current_pos, current_rot_y = controller.get_pose()
        current_pos = np.asarray(current_pos, dtype=np.float32)

        # Time-based progress along the precomputed path (match simulator behavior)
        if total_time > 1e-9:
            progress = min(elapsed / total_time, 1.0)
        else:
            progress = 1.0
        s_cmd = progress * total_length

        # Target point along path
        current_target = interpolate_at_s(path, cum_dist, s_cmd)
        controller.give_pose(current_target, float(target_y_rotation_deg))

        # Optional logging
        if data_logger is not None:
            hardware_joints = controller.get_joint_angles()
            joint_angles = np.zeros(4, dtype=np.float32)
            if len(hardware_joints) > 0:
                n = min(4, len(hardware_joints))
                joint_angles[:n] = hardware_joints[:n]

            waypoint_idx = int(progress * (len(path) - 1)) if len(path) > 1 else 0
            planned_quat = quat_from_y_deg(float(target_y_rotation_deg))
            actual_quat = quat_from_y_deg(float(current_rot_y))

            data_logger.log_step(
                planned_pos=current_target,
                actual_pos=current_pos,
                planned_quat=planned_quat,
                actual_quat=actual_quat,
                joint_angles=joint_angles,
                waypoint_idx=waypoint_idx,
                progress=progress,
            )

        # Periodic debug output
        if elapsed - last_debug_print >= path_config.progress_update_interval:
            last_debug_print = elapsed

            if enable_debug:
                # Get debug data
                joint_angles_dbg = controller.get_joint_angles()
                hw_status = controller.get_hardware_status()
                perf_stats = controller.get_performance_stats()

                pos_error = current_pos - current_target
                distance_to_target = float(np.linalg.norm(pos_error))
                rot_error = abs(float(current_rot_y) - float(target_y_rotation_deg))

                # Main progress line with performance metrics
                line = (
                    f"[Execute] Progress: {progress * 100.0:.1f}%  ({elapsed:.1f}s / {total_time:.1f}s) | "
                    f"Loop: {perf_stats.get('avg_loop_time_ms', 0):.2f}ms "
                    f"(min={perf_stats.get('min_loop_time_ms', 0):.2f} max={perf_stats.get('max_loop_time_ms', 0):.2f}) | "
                    f"CPU: {perf_stats.get('cpu_usage_pct', 0):.1f}% | "
                    f"Headroom: {perf_stats.get('avg_headroom_ms', 0):.2f}ms | "
                    f"Rate: {perf_stats.get('actual_hz', 0):.1f}Hz"
                )
                logger.info(line)

                # Stage breakdown line
                stages = []
                sensor_avg = perf_stats.get('avg_sensor_ms', 0)
                sensor_min = perf_stats.get('min_sensor_ms', 0)
                sensor_max = perf_stats.get('max_sensor_ms', 0)
                if sensor_avg > 0:
                    stages.append(f"Sensors: {sensor_avg:.2f}ms ({sensor_min:.2f}-{sensor_max:.2f})")

                ik_avg = perf_stats.get('avg_ik_fk_ms', 0)
                ik_min = perf_stats.get('min_ik_fk_ms', 0)
                ik_max = perf_stats.get('max_ik_fk_ms', 0)
                if ik_avg > 0:
                    stages.append(f"IK/FK: {ik_avg:.2f}ms ({ik_min:.2f}-{ik_max:.2f})")

                pwm_avg = perf_stats.get('avg_pwm_ms', 0)
                pwm_min = perf_stats.get('min_pwm_ms', 0)
                pwm_max = perf_stats.get('max_pwm_ms', 0)
                if pwm_avg > 0:
                    stages.append(f"PWM: {pwm_avg:.2f}ms ({pwm_min:.2f}-{pwm_max:.2f})")

                if stages:
                    logger.info("         " + " | ".join(stages))

                # Position and tracking error line
                logger.info(
                    "         Pos=[%+.1f, %+.1f, %+.1f]mm, "
                    "Target=[%+.1f, %+.1f, %+.1f]mm, "
                    "Err=%.1fmm",
                    current_pos[0] * 1000.0,
                    current_pos[1] * 1000.0,
                    current_pos[2] * 1000.0,
                    current_target[0] * 1000.0,
                    current_target[1] * 1000.0,
                    current_target[2] * 1000.0,
                    distance_to_target * 1000.0,
                )

                # Rotation line
                logger.info(
                    "         RotY=%.1fdeg (target=%.1fdeg, err=%.1fdeg)",
                    current_rot_y,
                    target_y_rotation_deg,
                    rot_error,
                )

                # Joint angles and velocity line
                if joint_angles_dbg and len(joint_angles_dbg) >= 4:
                    joint_line = "         Joints(deg): "
                    names = ['slew', 'boom', 'arm', 'bucket']
                    parts = [f"{n}={float(a):+.1f}" for n, a in zip(names, joint_angles_dbg[:4])]
                    joint_line += ", ".join(parts)

                    # Add velocity info if available
                    last_vel = perf_stats.get('last_joint_vel_degps', [])
                    if last_vel:
                        vel_parts = [f"{n}={v:+.1f}" for n, v in zip(names, last_vel)]
                        joint_line += " | Vel(deg/s): " + ", ".join(vel_parts)

                        # Add velocity cap if limiter is enabled
                        vel_cap = perf_stats.get('effective_vel_cap_degps', [])
                        vel_lim_on = perf_stats.get('ik_vel_lim_enabled', False)
                        if vel_lim_on and vel_cap:
                            cap_parts = [f"{n}={c:.1f}" for n, c in zip(names, vel_cap)]
                            joint_line += " | Cap(deg/s): " + ", ".join(cap_parts)

                    logger.info(joint_line)

                # Hardware status (compact)
                if hw_status:
                    logger.info("         HW: %s", hw_status)
            else:
                # Non-debug mode: simpler progress output
                logger.info(
                    "[Execute] Progress: %.1f%%  (%.1fs / %.1fs)",
                    progress * 100.0,
                    elapsed,
                    total_time,
                )

        # Completion check: both path progress and final target tolerance
        final_pos_error = current_pos - path[-1]
        distance_to_final = float(np.linalg.norm(final_pos_error))
        rot_error_final = abs(float(current_rot_y) - float(target_y_rotation_deg))

        position_reached = (
            distance_to_final < path_config.final_target_tolerance
        )
        rotation_reached = (
            rot_error_final
            < np.degrees(path_config.orientation_tolerance)
        )

        if hasattr(controller, "ik_config") and getattr(
            controller.ik_config, "command_type", "pose"
        ) == "position":
            target_fully_reached = position_reached
        else:
            target_fully_reached = position_reached and rotation_reached

        if target_fully_reached and progress >= 1.0:
            if target_reached_time is None:
                target_reached_time = time.time()
                if data_logger is not None:
                    data_logger.mark_threshold_reached()
                logger.info(
                    "[Execute] Target reached: "
                    "pos_err=[%.1f, %.1f, %.1f]mm (%.1fmm), rot_err=%.1fdeg. "
                    "Holding for 5.0s...",
                    final_pos_error[0] * 1000.0,
                    final_pos_error[1] * 1000.0,
                    final_pos_error[2] * 1000.0,
                    distance_to_final * 1000.0,
                    rot_error_final,
                )
            elif time.time() - target_reached_time >= 5.0:
                logger.info(
                    "[Execute] Completed in %.1fs (est %.1fs)", elapsed, total_time
                )
                break
        else:
            target_reached_time = None

        # If we have exceeded twice the estimated time and progress is 100%,
        # keep user informed but do not force-stop; the operator can abort.
        if progress >= 1.0 and elapsed > total_time:
            extra = elapsed - total_time
            logger.info(
                "[Execute] Waiting for target (extra +%.1fs, pos_err=%.1fmm, rot_err=%.1fdeg)",
                extra,
                distance_to_final * 1000.0,
                rot_error_final,
            )

        # Scheduler: drift-corrected sleep until next tick
        next_run_time += loop_period
        sleep_time = next_run_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_run_time = time.perf_counter()

    if data_logger is not None:
        data_logger.save()


def _run_direct_target_test(
    controller: ExcavatorController,
    targets: List[Tuple[np.ndarray, float]],
    path_config: PathExecutionConfig,
    dwell_seconds: float = 10.0,
    once: bool = False,
) -> None:
    """Directly command the raw A/B targets without any path planning."""
    pos_tol = float(path_config.final_target_tolerance)
    rot_tol = float(np.degrees(path_config.orientation_tolerance))

    controller.resume()
    cycle = 0
    try:
        while True:
            target_idx = cycle % len(targets)
            goal_pos, goal_rot_y = targets[target_idx]
            target_label = "B" if target_idx == 0 else "A"

            logger.info(
                "[Test] Commanding target %s: pos=[%.1f, %.1f, %.1f]mm, rot=%.1fdeg "
                "(dwell=%.1fs, no pathing)",
                target_label,
                goal_pos[0] * 1000.0,
                goal_pos[1] * 1000.0,
                goal_pos[2] * 1000.0,
                goal_rot_y,
                dwell_seconds,
            )

            controller.give_pose(goal_pos, goal_rot_y)
            dwell_start = time.time()
            last_status = dwell_start - 2.0  # force immediate status log
            reached_once = False

            while True:
                now = time.time()
                elapsed = now - dwell_start

                current_pos, current_rot_y = controller.get_pose()
                current_pos = np.asarray(current_pos, dtype=np.float32)
                pos_err = float(np.linalg.norm(current_pos - goal_pos))
                rot_err = abs(float(current_rot_y) - goal_rot_y)

                if not reached_once and pos_err <= pos_tol and rot_err <= rot_tol:
                    logger.info(
                        "[Test] Target %s reached: pos_err=%.1fmm, rot_err=%.1fdeg",
                        target_label,
                        pos_err * 1000.0,
                        rot_err,
                    )
                    reached_once = True

                if now - last_status >= 2.0:
                    logger.info(
                        "[Test] Holding %s (%.1fs/%.1fs): pos_err=%.1fmm, rot_err=%.1fdeg",
                        target_label,
                        elapsed,
                        dwell_seconds,
                        pos_err * 1000.0,
                        rot_err,
                    )
                    last_status = now

                if elapsed >= dwell_seconds:
                    logger.info(
                        "[Test] Dwell complete for %s; moving to next target.",
                        target_label,
                    )
                    break

                time.sleep(0.2)

            cycle += 1
            if once and cycle >= len(targets):
                logger.info("[Test] Single A/B sweep complete (--once); stopping.")
                break
    finally:
        controller.pause()


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real hardware path-planning demo V2 (using normalized wrappers)."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="a_star",
        choices=[
            "a_star",
            "a_star_plane",
            "rrt",
            "rrt_star",
            "prm",
        ],
        help="Path planning algorithm to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and performance metrics.",
    )
    parser.add_argument(
        "--log-data",
        action="store_true",
        help="Log trajectory data to CSV (for later tinkering).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single A->B->stop instead of continuous cycling.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Toggle between raw A/B targets every 10s (no path planning).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.debug)

    env_config = DEFAULT_ENV_CONFIG
    path_config = DEFAULT_CONFIG

    logger.info("="*60)
    logger.info("  HARDWARE PATH PLANNING RUNNER V2")
    logger.info("="*60)
    logger.info("Algorithm: %s", args.algorithm)
    logger.info(
        "Debug: %s, Data logging: %s",
        "ENABLED" if args.debug else "DISABLED",
        "ENABLED" if args.log_data else "DISABLED",
    )

    # Points A and B (base/world frame, shared with sim)
    point_a = np.asarray(env_config.point_a_pos, dtype=np.float32)
    point_b = np.asarray(env_config.point_b_pos, dtype=np.float32)
    rot_a = float(env_config.point_a_rotation_deg)
    rot_b = float(env_config.point_b_rotation_deg)

    logger.info("Point A: [%+.1f, %+.1f, %+.1f]mm (Y-rot=%+.1f deg)",
                point_a[0] * 1000.0, point_a[1] * 1000.0, point_a[2] * 1000.0, rot_a)
    logger.info("Point B: [%+.1f, %+.1f, %+.1f]mm (Y-rot=%+.1f deg)",
                point_b[0] * 1000.0, point_b[1] * 1000.0, point_b[2] * 1000.0, rot_b)

    obstacles = _build_obstacles_from_env(env_config)

    logger.info("Initializing hardware interface...")
    hardware = HardwareInterface(
        perf_enabled=bool(args.debug),
    )
    logger.info("Initializing excavator controller...")
    controller = ExcavatorController(
        hardware_interface=hardware,
        enable_perf_tracking=bool(args.debug),
        log_level="DEBUG" if args.debug else "INFO",
    )

    controller.start()
    logger.info("Control loop started.")

    # Brief wait for controller initialization
    time.sleep(2.0)

    # Pause controller before first planning cycle
    logger.info("Pausing controller before first path planning...")
    controller.pause()

    # Blindly move to the starting pose (Point A) so planning begins from a known state
    _blind_move_to_start(
        controller=controller,
        target_pos=point_a,
        target_rot_deg=rot_a,
        path_config=path_config,
    )

    algorithm_logging_name = args.algorithm.replace("_", "")
    data_logger: HardwareDataLogger | None = None
    if args.log_data and not args.test:
        data_logger = HardwareDataLogger(
            algorithm_logging_name=algorithm_logging_name,
            base_log_dir="logs_hw",
        )
    elif args.log_data and args.test:
        logger.info("Data logging disabled in --test mode (no trajectories to record).")

    # Cycle between A/B points using current pose as start.
    targets: List[Tuple[np.ndarray, float]] = [
        (point_b, rot_b),
        (point_a, rot_a),
    ]
    target_index = 0
    cycle = 0

    try:
        if args.test:
            test_dwell_seconds = 10.0
            logger.info(
                "Test mode enabled: direct A/B pose commands every %.1fs (no path planning).",
                test_dwell_seconds,
            )
            _run_direct_target_test(
                controller=controller,
                targets=targets,
                path_config=path_config,
                dwell_seconds=test_dwell_seconds,
                once=bool(args.once),
            )
            return 0

        while True:
            goal_pos, goal_rot_y = targets[target_index]
            cycle += 1
            target_label = "B" if target_index == 0 else "A"

            logger.info("")
            logger.info("="*60)
            logger.info("  CYCLE %d -> TARGET %s", cycle, target_label)
            logger.info("="*60)

            # Use current measured pose as start
            current_pos, current_rot_y = controller.get_pose()
            start_pos = np.asarray(current_pos, dtype=np.float32)
            logger.info(
                "Start: [%+.1f, %+.1f, %+.1f]mm (Y-rot=%+.1f deg)",
                start_pos[0] * 1000.0, start_pos[1] * 1000.0, start_pos[2] * 1000.0,
                current_rot_y
            )
            logger.info(
                "Goal:  [%+.1f, %+.1f, %+.1f]mm (Y-rot=%+.1f deg)",
                goal_pos[0] * 1000.0, goal_pos[1] * 1000.0, goal_pos[2] * 1000.0,
                goal_rot_y
            )

            # Calculate straight-line distance for reference
            straight_dist = float(np.linalg.norm(goal_pos - start_pos))
            logger.info("Straight-line distance: %.1f mm", straight_dist * 1000.0)

            try:
                # Plan trajectory while controller is paused
                trajectory = _plan_trajectory(
                    algorithm=args.algorithm,
                    start_pos_world=start_pos,
                    target_pos_world=goal_pos.astype(np.float32),
                    obstacles=obstacles,
                    path_config=path_config,
                    enable_verbose=bool(args.debug),  # Verbose planning only when debugging
                )
            except CollisionError:
                logger.error(
                    "[FATAL] Start or goal in collision; adjust A/B points or wall."
                )
                break
            except NoPathFoundError:
                logger.error(
                    "[FATAL] No path exists to target; configuration may be infeasible."
                )
                break
            except PathPlanningError as e:
                logger.error("[FATAL] Path planning error: %s", e)
                break

            # Resume controller after planning
            logger.info("Resuming controller for path execution...")
            controller.resume()

            # Brief wait for controller to stabilize after resume
            # IMPORTANT: Send START position (where we are now), NOT goal position!
            # This prevents PID windup from commanding a distant target immediately after reset.
            logger.info("Holding start position for 5.0s to stabilize after pump reset...")
            hold_end = time.time() + 5.0
            while time.time() < hold_end:
                controller.give_pose(start_pos, current_rot_y)
                time.sleep(0.05)

            _execute_on_hardware(
                controller=controller,
                trajectory=trajectory,
                target_y_rotation_deg=goal_rot_y,
                path_config=path_config,
                data_logger=data_logger,
                enable_debug=bool(args.debug),
            )

            # Pause controller after execution completes
            logger.info("Path execution complete. Pausing controller...")
            controller.pause()

            # Final position check
            final_pos, final_rot = controller.get_pose()
            final_error = float(np.linalg.norm(np.asarray(final_pos) - goal_pos))
            final_rot_error = abs(float(final_rot) - goal_rot_y)
            logger.info(
                "Final error: pos=%.1fmm, rot=%.1fdeg",
                final_error * 1000.0,
                final_rot_error
            )

            # Wait 5.0s for system to settle before next planning cycle
            logger.info("Waiting 5.0s for settling before next planning cycle...")
            time.sleep(5.0)

            target_index = (target_index + 1) % len(targets)

            if args.once and cycle >= 1:
                logger.info("Single cycle completed (--once); stopping.")
                break

    except KeyboardInterrupt:
        logger.info("\n\nKeyboardInterrupt detected, stopping controller...")
    finally:
        logger.info("")
        logger.info("="*60)
        logger.info("  SHUTTING DOWN")
        logger.info("="*60)
        try:
            final_pos, final_rot_y = controller.get_pose()
            logger.info(
                "Final position: [%+.3f, %+.3f, %+.3f]m",
                final_pos[0], final_pos[1], final_pos[2]
            )
            logger.info("Final rotation Y: %+.2f°", final_rot_y)
            logger.info("Total cycles completed: %d", cycle)

            controller.stop()
            logger.info("Controller stopped successfully")
        except Exception:
            logger.exception("Error while stopping controller", exc_info=True)

        logger.info("="*60)
        logger.info("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
