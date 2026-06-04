#!/usr/bin/env python3
"""Hardware pathing + logging demo.

Mirrors the structure of ``Isaac-Pathing/scripts/masi/pathing/run_sim_v2.py`` so
that the per-trajectory CSV and ``metrics.csv`` files written here are directly
comparable with the simulation runs (same 38-column trajectory layout, same
metrics columns where available on real hardware).

Usage examples:
    sudo python run_hw_v2.py --algorithm a_star --task in-and-out --log --num-runs 10
    sudo python run_hw_v2.py --algorithm rrt -r --task rotation --log --debug
    sudo python run_hw_v2.py --test --num-runs 1   # direct A/B, no planner
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configuration_files.pathing_config import (
    DEFAULT_CONFIG,
    DEFAULT_ENV_CONFIG,
    EnvironmentConfig,
    PathExecutionConfig,
)
from modules.excavator_controller import ExcavatorController
from modules.hall_homing import HallSlewHoming, load_hall_config
from modules.hardware_interface import HardwareFaultError, HardwareInterface
from modules.quaternion_math import quat_from_axis_angle
from modules.rt_utils import SCHED_FIFO, apply_rt_to_thread, reset_to_normal
from pathing import path_planning_algorithms as planner_algos
from pathing import path_utils as path_utils_mod
from pathing.normalized_planners import (
    NormalizerParams,
    plan_radial_to_target,
    plan_to_target,
)
from pathing.path_planning_algorithms import (
    AStarParams,
    CollisionError,
    NoPathFoundError,
    PathPlanningError,
    PRMParams,
    RRTParams,
    RRTStarParams,
)
from pathing.path_utils import (
    interpolate_at_s,
    make_obstacle_data,
    precompute_cumulative_distances,
    print_path_info,
)


def _configured_workspace_bounds(
    obstacle_data,
    start_pos,
    goal_pos,
    padding: float = DEFAULT_CONFIG.workspace_padding,
    min_bounds: Tuple[float, float, float] = DEFAULT_CONFIG.workspace_min_bounds,
    max_bounds: Tuple[float, float, float] = DEFAULT_CONFIG.workspace_max_bounds,
):
    """Workspace bounds sourced from PathExecutionConfig (shared sim/IRL).

    Mirrors run_sim_v2.py:_configured_workspace_bounds. Bounds stretch to
    enclose start/goal/obstacles + padding, then clamp to be at least as
    wide as [min_bounds, max_bounds]. Tune in pathing_config.py.
    """
    points = [list(start_pos), list(goal_pos)]
    if obstacle_data:
        for obs in obstacle_data:
            points.append(list(obs.get("pos", obs.get("position", [0.0, 0.0, 0.0]))))
    pts = np.asarray(points, dtype=np.float32)
    bounds_min_calc = np.min(pts, axis=0) - padding
    bounds_max_calc = np.max(pts, axis=0) + padding
    bounds_min_calc = np.minimum(bounds_min_calc, np.asarray(min_bounds, dtype=np.float32))
    bounds_max_calc = np.maximum(bounds_max_calc, np.asarray(max_bounds, dtype=np.float32))
    return tuple(bounds_min_calc.tolist()), tuple(bounds_max_calc.tolist())


planner_algos.calculate_workspace_bounds = _configured_workspace_bounds
path_utils_mod.calculate_workspace_bounds = _configured_workspace_bounds


logger = logging.getLogger("run_hw_v2")


BASE_PLANNER_ALGORITHMS = ["a_star", "rrt", "rrt_star", "prm"]
TASK_NAMES = ["in-and-out", "rotation", "empty"]


# CSV column header — must stay aligned with run_sim_v2.py:TRAJECTORY_CSV_COLUMNS.
TRAJECTORY_CSV_COLUMNS = [
    "x_g", "y_g", "z_g", "x_e", "y_e", "z_e",
    "quat_g_w", "quat_g_x", "quat_g_y", "quat_g_z",
    "quat_e_w", "quat_e_x", "quat_e_y", "quat_e_z",
    "joint_1", "joint_2", "joint_3", "joint_4",
    "vel_1", "vel_2", "vel_3", "vel_4",
    "acc_1", "acc_2", "acc_3", "acc_4",
    "pos_des_1", "pos_des_2", "pos_des_3", "pos_des_4",
    "condition_number", "yoshikawa_index",
    "sv_1", "sv_2", "sv_3", "sv_4",
    "waypoint_idx", "progress",
]


def quat_from_y_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for pitch about the Y axis."""
    return quat_from_axis_angle(
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.float32(np.radians(float(angle_deg))),
    )


# ---------------------------------------------------------------------------
# Task presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskPreset:
    """Hardware task preset — coords match run_sim_v2.py for cross-domain comparison."""

    name: str
    goals: Tuple[Tuple[float, float, float], ...]
    rotations_deg: Tuple[float, ...]
    labels: Tuple[str, ...]
    use_wall_obstacle: bool = True


def make_task_preset(task_name: str) -> TaskPreset:
    if task_name == "in-and-out":
        env = DEFAULT_ENV_CONFIG
        return TaskPreset(
            name="in-and-out",
            goals=(tuple(env.point_a_pos), tuple(env.point_b_pos)),
            rotations_deg=(env.point_a_rotation_deg, env.point_b_rotation_deg),
            labels=("A_INSIDE", "B_OUTSIDE"),
        )
    if task_name == "rotation":
        return TaskPreset(
            name="rotation",
            goals=((-0.55, 0.25, -0.035), (-0.55, -0.25, -0.06)),
            rotations_deg=(0.0, 0.0),
            labels=("A_INSIDE_BEHIND", "B_OUTSIDE_BEHIND"),
        )
    if task_name == "empty":
        return TaskPreset(
            name="empty",
            goals=((-0.55, 0.25, -0.035), (-0.55, -0.25, -0.06)),
            rotations_deg=(0.0, 0.0),
            labels=("A_EMPTY_ROTATION", "B_EMPTY_ROTATION"),
            use_wall_obstacle=False,
        )
    raise ValueError(f"Unsupported task preset: {task_name}")


# ---------------------------------------------------------------------------
# CLI / logging
# ---------------------------------------------------------------------------


def _resolve_algorithm_name(base_algorithm: str, *, radial: bool, planar: bool) -> str:
    """Compose the variant name (e.g. ``radial_a_star_plane``)."""
    algorithm = base_algorithm
    if planar:
        algorithm = f"{algorithm}_plane"
    if radial:
        algorithm = f"radial_{algorithm}"
    return algorithm


def _logging_safe_name(algorithm_name: str) -> str:
    return algorithm_name.replace("_", "")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hardware path-planning demo (mirrors run_sim_v2.py)."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="a_star",
        choices=BASE_PLANNER_ALGORITHMS,
        help="Base planner to use.",
    )
    parser.add_argument("-r", "--radial", action="store_true", help="Wrap base planner with radial-first plan_radial_to_target.")
    parser.add_argument("-p", "--planar", action="store_true", help="Use the planar variant of the base planner.")
    parser.add_argument(
        "--radial-mode",
        type=str,
        default="reconstructed",
        choices=["reconstructed", "raw"],
        help="For radial mode, reconstruct the handoff path or skip reconstruction and execute the raw handoff plan.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="in-and-out",
        choices=TASK_NAMES,
        help="Task preset to run.",
    )
    parser.add_argument(
        "--obstacles-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Load obstacle list from a JSON file produced by run_sim_v2.py --dump-obstacles. "
             "Replaces the default wall obstacle for the chosen task.",
    )
    parser.add_argument("--log", action="store_true", help="Write trajectory CSV + metrics.csv under logs_hw/.")
    parser.add_argument("--debug", action="store_true", help="Verbose logging from controller, hardware, and planning internals.")
    parser.add_argument("--debug-planning", action="store_true", help="Verbose planning logs only.")
    parser.add_argument("--control-config", type=str, default="configuration_files/control_config.yaml", help="Controller/IK/IMU config YAML path.")
    parser.add_argument("--num-runs", type=int, default=None, metavar="N", help="Run N complete task sweeps, ending back at the first target.")
    parser.add_argument("--test", action="store_true", help="Direct A/B pose commands every 10s — no planner.")
    parser.add_argument("--rt-priority", type=int, default=75, help="RT priority for control loop (0=disable).")
    parser.add_argument("--imu-priority", type=int, default=70, help="RT priority for IMU thread (0=normal).")
    parser.add_argument("--adc-priority", type=int, default=70, help="Deprecated/ignored; run_hw_v2 does not initialize ADC.")
    return parser


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    args = _build_arg_parser().parse_args(argv)
    if args.num_runs is not None and args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    args.algorithm_name = _resolve_algorithm_name(
        args.algorithm, radial=args.radial, planar=args.planar
    )
    return args


def _configure_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # numba is excessively chatty at DEBUG; pin it to WARNING regardless.
    logging.getLogger("numba").setLevel(logging.WARNING)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for name in (
            "pathing.radial_planner",
            "pathing.normalized_planners",
            "pathing.path_planning_algorithms",
            "modules.excavator_controller",
            "modules.hardware_interface",
        ):
            logging.getLogger(name).setLevel(logging.DEBUG)
        return

    if args.debug_planning:
        for name in (
            "pathing.radial_planner",
            "pathing.normalized_planners",
        ):
            logging.getLogger(name).setLevel(logging.INFO)
        logging.getLogger("pathing.path_planning_algorithms").setLevel(logging.DEBUG)
        return

    logger.setLevel(logging.INFO)


def _load_control_config(path: str = "configuration_files/control_config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


def _load_obstacles_json(path: str) -> List[Dict[str, Any]]:
    """Load obstacles dumped by run_sim_v2.py --dump-obstacles.

    Accepts either the wrapped ``{"obstacles": [...]}`` payload or a bare list.
    Each entry is normalized via ``make_obstacle_data`` so size/pos/rot keys are
    canonical regardless of whether the dump used 'scale' or 'size'.
    """
    import json as _json
    with open(path, "r") as f:
        data = _json.load(f)
    raw = data.get("obstacles", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError(f"Obstacles JSON {path}: expected a list under 'obstacles' or top-level.")
    obstacles = []
    for i, entry in enumerate(raw):
        try:
            obstacles.append(make_obstacle_data(entry))
        except Exception as exc:
            raise ValueError(f"Obstacles JSON {path}: entry {i} invalid: {exc}") from exc
    if isinstance(data, dict) and "task" in data:
        logger.info(
            "[Obstacles] Loaded %d obstacles from %s (sim task=%s).",
            len(obstacles), path, data["task"],
        )
    else:
        logger.info("[Obstacles] Loaded %d obstacles from %s.", len(obstacles), path)
    for i, obs in enumerate(obstacles):
        size_mm = np.asarray(obs["size"]) * 1000.0
        pos_mm = np.asarray(obs["pos"]) * 1000.0
        logger.info(
            "  obs[%d]: size=[%.1f, %.1f, %.1f]mm, pos=[%+.1f, %+.1f, %+.1f]mm",
            i, size_mm[0], size_mm[1], size_mm[2], pos_mm[0], pos_mm[1], pos_mm[2],
        )
    return obstacles


def _build_obstacles_for_task(
    task: TaskPreset,
    env_config: EnvironmentConfig,
    obstacles_json_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if obstacles_json_path:
        return _load_obstacles_json(obstacles_json_path)

    if not task.use_wall_obstacle:
        logger.info("[Obstacles] Task '%s' runs obstacle-free.", task.name)
        return []

    wall = {
        "size": np.asarray(env_config.wall_size, dtype=np.float32),
        "pos": np.asarray(env_config.wall_pos, dtype=np.float32),
        "rot": np.asarray(env_config.wall_rot, dtype=np.float32),
    }
    size_mm = wall["size"] * 1000.0
    pos_mm = wall["pos"] * 1000.0
    logger.info(
        "[Obstacles] Wall: size=[%.1f, %.1f, %.1f]mm, pos=[%.1f, %.1f, %.1f]mm",
        size_mm[0], size_mm[1], size_mm[2],
        pos_mm[0], pos_mm[1], pos_mm[2],
    )
    return [wall]


# ---------------------------------------------------------------------------
# Planner dispatch
# ---------------------------------------------------------------------------


@dataclass
class PlannedTrajectory:
    positions: np.ndarray
    total_distance_m: float
    calculation_time_s: float
    chosen_variant: str = ""


def _make_normalizer(path_config: PathExecutionConfig) -> NormalizerParams:
    return NormalizerParams(
        speed_mps=path_config.speed_mps,
        dt=path_config.dt,
        return_poses=path_config.normalizer_return_poses,
        force_goal=path_config.normalizer_force_goal,
    )


def _plan_trajectory(
    *,
    base_algorithm: str,
    radial: bool,
    planar: bool,
    radial_mode: str,
    start_pos_world: np.ndarray,
    target_pos_world: np.ndarray,
    obstacles: List[Dict[str, Any]],
    path_config: PathExecutionConfig,
    enable_verbose: bool,
) -> PlannedTrajectory:
    """Dispatch to plan_to_target / plan_radial_to_target based on flags."""
    fallback = f"{base_algorithm}_plane" if planar else base_algorithm
    use_3d = not planar
    normalizer = _make_normalizer(path_config)

    astar_params = (
        AStarParams(max_iterations=path_config.astar_max_iterations)
        if base_algorithm == "a_star" else None
    )
    rrt_params = (
        RRTParams(
            max_iterations=path_config.rrt_max_iterations,
            max_step_size=path_config.rrt_max_step_size,
            goal_bias=path_config.rrt_goal_bias,
            goal_tolerance=path_config.rrt_goal_tolerance,
        )
        if base_algorithm == "rrt" else None
    )
    rrt_star_params = (
        RRTStarParams(
            max_iterations=path_config.rrt_max_iterations,
            max_step_size=path_config.rrt_max_step_size,
            goal_bias=path_config.rrt_goal_bias,
            goal_tolerance=path_config.rrt_goal_tolerance,
            rewire_radius=path_config.rrt_star_rewire_radius,
            minimum_iterations=path_config.rrt_star_min_iterations,
            cost_improvement_patience=path_config.rrt_star_cost_patience,
        )
        if base_algorithm == "rrt_star" else None
    )
    prm_params = (
        PRMParams(
            num_samples=path_config.prm_num_samples,
            connection_radius=path_config.prm_connection_radius,
            max_connections_per_node=path_config.prm_max_connections,
        )
        if base_algorithm == "prm" else None
    )

    calc_start = time.time()

    if radial:
        result = plan_radial_to_target(
            start_pos_world=tuple(start_pos_world),
            target_pos_world=tuple(target_pos_world),
            obstacle_data=obstacles,
            fallback_algorithm=fallback,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            normalizer_params=normalizer,
            astar_params=astar_params,
            rrt_params=rrt_params,
            rrt_star_params=rrt_star_params,
            prm_params=prm_params,
            use_3d=use_3d,
            verbose=enable_verbose,
            execute_raw=(radial_mode == "raw"),
            rdp_epsilon=path_config.radial_rdp_epsilon,
            radial_compensation_alpha=path_config.radial_compensation_alpha,
            smoothing_samples=path_config.radial_smoothing_samples,
        )
    else:
        result = plan_to_target(
            start_pos_world=tuple(start_pos_world),
            target_pos_world=tuple(target_pos_world),
            obstacle_data=obstacles,
            algorithm=fallback,
            grid_resolution=path_config.grid_resolution,
            safety_margin=path_config.safety_margin,
            normalizer_params=normalizer,
            astar_params=astar_params,
            rrt_params=rrt_params,
            rrt_star_params=rrt_star_params,
            prm_params=prm_params,
            use_3d=use_3d,
            verbose=enable_verbose,
        )

    calculation_time = time.time() - calc_start

    planning_stats = result.get("planning_stats") or {}
    if planning_stats:
        first_iter = planning_stats.get("first_solution_iteration")
        final_iter = planning_stats.get("final_iteration")
        max_iter = planning_stats.get("max_iterations")
        print(
            f"[Planner] {planning_stats.get('algorithm', fallback)} iterations: "
            f"first_path={first_iter}, returned_after={final_iter}/{max_iter}"
        )

    exec_positions = np.asarray(result["exec_positions"], dtype=np.float32)
    if exec_positions.ndim == 2 and exec_positions.shape[1] >= 7:
        exec_positions = exec_positions[:, :3].astype(np.float32)

    total_length_arr = result.get("total_length_m")
    total_length = (
        float(np.asarray(total_length_arr).flatten()[0])
        if total_length_arr is not None
        else 0.0
    )
    chosen_variant = str(result.get("chosen_variant", ""))

    # Snap the final waypoint to the goal if planner stopped short by more than half a step.
    if exec_positions.shape[0] > 0:
        last = exec_positions[-1]
        gap = float(np.linalg.norm(last - target_pos_world))
        min_append = path_config.speed_mps * path_config.dt * 0.5
        if gap > min_append:
            exec_positions = np.vstack([exec_positions, target_pos_world.astype(np.float32)])
            total_length += gap
            logger.info("Appended final target (%.2fmm gap)", gap * 1000.0)

    algo_label = _resolve_algorithm_name(base_algorithm, radial=radial, planar=planar)
    logger.info(
        "[%s] Path length: %.1fmm, calc=%.2fs, variant=%s",
        algo_label,
        total_length * 1000.0,
        calculation_time,
        chosen_variant or "n/a",
    )
    print_path_info(exec_positions, path_config, algo_label)

    return PlannedTrajectory(
        positions=exec_positions,
        total_distance_m=total_length,
        calculation_time_s=calculation_time,
        chosen_variant=chosen_variant,
    )


# ---------------------------------------------------------------------------
# Trajectory logger (sim-format CSV)
# ---------------------------------------------------------------------------


_PD_STIFFNESS = np.array([600.0, 600.0, 600.0, 600.0], dtype=np.float32)
_PD_DAMPING = np.array([40.0, 40.0, 40.0, 40.0], dtype=np.float32)


class HardwareTrajectoryLogger:
    """Per-trajectory CSV + cumulative metrics.csv, matching the sim layout."""

    def __init__(self, algorithm_name: str, base_log_dir: str = "logs_hw") -> None:
        script_dir = _ROOT
        base_dir = script_dir / base_log_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{algorithm_name}_"
        existing_nums: List[int] = []
        for entry in os.listdir(base_dir):
            if entry.startswith(prefix):
                suffix = entry[len(prefix):]
                if suffix.isdigit():
                    existing_nums.append(int(suffix))
        folder_num = max(existing_nums) + 1 if existing_nums else 1

        self.algorithm_name = algorithm_name
        self.log_dir = str(base_dir / f"{algorithm_name}_{folder_num}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.trajectory_counter = 0
        self._reset_accumulators()
        logger.info("[Logger] Logging to: %s", self.log_dir)

    def _reset_accumulators(self) -> None:
        self.trajectory_log: List[List[float]] = []
        self.execution_start_time: Optional[float] = None
        self.calculation_time: float = 0.0
        self.total_distance_planned: float = 0.0
        self.trajectory_waypoints: int = 0
        self.max_tracking_error: float = 0.0
        self.total_distance_executed: float = 0.0
        self.prev_ee_pos: Optional[np.ndarray] = None
        self.cumulative_abs_torque: float = 0.0
        self.cumulative_mechanical_work: float = 0.0
        self.cumulative_joint_velocity: float = 0.0
        self.cumulative_joint_accel: float = 0.0
        self.peak_torque: float = 0.0
        self.sum_condition_number: float = 0.0
        self.max_condition_number: float = 0.0
        self.sum_yoshikawa_index: float = 0.0
        self.min_yoshikawa_index: float = float("inf")
        self.ik_diagnostic_steps: int = 0
        self.at_threshold_time: Optional[float] = None
        self.prev_joint_vel: Optional[np.ndarray] = None
        self.prev_joint_vel_t: Optional[float] = None
        self.first_step_perf: Optional[float] = None
        self.last_step_perf: Optional[float] = None

    def start_trajectory(
        self,
        *,
        total_distance_planned: float,
        trajectory_waypoints: int,
        calculation_time: float,
    ) -> None:
        self._reset_accumulators()
        self.execution_start_time = time.time()
        self.total_distance_planned = float(total_distance_planned)
        self.trajectory_waypoints = int(trajectory_waypoints)
        self.calculation_time = float(calculation_time)

    def mark_threshold_reached(self) -> None:
        if self.at_threshold_time is None:
            self.at_threshold_time = time.time()

    def log_step(
        self,
        *,
        planned_pos: np.ndarray,
        actual_pos: np.ndarray,
        planned_quat: np.ndarray,
        actual_quat: np.ndarray,
        joint_pos_rad: np.ndarray,
        joint_vel_rad: np.ndarray,
        joint_pos_des_rad: np.ndarray,
        condition_number: float,
        yoshikawa_index: float,
        sv: np.ndarray,
        waypoint_idx: int,
        progress: float,
        sim_dt: float,
    ) -> None:
        # Acceleration: finite-diff over loop dt of joint velocity samples.
        now_t = time.perf_counter()
        if self.first_step_perf is None:
            self.first_step_perf = now_t
        self.last_step_perf = now_t
        if self.prev_joint_vel is not None and self.prev_joint_vel_t is not None:
            dt_acc = max(now_t - self.prev_joint_vel_t, 1e-6)
            joint_acc = (joint_vel_rad - self.prev_joint_vel) / dt_acc
        else:
            joint_acc = np.zeros(4, dtype=np.float32)
        self.prev_joint_vel = joint_vel_rad.copy()
        self.prev_joint_vel_t = now_t

        self.trajectory_log.append([
            float(planned_pos[0]), float(planned_pos[1]), float(planned_pos[2]),
            float(actual_pos[0]),  float(actual_pos[1]),  float(actual_pos[2]),
            float(planned_quat[0]), float(planned_quat[1]), float(planned_quat[2]), float(planned_quat[3]),
            float(actual_quat[0]),  float(actual_quat[1]),  float(actual_quat[2]),  float(actual_quat[3]),
            float(joint_pos_rad[0]), float(joint_pos_rad[1]), float(joint_pos_rad[2]), float(joint_pos_rad[3]),
            float(joint_vel_rad[0]), float(joint_vel_rad[1]), float(joint_vel_rad[2]), float(joint_vel_rad[3]),
            float(joint_acc[0]), float(joint_acc[1]), float(joint_acc[2]), float(joint_acc[3]),
            float(joint_pos_des_rad[0]), float(joint_pos_des_rad[1]), float(joint_pos_des_rad[2]), float(joint_pos_des_rad[3]),
            float(condition_number), float(yoshikawa_index),
            float(sv[0]), float(sv[1]), float(sv[2]), float(sv[3]),
            int(waypoint_idx),
            float(progress),
        ])

        # Tracking error / executed distance.
        error = float(np.linalg.norm(planned_pos - actual_pos))
        self.max_tracking_error = max(self.max_tracking_error, error)
        if self.prev_ee_pos is not None:
            self.total_distance_executed += float(np.linalg.norm(actual_pos - self.prev_ee_pos))
        self.prev_ee_pos = actual_pos.copy()

        # Approximate PD torque: τ ≈ k_p·(q_des − q) − k_d·q̇
        tau_approx = _PD_STIFFNESS * (joint_pos_des_rad[:4] - joint_pos_rad[:4]) - _PD_DAMPING * joint_vel_rad[:4]
        dt = max(sim_dt, 1e-6)
        self.cumulative_abs_torque += float(np.sum(np.abs(tau_approx))) * dt
        self.cumulative_mechanical_work += float(np.sum(np.abs(tau_approx * joint_vel_rad[:4]))) * dt
        self.cumulative_joint_velocity += float(np.sum(np.abs(joint_vel_rad[:4]))) * dt
        self.cumulative_joint_accel += float(np.sum(np.abs(joint_acc[:4]))) * dt
        self.peak_torque = max(self.peak_torque, float(np.max(np.abs(tau_approx))))

        if not math.isnan(condition_number):
            self.sum_condition_number += condition_number
            self.max_condition_number = max(self.max_condition_number, condition_number)
            self.ik_diagnostic_steps += 1
        if not math.isnan(yoshikawa_index) and yoshikawa_index > 0.0:
            self.sum_yoshikawa_index += yoshikawa_index
            self.min_yoshikawa_index = min(self.min_yoshikawa_index, yoshikawa_index)

    def save(
        self,
        *,
        task: TaskPreset,
        algorithm_name: str,
        path_config: PathExecutionConfig,
        env_config: EnvironmentConfig,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        obstacles_source: str = "wall",
        controller_perf_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.trajectory_log:
            return

        self.trajectory_counter += 1
        # Wall-clock execution time: start set inside _execute_on_hardware, so
        # calculation_time has already elapsed before it — do NOT subtract it.
        execution_time = (
            time.time() - self.execution_start_time
            if self.execution_start_time is not None
            else 0.0
        )
        sim_steps = len(self.trajectory_log)
        # Derive Hz from first/last step perf_counter timestamps — immune to
        # gaps between the final loop iteration and the save() call.
        if (
            self.first_step_perf is not None
            and self.last_step_perf is not None
            and sim_steps > 1
            and self.last_step_perf > self.first_step_perf
        ):
            sim_hz = (sim_steps - 1) / (self.last_step_perf - self.first_step_perf)
        else:
            sim_hz = None
        errors = [
            float(np.linalg.norm(np.array(row[3:6]) - np.array(row[0:3])))
            for row in self.trajectory_log
        ]
        avg_tracking_error = float(np.mean(errors)) if errors else 0.0
        at_threshold_time_s = (
            self.at_threshold_time - self.execution_start_time
            if self.at_threshold_time is not None and self.execution_start_time is not None
            else None
        )

        csv_path = os.path.join(
            self.log_dir, f"{self.algorithm_name}_{self.trajectory_counter}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRAJECTORY_CSV_COLUMNS)
            writer.writerows(self.trajectory_log)

        n = max(self.ik_diagnostic_steps, 1)
        ctrl_cfg = _load_control_config()
        ik_cfg = ctrl_cfg.get("ik", {}) if isinstance(ctrl_cfg.get("ik", {}), dict) else {}

        metrics = {
            "trajectory_id": self.trajectory_counter,
            "algorithm": algorithm_name,
            "data_source": "real",
            "task_name": task.name,
            "calculation_time_s": self.calculation_time,
            "execution_time_s": execution_time,
            "at_threshold_time_s": at_threshold_time_s,
            "trajectory_waypoints": self.trajectory_waypoints,
            "simulation_steps": sim_steps,
            "sim_hz": sim_hz,
            "planned_distance_m": self.total_distance_planned,
            "executed_distance_m": self.total_distance_executed,
            "max_tracking_error_m": self.max_tracking_error,
            "avg_tracking_error_m": avg_tracking_error,
            "efficiency_ratio": (
                self.total_distance_executed / self.total_distance_planned
                if self.total_distance_planned > 0
                else 0.0
            ),
            "speed_mps": path_config.speed_mps,
            "dt": path_config.dt,
            "grid_resolution": path_config.grid_resolution,
            "safety_margin": path_config.safety_margin,
            "use_3d": path_config.use_3d,
            "final_target_tolerance": path_config.final_target_tolerance,
            "orientation_tolerance": path_config.orientation_tolerance,
            "obstacles_source": obstacles_source,
            "obstacle_count": len(obstacles) if obstacles is not None else 0,
            "wall_size_x": env_config.wall_size[0],
            "wall_size_y": env_config.wall_size[1],
            "wall_size_z": env_config.wall_size[2],
            "wall_pos_x": env_config.wall_pos[0],
            "wall_pos_y": env_config.wall_pos[1],
            "wall_pos_z": env_config.wall_pos[2],
            "wall_rot_w": env_config.wall_rot[0],
            "wall_rot_x": env_config.wall_rot[1],
            "wall_rot_y": env_config.wall_rot[2],
            "wall_rot_z": env_config.wall_rot[3],
            "cumulative_abs_torque_Nms": self.cumulative_abs_torque,
            "cumulative_mech_work_J": self.cumulative_mechanical_work,
            "cumulative_joint_velocity_rad": self.cumulative_joint_velocity,
            "cumulative_joint_accel_rads": self.cumulative_joint_accel,
            "peak_torque_Nm": self.peak_torque,
            "avg_condition_number": self.sum_condition_number / n,
            "max_condition_number": self.max_condition_number,
            "avg_yoshikawa_index": self.sum_yoshikawa_index / n if self.ik_diagnostic_steps > 0 else float("nan"),
            "min_yoshikawa_index": self.min_yoshikawa_index if self.min_yoshikawa_index < float("inf") else float("nan"),
            "ik_method": ik_cfg.get("method"),
            "ik_command_type": ik_cfg.get("command_type"),
            "ik_velocity_mode": ik_cfg.get("velocity_mode"),
            "ik_velocity_error_gain": ik_cfg.get("velocity_error_gain"),
            "ik_use_rotational_velocity": ik_cfg.get("use_rotational_velocity"),
            "ik_use_relative_mode": ik_cfg.get("use_relative_mode"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # Controller-reported loop timing (authoritative, from RT thread).
            "controller_avg_loop_ms": (controller_perf_stats or {}).get("avg_loop_time_ms"),
            "controller_max_loop_ms": (controller_perf_stats or {}).get("max_loop_time_ms"),
            "controller_hz": (
                1000.0 / (controller_perf_stats or {})["avg_loop_time_ms"]
                if (controller_perf_stats or {}).get("avg_loop_time_ms")
                else None
            ),
        }

        metrics_path = os.path.join(self.log_dir, "metrics.csv")
        is_new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(metrics)

        logger.info("[Logger] Saved trajectory %d to %s", self.trajectory_counter, csv_path)


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------


def _read_controller_step(
    controller: ExcavatorController,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Snapshot controller state for the logger.

    Returns: (ee_pos, ee_rot_y_deg, joint_pos_rad, joint_vel_rad,
              condition_number, yoshikawa_index, singular_values)
    """
    pos, rot_y_deg = controller.get_pose()
    pos = np.asarray(pos, dtype=np.float32)

    angles_deg = controller.get_joint_angles()
    joint_pos_rad = np.zeros(4, dtype=np.float32)
    if angles_deg is not None and len(angles_deg) > 0:
        n = min(4, len(angles_deg))
        joint_pos_rad[:n] = np.radians(np.asarray(angles_deg[:n], dtype=np.float32))

    vel_degps = controller.get_joint_velocities_degps()
    joint_vel_rad = np.zeros(4, dtype=np.float32)
    if vel_degps is not None and len(vel_degps) > 0:
        n = min(4, len(vel_degps))
        joint_vel_rad[:n] = np.radians(np.asarray(vel_degps[:n], dtype=np.float32))

    try:
        cond = float(controller.get_condition_number())
        yoshikawa = float(controller.get_yoshikawa_index())
        sv = controller.get_singular_values()
    except Exception:
        cond = float("nan")
        yoshikawa = float("nan")
        sv = np.full(4, float("nan"), dtype=np.float32)

    return pos, float(rot_y_deg), joint_pos_rad, joint_vel_rad, cond, yoshikawa, sv


def _blind_move_to_start(
    controller: ExcavatorController,
    *,
    target_pos: np.ndarray,
    target_rot_deg: float,
    path_config: PathExecutionConfig,
    settle_time_s: float = 0.75,
    timeout_s: float = 15.0,
) -> None:
    target_pos = np.asarray(target_pos, dtype=np.float32)
    pos_tol = float(path_config.final_target_tolerance)
    rot_tol = float(np.degrees(path_config.orientation_tolerance))

    current_pos, current_rot_y = controller.get_pose()
    current_pos = np.asarray(current_pos, dtype=np.float32)
    pos_err = float(np.linalg.norm(current_pos - target_pos))
    rot_err = abs(float(current_rot_y) - float(target_rot_deg))

    if pos_err <= pos_tol and rot_err <= rot_tol:
        logger.info("Already near start (pos err=%.1fmm, rot err=%.1fdeg).", pos_err * 1000.0, rot_err)
        return

    logger.info("Moving to start pose (direct, no path planning)...")
    logger.info(
        "  Current=[%+.1f, %+.1f, %+.1f]mm rot=%+.1fdeg -> Target=[%+.1f, %+.1f, %+.1f]mm rot=%+.1fdeg",
        current_pos[0] * 1000.0, current_pos[1] * 1000.0, current_pos[2] * 1000.0, current_rot_y,
        target_pos[0] * 1000.0, target_pos[1] * 1000.0, target_pos[2] * 1000.0, target_rot_deg,
    )

    controller.resume()
    controller.give_pose(target_pos, float(target_rot_deg))

    start_time = time.time()
    settle_start: Optional[float] = None
    while True:
        current_pos, current_rot_y = controller.get_pose()
        current_pos = np.asarray(current_pos, dtype=np.float32)
        pos_err = float(np.linalg.norm(current_pos - target_pos))
        rot_err = abs(float(current_rot_y) - float(target_rot_deg))

        if pos_err <= pos_tol and rot_err <= rot_tol:
            if settle_start is None:
                settle_start = time.time()
                logger.info("  Reached start (pos err=%.1fmm, rot err=%.1fdeg), settling...", pos_err * 1000.0, rot_err)
            elif time.time() - settle_start >= settle_time_s:
                logger.info("Start pose reached and settled.")
                break
        else:
            settle_start = None

        if time.time() - start_time >= timeout_s:
            logger.warning(
                "[Blind Init] Timed out (pos_err=%.1fmm, rot_err=%.1fdeg).",
                pos_err * 1000.0, rot_err,
            )
            break
        time.sleep(0.05)

    controller.pause()


def _execute_on_hardware(
    controller: ExcavatorController,
    trajectory: PlannedTrajectory,
    target_y_rotation_deg: float,
    path_config: PathExecutionConfig,
    data_logger: Optional[HardwareTrajectoryLogger],
    enable_debug: bool,
) -> Dict[str, Any]:
    path = trajectory.positions
    if path.shape[0] < 2:
        logger.warning("Planned path has fewer than 2 waypoints; skipping.")
        return {}

    speed_mps = float(path_config.speed_mps)
    if speed_mps <= 0.0:
        raise RuntimeError("PathExecutionConfig.speed_mps must be > 0")

    cum_dist, total_length = precompute_cumulative_distances(path)
    total_time = total_length / speed_mps if speed_mps > 1e-9 else 0.0

    logger.info(
        "[Execute] Total length %.1fmm, estimated time %.1fs",
        total_length * 1000.0, total_time,
    )

    if data_logger is not None:
        data_logger.start_trajectory(
            total_distance_planned=trajectory.total_distance_m,
            trajectory_waypoints=int(path.shape[0]),
            calculation_time=trajectory.calculation_time_s,
        )

    update_frequency = float(path_config.update_frequency)
    if update_frequency <= 0.0:
        update_frequency = 50.0
    loop_period = 1.0 / update_frequency
    sim_dt = loop_period

    t0 = time.perf_counter()
    next_run_time = t0
    last_debug_print = 0.0
    target_reached_time: Optional[float] = None

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0

        ee_pos, ee_rot_y, joint_pos_rad, joint_vel_rad, cond, yoshikawa, sv = _read_controller_step(controller)

        if total_time > 1e-9:
            progress = min(elapsed / total_time, 1.0)
        else:
            progress = 1.0
        s_cmd = progress * total_length
        current_target = interpolate_at_s(path, cum_dist, s_cmd)
        controller.give_pose(current_target, float(target_y_rotation_deg))

        if data_logger is not None:
            waypoint_idx = int(progress * (len(path) - 1)) if len(path) > 1 else 0
            planned_quat = quat_from_y_deg(float(target_y_rotation_deg))
            actual_quat = quat_from_y_deg(float(ee_rot_y))
            # We don't have IK joint targets exposed; use measured joint pos as best-effort.
            joint_pos_des_rad = joint_pos_rad

            data_logger.log_step(
                planned_pos=current_target,
                actual_pos=ee_pos,
                planned_quat=planned_quat,
                actual_quat=actual_quat,
                joint_pos_rad=joint_pos_rad,
                joint_vel_rad=joint_vel_rad,
                joint_pos_des_rad=joint_pos_des_rad,
                condition_number=cond,
                yoshikawa_index=yoshikawa,
                sv=sv,
                waypoint_idx=waypoint_idx,
                progress=progress,
                sim_dt=sim_dt,
            )

        if elapsed - last_debug_print >= path_config.progress_update_interval:
            last_debug_print = elapsed
            err_mm = float(np.linalg.norm(ee_pos - current_target)) * 1000.0
            if enable_debug:
                perf_stats = controller.get_performance_stats()
                logger.info(
                    "[Execute] Progress=%.1f%% (%.1fs/%.1fs) | Loop=%.2fms (max=%.2f) | Err=%.1fmm | Cond=%.2f",
                    progress * 100.0, elapsed, total_time,
                    perf_stats.get("avg_loop_time_ms", 0.0),
                    perf_stats.get("max_loop_time_ms", 0.0),
                    err_mm, cond,
                )
            else:
                logger.info(
                    "[Execute] Progress=%.1f%% (%.1fs/%.1fs) | Err=%.1fmm",
                    progress * 100.0, elapsed, total_time, err_mm,
                )

        # Completion: fully reached and progress at 1.0 -> hold 5s for settling.
        final_err = float(np.linalg.norm(ee_pos - path[-1]))
        rot_err_final = abs(float(ee_rot_y) - float(target_y_rotation_deg))
        position_reached = final_err < path_config.final_target_tolerance
        rotation_reached = rot_err_final < np.degrees(path_config.orientation_tolerance)

        command_type = "pose"
        if hasattr(controller, "ik_config"):
            command_type = getattr(controller.ik_config, "command_type", "pose")
        target_fully_reached = (
            position_reached if command_type == "position"
            else position_reached and rotation_reached
        )

        if target_fully_reached and progress >= 1.0:
            if target_reached_time is None:
                target_reached_time = time.time()
                if data_logger is not None:
                    data_logger.mark_threshold_reached()
                logger.info(
                    "[Execute] Target reached: pos_err=%.1fmm rot_err=%.1fdeg. Holding 5s...",
                    final_err * 1000.0, rot_err_final,
                )
            elif time.time() - target_reached_time >= 5.0:
                logger.info("[Execute] Completed in %.1fs (est %.1fs)", elapsed, total_time)
                break
        else:
            target_reached_time = None

        next_run_time += loop_period
        sleep_time = next_run_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_run_time = time.perf_counter()

    try:
        perf_stats = controller.get_performance_stats() or {}
    except Exception:
        perf_stats = {}
    avg_ms = (perf_stats or {}).get("avg_loop_time_ms") or 0.0
    if avg_ms > 0.0:
        logger.info(
            "[Execute] Controller loop: avg=%.2fms, max=%.2fms (%.1fHz)",
            avg_ms,
            perf_stats.get("max_loop_time_ms", 0.0),
            1000.0 / avg_ms,
        )
    return perf_stats


def _run_direct_target_test(
    controller: ExcavatorController,
    targets: List[Tuple[np.ndarray, float]],
    path_config: PathExecutionConfig,
    *,
    dwell_seconds: float = 10.0,
    num_runs: Optional[int] = None,
) -> None:
    pos_tol = float(path_config.final_target_tolerance)
    rot_tol = float(np.degrees(path_config.orientation_tolerance))

    controller.resume()
    cycle = 0
    max_cycles = None if num_runs is None else int(num_runs) * len(targets)
    try:
        while True:
            target_idx = cycle % len(targets)
            goal_pos, goal_rot_y = targets[target_idx]
            label = chr(ord("A") + target_idx)

            logger.info(
                "[Test] Target %s: pos=[%.1f, %.1f, %.1f]mm rot=%.1fdeg (dwell=%.1fs)",
                label,
                goal_pos[0] * 1000.0, goal_pos[1] * 1000.0, goal_pos[2] * 1000.0,
                goal_rot_y, dwell_seconds,
            )
            controller.give_pose(goal_pos, goal_rot_y)
            dwell_start = time.time()
            last_status = dwell_start - 2.0
            reached_once = False

            while True:
                now = time.time()
                elapsed = now - dwell_start
                current_pos, current_rot_y = controller.get_pose()
                current_pos = np.asarray(current_pos, dtype=np.float32)
                pos_err = float(np.linalg.norm(current_pos - goal_pos))
                rot_err = abs(float(current_rot_y) - goal_rot_y)

                if not reached_once and pos_err <= pos_tol and rot_err <= rot_tol:
                    logger.info("[Test] Target %s reached (pos_err=%.1fmm rot_err=%.1fdeg)", label, pos_err * 1000.0, rot_err)
                    reached_once = True
                if now - last_status >= 2.0:
                    logger.info(
                        "[Test] Holding %s (%.1fs/%.1fs): pos_err=%.1fmm rot_err=%.1fdeg",
                        label, elapsed, dwell_seconds, pos_err * 1000.0, rot_err,
                    )
                    last_status = now
                if elapsed >= dwell_seconds:
                    logger.info("[Test] Dwell complete for %s; advancing.", label)
                    break
                time.sleep(0.2)

            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                logger.info("[Test] Completed %d full task run(s); stopping.", int(num_runs))
                break
    finally:
        controller.pause()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args)

    env_config = DEFAULT_ENV_CONFIG
    path_config = DEFAULT_CONFIG
    task = make_task_preset(args.task)

    logger.info("=" * 60)
    logger.info("  HARDWARE PATH PLANNING RUNNER V2")
    logger.info("=" * 60)
    logger.info("Task: %s | Algorithm: %s", args.task, args.algorithm_name)
    logger.info("Debug=%s, Log=%s, Num runs=%s", args.debug, args.log, args.num_runs)
    logger.info("Control config: %s", args.control_config)
    logger.info("RT priorities: ctrl=%d, imu=%d", args.rt_priority, args.imu_priority)

    targets: List[Tuple[np.ndarray, float]] = [
        (np.asarray(g, dtype=np.float32), float(r))
        for g, r in zip(task.goals, task.rotations_deg)
    ]
    for i, (pos, rot) in enumerate(targets):
        logger.info(
            "Goal %s (%s): [%+.1f, %+.1f, %+.1f]mm rot=%+.1fdeg",
            chr(ord("A") + i), task.labels[i],
            pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0, rot,
        )

    obstacles = _build_obstacles_for_task(
        task, env_config, obstacles_json_path=args.obstacles_json
    )
    if args.obstacles_json:
        obstacles_source = f"json:{os.path.basename(args.obstacles_json)}"
    elif obstacles:
        obstacles_source = "wall"
    else:
        obstacles_source = "none"

    logger.info("Initializing hardware interface...")
    hardware = HardwareInterface(
        control_config_file=args.control_config,
        perf_enabled=bool(args.debug),
        cleanup_disable_osc=False,
        enable_adc=False,
        start_adc_reader=False,
        imu_rt_priority=args.imu_priority,
    )

    logger.info("Initializing excavator controller...")
    controller = ExcavatorController(
        hardware_interface=hardware,
        enable_perf_tracking=True,
        log_level="DEBUG" if args.debug else "INFO",
        rt_priority=args.rt_priority,
        control_config_file=args.control_config,
    )
    controller.start()

    logger.info("Waiting 5.0s for IK/numba warmup...")
    time.sleep(5.0)

    if args.rt_priority > 0:
        logger.info("Applying SCHED_FIFO-%d to main thread...", args.rt_priority)
        if not apply_rt_to_thread(priority=args.rt_priority, policy=SCHED_FIFO, quiet=True):
            logger.warning("Failed to apply RT priority (run as root for RT scheduling).")

    logger.info("Waiting for IMU to become ready (Pico self-calibrates ~30 s stationary)...")
    _hw_wait_start = time.time()
    _hw_wait_timeout = 90.0
    _hw_last_status_log = -999.0
    while True:
        try:
            if hardware.is_hardware_ready():
                logger.info("Hardware ready after %.1f s.", time.time() - _hw_wait_start)
                break
        except HardwareFaultError as exc:
            logger.critical("Hardware fault during startup — cannot continue: %s", exc)
            try:
                hardware.shutdown()
            except Exception:
                pass
            return 1
        elapsed = time.time() - _hw_wait_start
        if elapsed >= _hw_wait_timeout:
            logger.critical(
                "Timed out waiting for hardware readiness after %.0f s. "
                "Check IMU connection and ensure robot is stationary during calibration.",
                elapsed,
            )
            try:
                hardware.shutdown()
            except Exception:
                pass
            return 1
        if elapsed - _hw_last_status_log >= 5.0:
            status = hardware.get_status()
            logger.info(
                "  IMU state: %s | %.0f s elapsed (timeout %.0f s)",
                status.get("imu_state", "unknown"), elapsed, _hw_wait_timeout,
            )
            _hw_last_status_log = elapsed
        time.sleep(0.5)

    controller.pause()

    hall_cfg = load_hall_config(_load_control_config(args.control_config))
    if hall_cfg is not None:
        logger.info(
            "Running Hall slew homing before pathing "
            "(limit=%.1fdeg, timeout=%.1fs)...",
            hall_cfg.search_limit_deg,
            hall_cfg.search_timeout_s,
        )
        homing = HallSlewHoming(
            controller=controller,
            hardware=hardware,
            config=hall_cfg,
            on_progress=lambda msg: logger.info("[Hall] %s", msg),
        )
        homing_result = homing.run()
        if not homing_result.success:
            logger.critical("Hall homing failed; aborting pathing run: %s", homing_result.reason)
            try:
                controller.stop()
            except Exception:
                pass
            try:
                hardware.shutdown()
            except Exception:
                pass
            reset_to_normal(quiet=True)
            return 1
    else:
        logger.info("Hall homing disabled or not configured; continuing without yaw zeroing.")

    # Drive to first goal as start pose so planning begins from a known state.
    start_pos, start_rot = targets[0]
    _blind_move_to_start(
        controller=controller,
        target_pos=start_pos,
        target_rot_deg=start_rot,
        path_config=path_config,
    )

    algorithm_log_name = _logging_safe_name(args.algorithm_name)
    data_logger: Optional[HardwareTrajectoryLogger] = None
    if args.log and not args.test:
        data_logger = HardwareTrajectoryLogger(algorithm_name=algorithm_log_name)
        if args.obstacles_json:
            import shutil
            try:
                shutil.copy2(
                    args.obstacles_json,
                    os.path.join(data_logger.log_dir, "obstacles.json"),
                )
                logger.info("[Logger] Copied obstacles JSON into %s", data_logger.log_dir)
            except Exception as exc:
                logger.warning("Could not copy obstacles JSON into log dir: %s", exc)
    elif args.log and args.test:
        logger.info("Data logging disabled in --test mode.")

    target_index = 1 % len(targets)  # we are AT goal 0; aim at next first
    cycle = 0
    max_cycles = None if args.num_runs is None else int(args.num_runs) * len(targets)

    try:
        if args.test:
            logger.info("Test mode: direct A/B pose commands every 10s (no planning).")
            _run_direct_target_test(
                controller=controller,
                targets=targets,
                path_config=path_config,
                dwell_seconds=10.0,
                num_runs=args.num_runs,
            )
            return 0

        while True:
            goal_pos, goal_rot_y = targets[target_index]
            label = task.labels[target_index]
            cycle += 1

            logger.info("")
            logger.info("=" * 60)
            logger.info("  CYCLE %d -> %s", cycle, label)
            logger.info("=" * 60)

            current_pos, current_rot_y = controller.get_pose()
            start_pos = np.asarray(current_pos, dtype=np.float32)
            logger.info(
                "Start: [%+.1f, %+.1f, %+.1f]mm rot=%+.1fdeg",
                start_pos[0] * 1000.0, start_pos[1] * 1000.0, start_pos[2] * 1000.0, current_rot_y,
            )
            logger.info(
                "Goal:  [%+.1f, %+.1f, %+.1f]mm rot=%+.1fdeg",
                goal_pos[0] * 1000.0, goal_pos[1] * 1000.0, goal_pos[2] * 1000.0, goal_rot_y,
            )
            straight_dist = float(np.linalg.norm(goal_pos - start_pos))
            logger.info("Straight-line distance: %.1fmm", straight_dist * 1000.0)

            try:
                trajectory = _plan_trajectory(
                    base_algorithm=args.algorithm,
                    radial=args.radial,
                    planar=args.planar,
                    radial_mode=args.radial_mode,
                    start_pos_world=start_pos,
                    target_pos_world=goal_pos,
                    obstacles=obstacles,
                    path_config=path_config,
                    enable_verbose=bool(args.debug),
                )
            except CollisionError:
                logger.error("[FATAL] Start or goal in collision; adjust task or wall.")
                break
            except NoPathFoundError:
                logger.error("[FATAL] No path found to target.")
                break
            except PathPlanningError as e:
                logger.error("[FATAL] Path planning error: %s", e)
                break

            controller.resume()

            # Hold start pose briefly to avoid PID windup on first command after resume.
            hold_end = time.time() + 1.0
            while time.time() < hold_end:
                controller.give_pose(start_pos, current_rot_y)
                time.sleep(0.05)

            ctrl_perf = _execute_on_hardware(
                controller=controller,
                trajectory=trajectory,
                target_y_rotation_deg=goal_rot_y,
                path_config=path_config,
                data_logger=data_logger,
                enable_debug=bool(args.debug),
            )

            if data_logger is not None:
                data_logger.save(
                    task=task,
                    algorithm_name=args.algorithm_name,
                    path_config=path_config,
                    env_config=env_config,
                    obstacles=obstacles,
                    obstacles_source=obstacles_source,
                    controller_perf_stats=ctrl_perf,
                )

            controller.pause()

            final_pos, final_rot = controller.get_pose()
            final_err = float(np.linalg.norm(np.asarray(final_pos) - goal_pos))
            final_rot_err = abs(float(final_rot) - goal_rot_y)
            logger.info("Final error: pos=%.1fmm rot=%.1fdeg", final_err * 1000.0, final_rot_err)

            logger.info("Settling 5.0s before next planning cycle...")
            time.sleep(5.0)

            target_index = (target_index + 1) % len(targets)
            if max_cycles is not None and cycle >= max_cycles:
                logger.info(
                    "Completed %d full task run(s) (%d planned legs); stopping at %s.",
                    int(args.num_runs), cycle, task.labels[0],
                )
                break

    except KeyboardInterrupt:
        logger.info("\n\nKeyboardInterrupt — stopping.")
    finally:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  SHUTTING DOWN")
        logger.info("=" * 60)
        try:
            final_pos, final_rot = controller.get_pose()
            logger.info(
                "Final position: [%+.3f, %+.3f, %+.3f]m rot=%+.2fdeg | cycles=%d",
                final_pos[0], final_pos[1], final_pos[2], final_rot, cycle,
            )
            controller.stop()
        except Exception:
            logger.exception("Error stopping controller")
        try:
            hardware.shutdown()
        except Exception:
            logger.exception("Error shutting down hardware")
        reset_to_normal(quiet=True)
        logger.info("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
