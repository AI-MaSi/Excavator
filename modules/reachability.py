"""Pre-flight reachability check for the excavator IK target.

Runs a simulated IK rollout from the current joint state on a deep-copied
solver, returning the closest pose the arm can reach. Used by
``ExcavatorController.give_pose`` to reject targets the geometry cannot
satisfy before any hydraulic motion starts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .differential_ik import (
    joint_angles_to_absolute_quaternions,
    forward_kinematics_with_ee_offset_core,
)
from .excavator_ik_utils import compute_relative_joint_angles
from .quaternion_math import quat_from_axis_angle, quat_multiply, quat_normalize


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    closest_position: np.ndarray
    pos_error_m: float
    iters: int
    final_cond_number: float


def _stall_detected(history: list, window: int, rel_improvement: float) -> bool:
    if len(history) < window + 1:
        return False
    recent = history[-(window + 1):]
    best_old = min(recent[:-1])
    latest = recent[-1]
    if best_old <= 1e-9:
        return True
    return (best_old - latest) / best_old < rel_improvement


def check_reachability(
    ik_controller,
    robot_config,
    current_joint_quats: np.ndarray | None = None,
    current_joint_angles: np.ndarray | None = None,
    target_pos: np.ndarray | None = None,
    target_rot_y_deg: float = 0.0,
    *,
    pos_tol: float = 0.005,
    max_iters: int = 80,
    cond_threshold: float = 0.0,
    dt: float = 0.01,
    stall_window: int = 10,
    stall_rel_improvement: float = 0.01,
) -> ReachabilityResult:
    """Simulate an IK rollout from canonical joint angles toward ``target_pos``.

    Operates on a deep copy of ``ik_controller`` so live solver state
    (``ee_pos_des``, ``last_condition_number``) is not disturbed.

    Args:
        ik_controller: Live IKController. Deep-copied internally.
        robot_config: Robot model.
        current_joint_quats: Legacy absolute quats. Used only to derive angles
            when ``current_joint_angles`` is not provided.
        current_joint_angles: Canonical relative joint angles (n_joints,).
        target_pos: Desired EE position [x, y, z].
        target_rot_y_deg: Desired pitch about Y in degrees.
        pos_tol: Convergence tolerance in meters.
        max_iters: Hard iteration cap.
        cond_threshold: Reject as unreachable if Jacobian condition exceeds
            this value at the converged pose. ``0`` disables the check.
        dt: Integration timestep handed to ``IKController.compute``.
        stall_window: Iterations to look back when measuring improvement.
        stall_rel_improvement: Minimum fractional reduction in pos_error
            over ``stall_window`` iterations; below this we early-exit.

    Returns:
        ReachabilityResult.
    """
    if target_pos is None:
        raise ValueError("target_pos is required")
    if current_joint_angles is None:
        if current_joint_quats is None:
            raise ValueError("current_joint_angles or current_joint_quats is required")
        current_joint_angles = compute_relative_joint_angles(current_joint_quats, robot_config)

    target_pos = np.asarray(target_pos, dtype=np.float32)
    angles = np.asarray(current_joint_angles, dtype=np.float32).copy()

    pitch_quat = quat_from_axis_angle(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.float32(np.radians(target_rot_y_deg)),
    )

    sim_ik = copy.deepcopy(ik_controller)
    sim_ik.ee_pos_des = target_pos.copy()

    err_history: list = []
    ee_pos = np.zeros(3, dtype=np.float32)

    iters_used = 0
    for i in range(max_iters):
        iters_used = i + 1
        quats = joint_angles_to_absolute_quaternions(angles, robot_config)
        _, ee_pos = forward_kinematics_with_ee_offset_core(
            quats,
            robot_config.link_lengths,
            robot_config.link_directions,
            robot_config.origin_offset,
            robot_config.ee_offset,
        )
        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        ee_quat = quats[-1].copy()
        # Live IK composes tool pitch with the current slew every control tick
        # so yaw error stays zero while slew remains free to chase position.
        slew_quat = quat_from_axis_angle(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.float32(angles[0]),
        )
        sim_ik.ee_quat_des = quat_normalize(quat_multiply(slew_quat, pitch_quat))

        err = float(np.linalg.norm(target_pos - ee_pos))
        err_history.append(err)

        if err <= pos_tol:
            break

        if _stall_detected(err_history, stall_window, stall_rel_improvement):
            break

        new_angles = sim_ik.compute(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            joint_angles=angles,
            joint_quats=quats,
            dt=dt,
        )
        angles = np.asarray(new_angles, dtype=np.float32)

    final_cond = float(getattr(sim_ik, "last_condition_number", 0.0))
    pos_error = float(np.linalg.norm(target_pos - ee_pos))

    cond_ok = (cond_threshold <= 0) or (final_cond <= cond_threshold)
    reachable = (pos_error <= pos_tol) and cond_ok

    return ReachabilityResult(
        reachable=reachable,
        closest_position=ee_pos.copy(),
        pos_error_m=pos_error,
        iters=iters_used,
        final_cond_number=final_cond,
    )
