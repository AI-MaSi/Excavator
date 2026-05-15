"""Shared excavator IK target-state helpers for Cartesian and radial control.

This module intentionally stays above the low-level controller boundary.
The IRL controller and protocol still consume Cartesian poses; radial mode is a
task-space convenience layer that composes back into Cartesian commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class IKControlSpace(str, Enum):
    """Task-space input semantics for IK mode."""

    CARTESIAN = "cartesian"
    RADIAL = "radial"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _wrap_degrees(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    # Keep +180 instead of -180 for a stable human-facing representation.
    if math.isclose(wrapped, -180.0, abs_tol=1e-9):
        return 180.0
    return wrapped


@dataclass
class ExcavatorTargetState:
    """Unified real-side IK target state.

    The state keeps both Cartesian and radial views synchronized. Toggling
    between Cartesian and radial control therefore does not move the target; it
    only changes how the next operator delta is interpreted.
    """

    cartesian_x: float = 0.0
    cartesian_y: float = 0.0
    z: float = 0.0
    radius: float = 0.0
    slew_yaw_deg: float = 0.0
    rot_y_deg: float = 0.0

    @classmethod
    def from_cartesian_pose(
        cls,
        x: float,
        y: float,
        z: float,
        rot_y_deg: float = 0.0,
    ) -> "ExcavatorTargetState":
        state = cls()
        state.set_cartesian_pose(x, y, z, rot_y_deg)
        return state

    def copy(self) -> "ExcavatorTargetState":
        return ExcavatorTargetState(
            cartesian_x=self.cartesian_x,
            cartesian_y=self.cartesian_y,
            z=self.z,
            radius=self.radius,
            slew_yaw_deg=self.slew_yaw_deg,
            rot_y_deg=self.rot_y_deg,
        )

    def sync_from_cartesian_pose(self, position: Sequence[float], rot_y_deg: float = 0.0) -> None:
        if len(position) != 3:
            raise ValueError(f"Expected 3D position, got {len(position)} values")

        x = float(position[0])
        y = float(position[1])
        z = float(position[2])
        radius = math.hypot(x, y)
        slew_yaw_deg = _wrap_degrees(math.degrees(math.atan2(y, x))) if radius > 1e-12 else 0.0

        self.cartesian_x = x
        self.cartesian_y = y
        self.z = z
        self.radius = radius
        self.slew_yaw_deg = slew_yaw_deg
        self.rot_y_deg = float(rot_y_deg)

    def set_cartesian_pose(self, x: float, y: float, z: float, rot_y_deg: float = 0.0) -> None:
        self.sync_from_cartesian_pose((x, y, z), rot_y_deg)

    def set_radial_pose(self, radius: float, slew_yaw_deg: float, z: float, rot_y_deg: float = 0.0) -> None:
        radius = max(0.0, float(radius))
        slew_yaw_deg = _wrap_degrees(slew_yaw_deg)
        yaw_rad = math.radians(slew_yaw_deg)

        self.radius = radius
        self.slew_yaw_deg = slew_yaw_deg
        self.z = float(z)
        self.rot_y_deg = float(rot_y_deg)
        self.cartesian_x = radius * math.cos(yaw_rad)
        self.cartesian_y = radius * math.sin(yaw_rad)

    def apply_cartesian_delta(self, dx: float, dy: float, dz: float, d_rot_y_deg: float) -> None:
        self.set_cartesian_pose(
            self.cartesian_x + float(dx),
            self.cartesian_y + float(dy),
            self.z + float(dz),
            self.rot_y_deg + float(d_rot_y_deg),
        )

    def apply_radial_delta(self, d_radius: float, d_slew_yaw_deg: float, dz: float, d_rot_y_deg: float) -> None:
        self.set_radial_pose(
            self.radius + float(d_radius),
            self.slew_yaw_deg + float(d_slew_yaw_deg),
            self.z + float(dz),
            self.rot_y_deg + float(d_rot_y_deg),
        )

    def clamp_cartesian(
        self,
        workspace_limits: Mapping[str, float],
        *,
        min_rot_y_deg: float = -45.0,
        max_rot_y_deg: float = 45.0,
    ) -> None:
        self.set_cartesian_pose(
            _clamp(self.cartesian_x, workspace_limits["x_min"], workspace_limits["x_max"]),
            _clamp(self.cartesian_y, workspace_limits["y_min"], workspace_limits["y_max"]),
            _clamp(self.z, workspace_limits["z_min"], workspace_limits["z_max"]),
            _clamp(self.rot_y_deg, min_rot_y_deg, max_rot_y_deg),
        )

    def compose_cartesian_position(self) -> tuple[float, float, float]:
        return (self.cartesian_x, self.cartesian_y, self.z)

    def compose_cartesian_pose(self) -> tuple[float, float, float, float]:
        return (self.cartesian_x, self.cartesian_y, self.z, self.rot_y_deg)

    def compose_radial_pose(self) -> tuple[float, float, float, float]:
        return (self.radius, self.slew_yaw_deg, self.z, self.rot_y_deg)
