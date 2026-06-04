"""Hall-sensor slew homing for IMU yaw zeroing.

The slew bearing has a Hall sensor that toggles HIGH inside a ~10-20° angular
band centered on mechanical zero. This module:

1. Reads the GPIO via lgpio (Raspberry Pi 5).
2. Drives the slew at a slow open-loop PWM (via ExcavatorController.direct mode)
   and watches for Hall transitions, recording the raw IMU yaw and command
   direction at each edge.
3. Computes the band center, drives back to it, and writes the result via
   ``HardwareInterface.set_base_yaw_offset_deg`` so all future yaw reads are
   zeroed at the mechanical reference.

Wiring convention (default config):
    - active_high=True  -> Hall reads 1 INSIDE the band, 0 OUTSIDE.
    - active_high=False -> inverted.

The routine handles three startup cases:
    (a) start OUTSIDE: drive +, wait rising edge (enter), continue until falling
        edge (exit other side), -> center = mean(rising, falling).
    (b) start INSIDE:  drive + past the exit edge, reverse and drive - past the
        opposite exit edge, -> center = mean(both exits, sign-aware).
    (c) start INSIDE but already drifted near an edge: same as (b); the
        recenter_band_factor parameter biases how far past each exit we travel
        before reversing, so we don't bounce inside the zone.

Direction tracking is essential because Hall is a boolean: only the sign of the
slew PWM at the transition instant tells us *which* edge we crossed. Backlash
is canceled by averaging same-direction edges across a full forward+reverse
sweep.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .differential_ik import extract_axis_rotation
from .quaternion_math import quat_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPIO backend
# ---------------------------------------------------------------------------


class HallReader:
    """Thin wrapper around lgpio for boolean Hall sensor reads.

    Holds an lgpio chip handle and exposes ``read()`` (debounced single sample)
    plus ``close()``. Tested target is Raspberry Pi 5 (gpio chip 4); pass
    ``chip=0`` for Pi 4/3.
    """

    def __init__(
        self,
        *,
        gpio_pin: int,
        active_high: bool = True,
        pull_up: bool = False,
        gpio_chip: int = 4,
    ) -> None:
        if gpio_pin is None or int(gpio_pin) < 0:
            raise ValueError(f"hall.gpio_pin not configured (got {gpio_pin})")
        self._pin = int(gpio_pin)
        self._active_high = bool(active_high)
        self._handle: Optional[int] = None
        try:
            import lgpio  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "lgpio not installed. On Raspberry Pi OS run: sudo apt install python3-lgpio"
            ) from exc
        self._lgpio = lgpio
        flags = lgpio.SET_PULL_UP if pull_up else 0
        self._handle = lgpio.gpiochip_open(int(gpio_chip))
        try:
            lgpio.gpio_claim_input(self._handle, self._pin, flags)
        except Exception:
            lgpio.gpiochip_close(self._handle)
            self._handle = None
            raise

    def read(self) -> bool:
        """Return True when Hall is INSIDE the active zone."""
        if self._handle is None:
            raise RuntimeError("HallReader is closed")
        raw = self._lgpio.gpio_read(self._handle, self._pin)
        return bool(raw) if self._active_high else not bool(raw)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._lgpio.gpio_free(self._handle, self._pin)
        finally:
            try:
                self._lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            self._handle = None

    def __enter__(self) -> "HallReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Yaw extraction helper (raw IMU yaw, decoupled from controller smoothing)
# ---------------------------------------------------------------------------


def _yaw_deg_from_quat(quat: np.ndarray) -> float:
    """Return world-Z twist of a [w, x, y, z] quaternion in degrees."""
    q = quat_normalize(np.asarray(quat, dtype=np.float32))
    yaw_rad = extract_axis_rotation(q, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    return float(np.degrees(yaw_rad))


def _wrap_deg(a: float) -> float:
    """Wrap to (-180, 180]."""
    return (float(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Public config / result
# ---------------------------------------------------------------------------


@dataclass
class HallHomingConfig:
    gpio_pin: int
    active_high: bool = True
    pull_up: bool = False
    gpio_chip: int = 4
    wiggle_speed: float = 0.20
    search_timeout_s: float = 10.0
    search_limit_deg: float = 30.0
    settle_time_s: float = 0.75
    poll_dt_s: float = 0.005
    recenter_band_factor: float = 1.5
    cached_band_width_deg: Optional[float] = None


@dataclass
class HallHomingResult:
    success: bool
    measured_center_yaw_deg: float = 0.0
    band_width_deg: float = 0.0
    edges: List[Tuple[str, float]] = field(default_factory=list)  # (label, yaw_deg)
    duration_s: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Controller adapter
# ---------------------------------------------------------------------------


class _SlewDriver:
    """Adapter that opens a direct-control window on ExcavatorController.

    Zeros all four valves but ``rotate``, which is driven at ``±speed``.
    Restores prior mode on ``close()``.
    """

    def __init__(self, controller, hardware) -> None:
        self._controller = controller
        self._hardware = hardware
        self._opened = False

    def __enter__(self) -> "_SlewDriver":
        self._controller.resume()
        self._controller.enter_direct_mode()
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        finally:
            if self._opened:
                try:
                    self._controller.exit_direct_mode()
                except Exception:
                    logger.exception("exit_direct_mode failed")
                try:
                    self._controller.pause()
                except Exception:
                    pass
            self._opened = False

    def drive(self, speed: float) -> None:
        """speed in [-1, 1]; positive = +Z slew, negative = -Z slew."""
        self._controller.give_direct_commands({
            'rotate': float(speed),
            'lift_boom': 0.0,
            'tilt_boom': 0.0,
            'scoop': 0.0,
        })

    def stop(self) -> None:
        self._controller.give_direct_commands({
            'rotate': 0.0, 'lift_boom': 0.0, 'tilt_boom': 0.0, 'scoop': 0.0,
        })

    def yaw_deg(self) -> Optional[float]:
        """Best-effort raw base-IMU yaw (degrees); None if IMU not ready."""
        base = self._hardware.read_base_imu()
        if base is None or 'quat' not in base:
            return None
        return _yaw_deg_from_quat(base['quat'])


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------


class HallSlewHoming:
    """Find the Hall-zone center and zero the IMU yaw to it.

    Designed to be invoked once at script startup after IMU has streamed
    enough samples to be ready. Caller passes a live ``ExcavatorController``
    (its background loop must be running) and the underlying
    ``HardwareInterface``.
    """

    def __init__(
        self,
        controller,
        hardware,
        config: HallHomingConfig,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._controller = controller
        self._hardware = hardware
        self._cfg = config
        self._progress = on_progress or (lambda msg: None)

    def run(self) -> HallHomingResult:
        cfg = self._cfg
        # Clear any previous offset so we work in raw yaw space.
        prev_offset = self._hardware.get_base_yaw_offset_deg()
        self._hardware.clear_base_yaw_offset()

        t_start = time.time()
        try:
            with HallReader(
                gpio_pin=cfg.gpio_pin,
                active_high=cfg.active_high,
                pull_up=cfg.pull_up,
                gpio_chip=cfg.gpio_chip,
            ) as hall:
                with _SlewDriver(self._controller, self._hardware) as drv:
                    result = self._sweep_and_center(hall, drv)
        except Exception as exc:
            # Restore prior offset on failure so we don't leave a half-applied state.
            if abs(prev_offset) > 1e-9:
                self._hardware.set_base_yaw_offset_deg(prev_offset)
            logger.exception("Hall homing failed: %s", exc)
            return HallHomingResult(
                success=False,
                duration_s=time.time() - t_start,
                reason=f"exception: {exc}",
            )

        result.duration_s = time.time() - t_start
        if not result.success:
            if abs(prev_offset) > 1e-9:
                self._hardware.set_base_yaw_offset_deg(prev_offset)
            return result

        # Apply the new offset: subsequent reads will report yaw=0 at center.
        self._hardware.set_base_yaw_offset_deg(result.measured_center_yaw_deg)
        self._progress(
            f"Homing OK: center_yaw={result.measured_center_yaw_deg:+.3f}deg "
            f"band_width={result.band_width_deg:.2f}deg in {result.duration_s:.1f}s"
        )
        return result

    # --- internals -------------------------------------------------------

    def _read(self, drv: _SlewDriver) -> Tuple[Optional[float], float]:
        """Return (yaw_deg or None, monotonic_ts)."""
        return drv.yaw_deg(), time.monotonic()

    def _wait_yaw_ready(self, drv: _SlewDriver, timeout_s: float = 5.0) -> float:
        """Block until base IMU yaw is readable. Raise on timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            y = drv.yaw_deg()
            if y is not None:
                return y
            time.sleep(0.02)
        raise RuntimeError("Base IMU yaw unavailable at homing start")

    def _drive_until_transition(
        self,
        hall: HallReader,
        drv: _SlewDriver,
        *,
        direction: int,
        from_state: bool,
        timeout_s: float,
    ) -> Tuple[float, bool, int]:
        """Drive at `direction * wiggle_speed` until Hall state flips.

        Returns (yaw_deg_at_transition, True, actual_direction) on success;
        (last_yaw, False, last_direction) on failure. Each search leg is
        bounded to cfg.search_limit_deg from the starting yaw; after the first
        leg it returns to the start yaw and tries the opposite direction.
        """
        cfg = self._cfg
        target_state = not from_state
        start_yaw = self._wait_yaw_ready(drv)
        last_yaw = start_yaw
        deadline = time.monotonic() + timeout_s

        for leg_idx, leg_direction in enumerate((direction, -direction)):
            drv.drive(leg_direction * cfg.wiggle_speed)
            while time.monotonic() < deadline:
                now_state = hall.read()
                if now_state == target_state:
                    y = drv.yaw_deg()
                    if y is None:
                        # Lost IMU mid-sweep; bail.
                        drv.stop()
                        return last_yaw, False, leg_direction
                    return y, True, leg_direction

                y = drv.yaw_deg()
                if y is not None:
                    last_yaw = y
                    if abs(_wrap_deg(y - start_yaw)) >= cfg.search_limit_deg:
                        drv.stop()
                        self._progress(
                            f"Hall search reached {cfg.search_limit_deg:.1f}deg "
                            f"from start yaw {start_yaw:+.2f}deg without transition"
                        )
                        break
                time.sleep(cfg.poll_dt_s)

            drv.stop()
            if leg_idx == 0 and time.monotonic() < deadline:
                self._progress(
                    f"Returning to start yaw {start_yaw:+.2f}deg before trying opposite direction"
                )
                if not self._drive_to_yaw(
                    drv,
                    start_yaw,
                    tol_deg=0.4,
                    timeout_s=min(10.0, max(0.1, deadline - time.monotonic())),
                ):
                    self._progress("ERROR: failed to return to Hall search start yaw")
                    return last_yaw, False, leg_direction
                if hall.read() == target_state:
                    y = drv.yaw_deg()
                    if y is not None:
                        return y, True, -leg_direction
            elif leg_idx == 1:
                break

        drv.stop()
        self._progress(
            f"ERROR: Hall transition not found within +/-{cfg.search_limit_deg:.1f}deg "
            f"from start yaw {start_yaw:+.2f}deg"
        )
        return last_yaw, False, -direction

    def _drive_to_yaw(
        self,
        drv: _SlewDriver,
        target_yaw_deg: float,
        *,
        tol_deg: float = 0.4,
        timeout_s: float = 10.0,
    ) -> bool:
        """Open-loop bang-bang: drive toward target_yaw, stop within tol."""
        cfg = self._cfg
        deadline = time.monotonic() + timeout_s
        prev_err: Optional[float] = None
        while time.monotonic() < deadline:
            y = drv.yaw_deg()
            if y is None:
                time.sleep(0.02)
                continue
            err = _wrap_deg(target_yaw_deg - y)
            if abs(err) <= tol_deg:
                drv.stop()
                return True
            # Sign-of-error drive. Slow speed since we're already close.
            sign = 1.0 if err > 0.0 else -1.0
            drv.drive(sign * cfg.wiggle_speed * 0.7)
            # If we overshot (sign flipped), shorten the next pulse.
            if prev_err is not None and (prev_err * err) < 0.0:
                time.sleep(cfg.poll_dt_s * 2)
            prev_err = err
            time.sleep(cfg.poll_dt_s)
        drv.stop()
        return False

    def _sweep_and_center(
        self,
        hall: HallReader,
        drv: _SlewDriver,
    ) -> HallHomingResult:
        """Two-sweep homing routine. See module docstring for the state cases."""
        cfg = self._cfg
        edges: List[Tuple[str, float]] = []
        edge_by_label = {}

        def record_edge(edge_type: str, actual_direction: int, yaw_deg: float) -> None:
            label = f"{edge_type}_{'+' if actual_direction > 0 else '-'}"
            edges.append((label, yaw_deg))
            edge_by_label[label] = yaw_deg

        start_yaw = self._wait_yaw_ready(drv)
        start_in_zone = hall.read()
        self._progress(
            f"Hall start: in_zone={start_in_zone}, raw_yaw={start_yaw:+.2f}deg"
        )

        # --- Phase 1: ensure we're OUTSIDE in +direction first ---------
        # If currently inside, drive + until we exit (falling edge of "inside").
        # If currently outside, we already are.
        if start_in_zone:
            y_exit_pos, ok, actual_dir = self._drive_until_transition(
                hall, drv, direction=+1, from_state=True,
                timeout_s=cfg.search_timeout_s,
            )
            if not ok:
                return HallHomingResult(
                    success=False, reason="timeout exiting zone in +dir", edges=edges,
                )
            record_edge("exit", actual_dir, y_exit_pos)
            drv.stop()
            time.sleep(0.1)

        # --- Phase 2: cross the band in the first available direction ---
        # Prefer - first; the bounded helper may reverse if the edge is not
        # within cfg.search_limit_deg from the starting yaw.
        y_enter_neg, ok, actual_dir = self._drive_until_transition(
            hall, drv, direction=-1, from_state=hall.read(),
            timeout_s=cfg.search_timeout_s,
        )
        if not ok:
            return HallHomingResult(
                success=False, reason="timeout finding -dir entry edge", edges=edges,
            )
        record_edge("enter", actual_dir, y_enter_neg)

        y_exit_neg, ok, actual_dir = self._drive_until_transition(
            hall, drv, direction=actual_dir, from_state=hall.read(),
            timeout_s=cfg.search_timeout_s,
        )
        if not ok:
            return HallHomingResult(
                success=False, reason="timeout finding -dir exit edge", edges=edges,
            )
        record_edge("exit", actual_dir, y_exit_neg)
        drv.stop()
        time.sleep(0.1)

        # --- Phase 3: cross back through the band in the opposite direction
        next_dir = -actual_dir
        y_enter_pos, ok, actual_dir = self._drive_until_transition(
            hall, drv, direction=next_dir, from_state=hall.read(),
            timeout_s=cfg.search_timeout_s,
        )
        if not ok:
            return HallHomingResult(
                success=False, reason="timeout finding +dir entry edge", edges=edges,
            )
        record_edge("enter", actual_dir, y_enter_pos)

        # The second exit closes the loop; capture it for symmetry.
        y_exit_pos2, ok, actual_dir = self._drive_until_transition(
            hall, drv, direction=actual_dir, from_state=hall.read(),
            timeout_s=cfg.search_timeout_s,
        )
        if not ok:
            return HallHomingResult(
                success=False, reason="timeout finding +dir exit edge", edges=edges,
            )
        record_edge("exit", actual_dir, y_exit_pos2)
        drv.stop()

        # --- Compute center -------------------------------------------
        # Backlash-aware: each direction has its own pair of (enter, exit)
        # edges. The midpoint of (enter, exit) in one direction approximates
        # the zone center as seen from that direction; averaging both
        # directions cancels mechanical backlash.
        required = ("enter_+", "exit_+", "enter_-", "exit_-")
        missing = [label for label in required if label not in edge_by_label]
        if missing:
            return HallHomingResult(
                success=False,
                reason=f"missing Hall edge(s) after bounded search: {', '.join(missing)}",
                edges=edges,
            )

        center_from_pos = 0.5 * (edge_by_label["enter_+"] + edge_by_label["exit_+"])
        center_from_neg = 0.5 * (edge_by_label["enter_-"] + edge_by_label["exit_-"])
        # Unwrap relative to start_yaw so wrap-around doesn't confuse the mean.
        c1 = start_yaw + _wrap_deg(center_from_pos - start_yaw)
        c2 = start_yaw + _wrap_deg(center_from_neg - start_yaw)
        center_yaw = 0.5 * (c1 + c2)

        # Band width: distance between entries (each direction's "inside").
        # Use absolute, wrap-aware.
        width_pos = abs(_wrap_deg(edge_by_label["exit_+"] - edge_by_label["enter_+"]))
        width_neg = abs(_wrap_deg(edge_by_label["exit_-"] - edge_by_label["enter_-"]))
        band_width = 0.5 * (width_pos + width_neg)

        # Sanity: live measurement vs cached.
        if cfg.cached_band_width_deg is not None and cfg.cached_band_width_deg > 0.0:
            rel_err = abs(band_width - cfg.cached_band_width_deg) / cfg.cached_band_width_deg
            if rel_err > 0.30:
                self._progress(
                    f"WARN: live band width {band_width:.2f}deg differs from "
                    f"cached {cfg.cached_band_width_deg:.2f}deg by {rel_err*100:.0f}%"
                )

        # Drive back to center for the final read.
        self._progress(f"Centering on {center_yaw:+.2f}deg (band={band_width:.2f}deg)")
        ok = self._drive_to_yaw(drv, center_yaw, tol_deg=0.4, timeout_s=10.0)
        if not ok:
            self._progress("WARN: center approach did not converge within tolerance")
        drv.stop()
        time.sleep(cfg.settle_time_s)

        # Re-read final yaw only to report centering error. The offset must use
        # the edge-derived center, otherwise bang-bang stop error becomes yaw
        # zero error.
        final_yaw = drv.yaw_deg()
        if final_yaw is None:
            return HallHomingResult(
                success=False,
                reason="lost base IMU after centering",
                edges=edges,
                band_width_deg=band_width,
            )
        center_error = _wrap_deg(final_yaw - center_yaw)
        self._progress(f"Center residual after stop: {center_error:+.3f}deg")

        return HallHomingResult(
            success=True,
            measured_center_yaw_deg=float(center_yaw),
            band_width_deg=float(band_width),
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Config loader (yaml -> HallHomingConfig)
# ---------------------------------------------------------------------------


def load_hall_config(control_config_yaml: dict) -> Optional[HallHomingConfig]:
    """Return a HallHomingConfig from the loaded control_config dict, or None.

    Returns None when 'hall' is missing, disabled, or has no pin.
    """
    hall_cfg = (control_config_yaml or {}).get("hall")
    if not isinstance(hall_cfg, dict):
        return None
    if not bool(hall_cfg.get("enabled", False)):
        return None
    pin = int(hall_cfg.get("gpio_pin", -1))
    if pin < 0:
        logger.warning("hall.enabled=true but hall.gpio_pin is unset; skipping homing")
        return None
    cached_band_width = hall_cfg.get("cached_band_width_deg")
    if cached_band_width is not None:
        cached_band_width = float(cached_band_width)
    return HallHomingConfig(
        gpio_pin=pin,
        active_high=bool(hall_cfg.get("active_high", True)),
        pull_up=bool(hall_cfg.get("pull_up", False)),
        gpio_chip=int(hall_cfg.get("gpio_chip", 4)),
        wiggle_speed=float(hall_cfg.get("wiggle_speed", 0.20)),
        search_timeout_s=float(hall_cfg.get("search_timeout_s", 10.0)),
        search_limit_deg=float(hall_cfg.get("search_limit_deg", 30.0)),
        settle_time_s=float(hall_cfg.get("settle_time_s", 0.75)),
        poll_dt_s=float(hall_cfg.get("poll_dt_s", 0.005)),
        recenter_band_factor=float(hall_cfg.get("recenter_band_factor", 1.5)),
        cached_band_width_deg=cached_band_width,
    )
