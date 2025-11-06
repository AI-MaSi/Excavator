#!/usr/bin/env python3
"""
Circle Pose Generator - Stream circular target poses for PID tuning

Generates target poses along a circle in the XZ plane at a fixed Y.
The controller follows successive targets, useful for tuning PIDs.

Configurable:
- diameter: circle diameter (meters)
- center: world position of circle center (cx, cy, cz)
- speed: target tangential velocity (mm/s)

Notes:
- End-effector rotation about Y is forced to 0 deg.
- Runs until interrupted (Ctrl-C). Prints target, current pose, error, and
  target velocity in mm/s (both set and step-estimated).
"""

import sys
import time
import math
import argparse
import numpy as np

from modules.excavator_controller import ExcavatorController, ControllerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream circular target poses for PID tuning"
    )

    parser.add_argument("--diameter", "-d", type=float, default=0.15,
                        help="Circle diameter in meters")
    parser.add_argument("--center-x", type=float, default=0.55,
                        help="Center X position in meters")
    parser.add_argument("--center-y", type=float, default=0.0,
                        help="Center Y position in meters")
    parser.add_argument("--center-z", type=float, default=-0.1,
                        help="Center Z position in meters")

    parser.add_argument("--speed", "-s", type=float, default=20.0,
                        help=(
                            "Target tangential speed along the circle in mm/s "
                        ))

    parser.add_argument("--dt", type=float, default=0.1,
                        help="Seconds between target updates")

    parser.add_argument("--clockwise", action="store_true",
                        help="Traverse circle clockwise")

    parser.add_argument("--relative", action="store_true",
                        help="Enable relative IK mode (pull toward target)")
    parser.add_argument("--rel-pos-gain", type=float, default=0.2,
                        help="Relative mode position pull gain per control step (0..1)")
    parser.add_argument("--rel-rot-gain", type=float, default=0.4,
                        help="Relative mode rotation pull gain per control step (0..1)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    diameter = float(args.diameter)
    radius = diameter / 2.0
    center = np.array([args.center_x, args.center_y, args.center_z], dtype=np.float32)
    
    # Velocity-based stepping
    v_mmps = float(args.speed)  # target tangential speed in mm/s
    dt = max(0.005, float(args.dt))

    direction = -1.0 if args.clockwise else 1.0

    circumference_m = math.tau * radius
    circumference_mm = circumference_m * 1000.0

    step_arc_mm = v_mmps * dt
    rev_time_s = (circumference_mm / v_mmps) if v_mmps > 1e-6 else float('inf')

    print("\nCircle configuration (XZ plane, Y fixed):")
    print(f"  Center: [{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}] m")
    print(f"  Diameter: {diameter:.3f} m  |  Radius: {radius:.3f} m")
    print(f"  Target speed: {v_mmps:.1f} mm/s  |  dt: {dt:.3f} s (~{1.0/dt:.1f} Hz)")
    print(f"  Step arc: {step_arc_mm:.1f} mm  |  Rev time: {rev_time_s:.2f} s")
    print(f"  Y-rotation (forced): 0.0 deg")

    # Initialize hardware and controller
    print("\nInitializing hardware and controller...")
    from modules.hardware_interface import HardwareInterface
    hardware = HardwareInterface()

    config = ControllerConfig(
        control_frequency=100.0
    )

    controller = ExcavatorController(hardware, config, enable_perf_tracking=False)
    controller.start()
    # Optional: enable relative IK control with gains
    if args.relative:
        controller.set_relative_control(True, pos_gain=args.rel_pos_gain, rot_gain=args.rel_rot_gain)

    try:
        print("\nStreaming circular targets. Press Ctrl-C to stop.\n")
        t0 = time.time()
        theta = 0.0
        last_target = None

        while True:
            # Compute angular step from desired tangential speed
            radius_mm = radius * 1000.0
            dtheta = 0.0 if radius_mm < 1e-6 else direction * (v_mmps / radius_mm) * dt
            theta = (theta + dtheta) % (2.0 * math.pi)

            # Circle in XZ plane at fixed Y
            target = center.copy()
            target[0] = center[0] + radius * math.cos(theta)  # X
            target[2] = center[2] + radius * math.sin(theta)  # Z
            # target[1] remains center[1]

            # Command target pose (force rotation to 0 deg)
            controller.give_pose(target, 0.0)

            # Read current pose and compute error
            current_pos, current_rot_y = controller.get_pose()
            pos_err_vec = (current_pos - target)
            pos_err_mm = np.linalg.norm(pos_err_vec) * 1000.0

            # Estimate target step velocity from last target (for debug)
            step_speed_mmps = None
            if last_target is not None:
                step_mm = float(np.linalg.norm(target - last_target) * 1000.0)
                step_speed_mmps = step_mm / dt
            last_target = target.copy()

            now = time.time()
            # Print each step (compact, one line)
            v_step_str = f"v_step {step_speed_mmps:6.1f} mm/s" if step_speed_mmps is not None else "v_step   n/a  mm/s"
            print(
                f"t+{now - t0:6.2f}s | v_set {v_mmps:6.1f} mm/s | {v_step_str} | "
                f"target [{target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}] m | "
                f"current [{current_pos[0]:+.3f},{current_pos[1]:+.3f},{current_pos[2]:+.3f}] m | "
                f"err {pos_err_mm:6.1f} mm"
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()
        hardware.reset(reset_pump=True)
        # Final status
        final_pos, final_rot_y = controller.get_pose()
        print("\nFinal pose:")
        print(f"  Position: [{final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}] m")
        print(f"  Rotation Y: {final_rot_y:+.2f} deg")


if __name__ == "__main__":
    main()
