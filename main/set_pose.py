#!/usr/bin/env python3
"""
Set Pose - Move to Target and Hold

Simple script to move the excavator to a target position and hold it there.

Usage:
    python set_pose.py <x> <y> <z>
    python set_pose.py <x> <y> <z> --interpolate 25

Example:
    python set_pose.py 0.4 0.0 0.1              # Direct movement
    python set_pose.py 0.4 0.0 0.1 --interpolate 50   # Interpolated at 50mm/s

Target tolerance: 5 degrees / 15mm
"""

import sys
import time
import argparse
import numpy as np
from modules.excavator_controller import ExcavatorController, ControllerConfig


def generate_interpolated_path(start_pos, end_pos, max_speed_mm_s, control_freq):
    """
    Generate waypoints for interpolated path with velocity limiting.

    Args:
        start_pos: Starting position [x, y, z] in meters
        end_pos: Target position [x, y, z] in meters
        max_speed_mm_s: Maximum speed in mm/s
        control_freq: Control frequency in Hz

    Returns:
        List of waypoints (each is [x, y, z] in meters)
    """
    start = np.array(start_pos)
    end = np.array(end_pos)

    # Calculate total distance in meters
    distance = np.linalg.norm(end - start)

    # Convert max speed to m/s
    max_speed_m_s = max_speed_mm_s / 1000.0

    # Calculate time needed to travel the distance
    travel_time = distance / max_speed_m_s

    # Calculate number of waypoints based on control frequency
    num_waypoints = int(travel_time * control_freq)

    # Ensure at least 2 waypoints (start and end)
    num_waypoints = max(2, num_waypoints)

    # Generate evenly spaced waypoints
    waypoints = []
    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        waypoint = start + t * (end - start)
        waypoints.append(waypoint.tolist())

    return waypoints


def main():
    """Move to target pose, hold for 3 seconds, then quit"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Move excavator to target position and hold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Direct movement:
    python set_pose.py 0.4 0.0 0.1

  Interpolated movement with velocity limiting:
    python set_pose.py 0.4 0.0 0.1 --interpolate 50
    python set_pose.py 0.4 0.0 0.1 --interpolate 25
        """
    )
    parser.add_argument('x', type=float, help='Target X position (meters)')
    parser.add_argument('y', type=float, help='Target Y position (meters)')
    parser.add_argument('z', type=float, help='Target Z position (meters)')
    parser.add_argument('--interpolate', type=float, metavar='SPEED',
                        help='Enable path interpolation with max speed (mm/s)')

    args = parser.parse_args()

    target_pos = [args.x, args.y, args.z]

    # Target tolerances
    POSITION_TOLERANCE = 0.010  # 10mm in meters
    ROTATION_TOLERANCE = 4.0    # 4 degrees
    HOLD_TIME = 5.0             # 3 seconds

    print(f"Target position: [{args.x}, {args.y}, {args.z}]")
    print(f"Tolerance: {POSITION_TOLERANCE*1000:.0f}mm position, {ROTATION_TOLERANCE:.0f}° rotation")
    print(f"Hold time: {HOLD_TIME:.0f}s")
    if args.interpolate is not None:
        print(f"Mode: INTERPOLATED (max speed: {args.interpolate:.1f} mm/s)")
    else:
        print(f"Mode: DIRECT")

    # Create hardware interface
    print("\nInitializing hardware...")
    from modules.hardware_interface import HardwareInterface
    hardware = HardwareInterface()

    # Create controller with standard config
    config = ControllerConfig(
        control_frequency=100.0
    )

    print("Initializing controller (includes numba warmup)...")
    controller = ExcavatorController(hardware, config, enable_perf_tracking=False)

    # Start the background control loop
    controller.start()
    print("Waiting for controller to initialize...")
    time.sleep(5.0)  # Allow some time for initialization

    try:
        # Choose movement mode
        if args.interpolate is not None:
            # Get initial position
            initial_pos, initial_rot_y = controller.get_pose()
            print(f"\nInitial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")

            # Hold current position for 5 seconds to stabilize hardware
            print("Stabilizing at current position for 5 seconds...")
            controller.give_pose(initial_pos.tolist(), initial_rot_y)

            stabilize_start = time.time()
            while time.time() - stabilize_start < 5.0:
                elapsed = time.time() - stabilize_start
                remaining = 5.0 - elapsed
                current_pos, _ = controller.get_pose()

                # Print status every 0.5 seconds
                if int(elapsed * 2) != int((elapsed - 0.1) * 2):  # Print every 0.5s
                    print(f"  Stabilizing... {remaining:.1f}s remaining | "
                          f"Pos=[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")

                time.sleep(0.1)

            # Get stabilized position as start position
            start_pos, _ = controller.get_pose()
            print(f"Stabilized position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")

            # Generate interpolated waypoints from stabilized position
            waypoints = generate_interpolated_path(
                start_pos, target_pos, args.interpolate, config.control_frequency
            )

            distance = np.linalg.norm(np.array(target_pos) - np.array(start_pos))
            print(f"\nDistance to target: {distance*1000:.1f}mm")
            print(f"Generated {len(waypoints)} waypoints for interpolated path")
            print(f"Estimated travel time: {len(waypoints) / config.control_frequency:.1f}s")
            print(f"\nMoving through waypoints at max {args.interpolate:.1f}mm/s...")

            # Send waypoints sequentially
            waypoint_interval = 1.0 / config.control_frequency
            for i, waypoint in enumerate(waypoints):
                controller.give_pose(waypoint, 0.0)
                if i < len(waypoints) - 1:  # Don't sleep after last waypoint
                    time.sleep(waypoint_interval)

                # Print progress every 10 waypoints
                if i % 10 == 0 or i == len(waypoints) - 1:
                    current_pos, _ = controller.get_pose()
                    progress = (i + 1) / len(waypoints) * 100
                    print(f"  Waypoint {i+1}/{len(waypoints)} ({progress:.0f}%): "
                          f"Target=[{waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f}] "
                          f"Actual=[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")

            print("\nInterpolation complete, waiting for final position to stabilize...")
        else:
            # Direct movement to target (original behavior)
            print(f"\nMoving to target: {target_pos}")
            controller.give_pose(target_pos, 0.0)

        # Wait until target is reached within tolerance
        print("Approaching target...\n")
        print("=" * 80)
        start_time = time.time()
        last_print_time = 0

        while True:
            current_pos, current_rot_y = controller.get_pose()

            # Calculate errors
            position_error = np.linalg.norm(current_pos - np.array(target_pos))
            rotation_error = abs(current_rot_y - 0.0)

            # Calculate per-axis errors
            error_x = current_pos[0] - target_pos[0]
            error_y = current_pos[1] - target_pos[1]
            error_z = current_pos[2] - target_pos[2]

            # Print detailed status every 0.5 seconds
            elapsed = time.time() - start_time
            if elapsed - last_print_time >= 0.5:
                # Get joint angles and hardware status
                joint_angles = controller.get_joint_angles()
                hw_status = controller.get_hardware_status()

                print(f"\n>>> APPROACHING TARGET - Elapsed: {elapsed:.1f}s <<<")
                print(f"\nCURRENT POSE:")
                print(f"  Position: [{current_pos[0]:+.3f}, {current_pos[1]:+.3f}, {current_pos[2]:+.3f}]m")
                print(f"  Rotation Y: {current_rot_y:+.2f}°")

                print(f"\nTARGET POSE:")
                print(f"  Position: [{target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f}]m")
                print(f"  Rotation Y: 0.00°")

                print(f"\nPOSITION ERROR:")
                print(f"  Total distance: {position_error*1000:.2f}mm")
                print(f"  X-axis: {error_x*1000:+.2f}mm | Y-axis: {error_y*1000:+.2f}mm | Z-axis: {error_z*1000:+.2f}mm")
                print(f"  Rotation error: {rotation_error:.2f}°")

                print(f"\nJOINT ANGLES:")
                print(f"  Slew:   {joint_angles[0]:+7.2f}° | Boom:   {joint_angles[1]:+7.2f}°")
                print(f"  Arm:    {joint_angles[2]:+7.2f}° | Bucket: {joint_angles[3]:+7.2f}°")

                print(f"\nHARDWARE STATUS:")
                print(f"  Slew Encoder: {hw_status.get('slew_angle', 0.0):.4f}rad "
                      f"({np.degrees(hw_status.get('slew_angle', 0.0)):+.2f}°)")

                # Show tolerance status
                pos_status = "✓" if position_error <= POSITION_TOLERANCE else "✗"
                rot_status = "✓" if rotation_error <= ROTATION_TOLERANCE else "✗"
                print(f"\nTOLERANCE CHECK:")
                print(f"  Position: {pos_status} {position_error*1000:.2f}mm / {POSITION_TOLERANCE*1000:.0f}mm")
                print(f"  Rotation: {rot_status} {rotation_error:.2f}° / {ROTATION_TOLERANCE:.0f}°")

                print("=" * 80)
                last_print_time = elapsed

            # Check if within tolerance
            if position_error <= POSITION_TOLERANCE and rotation_error <= ROTATION_TOLERANCE:
                print(f"\n{'='*80}")
                print(f"✓✓✓ TARGET REACHED in {elapsed:.1f}s ✓✓✓")
                print(f"{'='*80}")
                print(f"  Final position: [{current_pos[0]:+.3f}, {current_pos[1]:+.3f}, {current_pos[2]:+.3f}]m")
                print(f"  Final error: {position_error*1000:.2f}mm position, {rotation_error:.2f}° rotation")
                print(f"{'='*80}\n")
                break

            time.sleep(0.05)

        # Hold at target for specified duration
        print(f"Holding position for {HOLD_TIME:.0f} seconds...\n")
        hold_start = time.time()
        hold_print_time = 0

        while time.time() - hold_start < HOLD_TIME:
            current_pos, current_rot_y = controller.get_pose()
            position_error = np.linalg.norm(current_pos - np.array(target_pos))
            rotation_error = abs(current_rot_y - 0.0)

            # Calculate per-axis errors
            error_x = current_pos[0] - target_pos[0]
            error_y = current_pos[1] - target_pos[1]
            error_z = current_pos[2] - target_pos[2]

            hold_elapsed = time.time() - hold_start
            remaining = HOLD_TIME - hold_elapsed

            # Print detailed status every 0.5 seconds during hold
            if hold_elapsed - hold_print_time >= 0.5:
                joint_angles = controller.get_joint_angles()

                print(f">>> HOLDING [{remaining:.1f}s remaining] <<<")
                print(f"  Current:  [{current_pos[0]:+.3f}, {current_pos[1]:+.3f}, {current_pos[2]:+.3f}]m | Rot: {current_rot_y:+.2f}°")
                print(f"  Target:   [{target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f}]m | Rot: 0.00°")
                print(f"  Error:    X:{error_x*1000:+.1f}mm Y:{error_y*1000:+.1f}mm Z:{error_z*1000:+.1f}mm | "
                      f"Dist:{position_error*1000:.1f}mm | Rot:{rotation_error:.1f}°")
                print(f"  Joints:   [Slew={joint_angles[0]:+.1f}°, Boom={joint_angles[1]:+.1f}°, "
                      f"Arm={joint_angles[2]:+.1f}°, Bucket={joint_angles[3]:+.1f}°]")
                print()
                hold_print_time = hold_elapsed

            time.sleep(0.1)

        print(f"{'='*80}")
        print(f"✓ HOLD COMPLETE")
        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        print("\nStopping controller...")
        controller.stop()
        hardware.reset(reset_pump=True)

        # Show final status
        final_pos, final_rot_y = controller.get_pose()
        final_joint_angles = controller.get_joint_angles()

        print(f"\n{'='*80}")
        print("FINAL STATUS:")
        print(f"  Position: [{final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}]m")
        print(f"  Rotation Y: {final_rot_y:+.2f}°")
        print(f"  Joint Angles: [Slew={final_joint_angles[0]:+.1f}°, Boom={final_joint_angles[1]:+.1f}°, "
              f"Arm={final_joint_angles[2]:+.1f}°, Bucket={final_joint_angles[3]:+.1f}°]")
        print(f"{'='*80}")
        print("Done.")


if __name__ == "__main__":
    main()
