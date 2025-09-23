#!/usr/bin/env python3
"""
Excavator Control Example

This demonstrates the interface:
- give_pose(position, rotation_y_deg): Set target
- get_pose(): Get current pose
- Controller runs autonomously in background

"""

import time
import numpy as np
from modules.excavator_controller import ExcavatorController, ControllerConfig


def main():
    """Simple excavator control"""
    
    # Get user input for target Y-axis rotation
    try:
        target_y_rotation_deg = float(input("Enter target Y-axis rotation in degrees (0 = horizontal): "))
        print(f"Target Y-axis rotation: {target_y_rotation_deg}°")
    except ValueError:
        print("Invalid input, using 0 degrees (horizontal)")
        target_y_rotation_deg = 0.0
    
    # Create hardware interface (real hardware only)
    print("Initializing hardware...")
    from modules.hardware_interface import HardwareInterface
    hardware = HardwareInterface()
    
    # Create controller with standard config
    config = ControllerConfig(
        control_frequency=60.0
    )
    
    print("Initializing controller (includes numba warmup)...")
    controller = ExcavatorController(hardware, config)
    
    # Start the background control loop
    controller.start()
    
    # Define target points
    point_a = [0.30, 0.0, 0.0]  # Point A
    point_b = [0.60, 0.0, 0.0]  # Point B
    
    print(f"Point A: {point_a}")
    print(f"Point B: {point_b}")
    print(f"Target Y rotation: {target_y_rotation_deg}°")
    print("Starting movement in 3 seconds...")
    time.sleep(3)
    
    try:
        # Movement sequence: A → B → A (continuous loop)
        targets = [point_a, point_b, point_a]
        target_names = ["A", "B", "A"]
        
        cycle_count = 0
        while True:  # Continuous loop
            cycle_count += 1
            print(f"\n=== Cycle #{cycle_count} ===")
            
            for target, name in zip(targets, target_names):
                print(f"\n--- Moving to Point {name} ---")
                
                # This is all you need to do! Set the target and wait
                controller.give_pose(target, target_y_rotation_deg)
                
                # Monitor progress until target is reached
                start_time = time.time()
                max_wait = 15.0
                
                while time.time() - start_time < max_wait:
                    # Get current pose
                    current_pos, current_rot_y = controller.get_pose()
                    
                    # Calculate distance to target
                    distance = np.linalg.norm(current_pos - np.array(target))
                    rot_error = abs(current_rot_y - target_y_rotation_deg)
                    
                    # Print status every 2 seconds to reduce debug spam
                    elapsed = time.time() - start_time
                    if int(elapsed/2) != int((elapsed - 0.1)/2):  # Every 2 seconds
                        joint_angles = controller.get_joint_angles()
                        print(f"\n📊 STATUS @{elapsed:.1f}s:")
                        print(f"  pos=[{current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f}], "
                              f"dist={distance:.3f}m, rot_y={current_rot_y:.1f}° (err={rot_error:.1f}°)")
                        print(f"  Joints: [{joint_angles[0]:.1f}°, {joint_angles[1]:.1f}°, {joint_angles[2]:.1f}°]")
                        print("=" * 60)
                    
                    # Just continue monitoring - let IK controller handle the movement
                    
                    time.sleep(1.0/60.0)  # Check 60 times per second (matches config)
                
                # After 15 seconds, move to next target
                current_pos, current_rot_y = controller.get_pose()
                distance = np.linalg.norm(current_pos - np.array(target))
                print(f"✓ Moving to next target after 15s at Point {name} (final distance: {distance:.3f}m)")
                
                # Pause between targets
                if target != targets[-1]:  # Don't pause after last target
                    print("Pausing for 3 seconds...")
                    time.sleep(3)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during movement: {e}")
    
    finally:
        # Clean shutdown
        print("Stopping controller...")
        controller.stop()
        hardware.reset(reset_pump=True)
        
        # Show final status
        final_pos, final_rot_y = controller.get_pose()
        print(f"Final position: {final_pos}")
        print(f"Final Y rotation: {final_rot_y:.1f}°")
        print("Test completed.")


if __name__ == "__main__":
    main()