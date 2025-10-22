#!/usr/bin/env python3
"""
Live Forward Kinematics Demo

Reads live IMU data from excavator hardware and displays real-time joint angles
and end-effector position. No PWM control - just reads and displays.

Now includes joint velocity tracking with 10-second peak speed monitoring.

Usage: python fk_demo.py
"""

import time
import numpy as np
from collections import deque
from modules.hardware_interface import HardwareInterface
from modules.diff_ik import (create_excavator_config, get_end_effector_position,
                           get_joint_positions, apply_imu_offsets, extract_axis_rotation,
                           project_to_rotation_axes)
from modules.quaternion_math import quat_normalize

def main():
    print("LIVE EXCAVATOR FORWARD KINEMATICS")
    print("=" * 50)
    print("Reading live IMU data from hardware...")
    print("Press Ctrl+C to exit")
    print()
    
    # Initialize hardware (IMU only, no PWM control)
    try:
        hardware = HardwareInterface()
        robot_config = create_excavator_config()
        
        # Wait for hardware to be ready
        print("Waiting for hardware data (IMU + ADC/encoder)...")
        timeout = 10
        start_time = time.time()

        while not hardware.is_hardware_ready():
            if time.time() - start_time > timeout:
                print("ERROR: Hardware not ready after 10 seconds")
                return
            time.sleep(0.1)

        print("Hardware ready! Streaming live data...")
        print("-" * 60)

        # Initialize velocity tracking
        prev_joint_angles = None
        prev_time = None
        # Store (timestamp, velocities) for last 10 seconds
        velocity_history = deque(maxlen=1000)  # At 20Hz, 10sec = 200 samples (with margin)

        # Real-time loop
        while True:
            # Read live IMU data (3 IMUs: boom, arm, bucket)
            raw_imu_quats = hardware.read_imu_data()

            # Read slew quaternion directly from encoder (already computed in hardware)
            slew_quat = hardware.read_slew_quaternion()
            slew_angle = hardware.read_slew_angle()

            if raw_imu_quats is not None and len(raw_imu_quats) >= 3:
                # Combine slew + 3 IMUs to get all 4 joints
                all_quaternions = [slew_quat] + raw_imu_quats

                # Apply IMU mounting offset corrections
                corrected_quats = apply_imu_offsets(np.array(all_quaternions), robot_config)

                # Project to rotation axes
                projected_quats = project_to_rotation_axes(corrected_quats, robot_config)

                # Extract joint angles from quaternions (now 4 joints)
                joint_angles_rad = np.zeros(4, dtype=np.float32)
                for i in range(4):
                    axis = robot_config.rotation_axes[i]
                    joint_angles_rad[i] = extract_axis_rotation(projected_quats[i], axis)

                joint_angles_deg = np.degrees(joint_angles_rad)

                # Calculate joint velocities
                current_time = time.time()
                current_velocities = np.zeros(4, dtype=np.float32)  # rad/s
                current_velocities_deg = np.zeros(4, dtype=np.float32)  # deg/s

                if prev_joint_angles is not None and prev_time is not None:
                    dt = current_time - prev_time
                    if dt > 0:
                        # Calculate velocities (rad/s and deg/s)
                        current_velocities = (joint_angles_rad - prev_joint_angles) / dt
                        current_velocities_deg = np.degrees(current_velocities)

                        # Add to history with timestamp
                        velocity_history.append((current_time, current_velocities.copy()))

                        # Clean old entries (older than 10 seconds)
                        while velocity_history and (current_time - velocity_history[0][0]) > 10.0:
                            velocity_history.popleft()

                # Calculate max positive and negative velocities from last 10 seconds
                max_pos_vel = np.zeros(4, dtype=np.float32)
                max_neg_vel = np.zeros(4, dtype=np.float32)

                if velocity_history:
                    all_vels = np.array([v for _, v in velocity_history])
                    max_pos_vel = np.max(all_vels, axis=0)
                    max_neg_vel = np.min(all_vels, axis=0)

                # Update previous values
                prev_joint_angles = joint_angles_rad.copy()
                prev_time = current_time

                # Compute forward kinematics
                joint_positions = get_joint_positions(projected_quats, robot_config)
                ee_position = get_end_effector_position(projected_quats, robot_config)
                
                # Display results (clear screen for live update)
                print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                print("LIVE EXCAVATOR FORWARD KINEMATICS")
                print("=" * 50)
                print(f"Timestamp: {time.strftime('%H:%M:%S')}")
                print()
                
                print("Joint Angles (live):")
                print(f"  Slew:   {joint_angles_deg[0]:7.2f}° ({joint_angles_rad[0]:6.3f} rad)")
                print(f"  Boom:   {joint_angles_deg[1]:7.2f}° ({joint_angles_rad[1]:6.3f} rad)")
                print(f"  Arm:    {joint_angles_deg[2]:7.2f}° ({joint_angles_rad[2]:6.3f} rad)")
                print(f"  Bucket: {joint_angles_deg[3]:7.2f}° ({joint_angles_rad[3]:6.3f} rad)")
                print()

                print("Joint Velocities (live):")
                joint_names = ["Slew", "Boom", "Arm", "Bucket"]
                for i, name in enumerate(joint_names):
                    print(f"  {name:7s} Current: {current_velocities_deg[i]:7.2f}°/s  |  "
                          f"Max+: {np.degrees(max_pos_vel[i]):7.2f}°/s  |  "
                          f"Max-: {np.degrees(max_neg_vel[i]):7.2f}°/s  (10sec)")
                print()

                print("Joint Positions (end of each link):")
                print(f"  Boom mount:   [{joint_positions[0][0]*1000:6.1f}, {joint_positions[0][1]*1000:6.1f}, {joint_positions[0][2]*1000:6.1f}] mm  (end of slew link)")
                print(f"  Arm mount:    [{joint_positions[1][0]*1000:6.1f}, {joint_positions[1][1]*1000:6.1f}, {joint_positions[1][2]*1000:6.1f}] mm  (end of boom link)")
                print(f"  Bucket mount: [{joint_positions[2][0]*1000:6.1f}, {joint_positions[2][1]*1000:6.1f}, {joint_positions[2][2]*1000:6.1f}] mm  (end of arm link)")
                print(f"  Tool mount:   [{joint_positions[3][0]*1000:6.1f}, {joint_positions[3][1]*1000:6.1f}, {joint_positions[3][2]*1000:6.1f}] mm  (end of bucket link)")
                print()
                
                print("End-Effector Position: (Tool mount with ee_offset)")
                print(f"  Tool tip: [{ee_position[0]*1000:6.1f}, {ee_position[1]*1000:6.1f}, {ee_position[2]*1000:6.1f}] mm")
                print()
                print("Raw Sensor Data:")
                print(f"  Slew quat:    [{slew_quat[0]:6.3f}, {slew_quat[1]:6.3f}, {slew_quat[2]:6.3f}, {slew_quat[3]:6.3f}]")
                for i, quat in enumerate(raw_imu_quats[:3]):
                    print(f"  IMU {i+1} (q):   [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]  ({'Boom' if i==0 else 'Arm' if i==1 else 'Bucket'})")
                print()
                print("Press Ctrl+C to exit")
                
            else:
                print("Waiting for IMU data...")
                
            time.sleep(0.05)  # 20Hz update rate
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure hardware is connected and IMUs are powered on.")

if __name__ == "__main__":
    main()