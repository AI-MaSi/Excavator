#!/usr/bin/env python3
"""
GUI Position Receiver - Receive position commands from excavator_gui.py

Receives XYZ position commands via UDP and sends them to the excavator controller.
Sends back live joint positions for visualization in the GUI.

Compatible with the 16-bit encoding (2 bytes per value) from excavator_gui.py.

Usage:
    python gui_position_receiver.py
"""

import time
import numpy as np
from modules.udp_socket import UDPSocket
from modules.excavator_controller import ExcavatorController, ControllerConfig
from modules.diff_ik import (create_excavator_config, get_joint_positions,
                            apply_imu_offsets, extract_axis_rotation,
                            project_to_rotation_axes)


def decode_bytes_to_position(bytes_list):
    """
    Decode 8 bytes back to position and rotation values [x, y, z, rot_y].

    This matches the encoding in excavator_gui.py.

    Args:
        bytes_list: List of 8 signed bytes [-128, 127]

    Returns:
        Tuple: ([x, y, z], rot_y) - position in meters, rotation in degrees
    """
    SCALE_POS = 13107.0
    SCALE_ROT = 182.0

    # Convert signed bytes back to unsigned
    def to_unsigned_byte(b):
        return b if b >= 0 else b + 256

    bytes_unsigned = [to_unsigned_byte(b) for b in bytes_list]

    # Reconstruct 16-bit integers
    x_int = (bytes_unsigned[0] << 8) | bytes_unsigned[1]
    y_int = (bytes_unsigned[2] << 8) | bytes_unsigned[3]
    z_int = (bytes_unsigned[4] << 8) | bytes_unsigned[5]
    rot_int = (bytes_unsigned[6] << 8) | bytes_unsigned[7]

    # Convert back to signed 16-bit
    if x_int >= 32768:
        x_int -= 65536
    if y_int >= 32768:
        y_int -= 65536
    if z_int >= 32768:
        z_int -= 65536
    if rot_int >= 32768:
        rot_int -= 65536

    # Scale back to meters and degrees
    x = x_int / SCALE_POS
    y = y_int / SCALE_POS
    z = z_int / SCALE_POS
    rot_y = rot_int / SCALE_ROT

    return ([x, y, z], rot_y)


def encode_joint_positions_to_bytes(joint_positions):
    """
    Encode 4 joint positions to 24 bytes (2 bytes per coordinate, 3 coords per joint).

    Args:
        joint_positions: List of 4 positions [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
                        Order: [boom_mount, arm_mount, bucket_mount, tool_mount]

    Returns:
        List of 24 signed bytes
    """
    SCALE = 13107.0

    result_bytes = []

    for position in joint_positions:
        x, y, z = position

        # Convert to 16-bit integers
        x_int = int(round(x * SCALE))
        y_int = int(round(y * SCALE))
        z_int = int(round(z * SCALE))

        # Clamp to 16-bit signed range
        x_int = max(-32768, min(32767, x_int))
        y_int = max(-32768, min(32767, y_int))
        z_int = max(-32768, min(32767, z_int))

        # Split into high and low bytes
        x_high = (x_int >> 8) & 0xFF
        x_low = x_int & 0xFF
        y_high = (y_int >> 8) & 0xFF
        y_low = y_int & 0xFF
        z_high = (z_int >> 8) & 0xFF
        z_low = z_int & 0xFF

        # Convert to signed bytes
        def to_signed_byte(b):
            return b if b < 128 else b - 256

        result_bytes.extend([
            to_signed_byte(x_high), to_signed_byte(x_low),
            to_signed_byte(y_high), to_signed_byte(y_low),
            to_signed_byte(z_high), to_signed_byte(z_low)
        ])

    return result_bytes


def main():
    """Receive position commands and control excavator"""

    print("="*60)
    print("  GUI POSITION RECEIVER WITH LIVE VISUALIZATION")
    print("  Receiving position commands from excavator_gui.py")
    print("  Sending back joint positions for live arm display")
    print("="*60)

    # Setup UDP server to receive 8 bytes (position + rotation) and send 24 bytes (joint positions)
    server = UDPSocket(local_id=2, max_age_seconds=0.5)
    server.setup("192.168.0.132", 8080, num_inputs=8, num_outputs=24, is_server=True)

    print("\nWaiting for GUI connection...")
    if not server.handshake(timeout=30.0):
        print("Handshake failed!")
        return

    print("✓ Connected to GUI!")

    # Create hardware interface
    print("\nInitializing hardware...")
    from modules.hardware_interface import HardwareInterface
    hardware = HardwareInterface()

    print("Waiting for hardware to be ready...")
    while not hardware.is_hardware_ready():
        time.sleep(0.1)
    print("✓ Hardware ready!")

    # Create controller
    config = ControllerConfig(control_frequency=100.0)
    print("\nInitializing controller (includes numba warmup)...")
    controller = ExcavatorController(hardware, config, enable_perf_tracking=False)

    # Start the background control loop
    controller.start()
    print("Waiting for controller to initialize...")
    time.sleep(5.0)

    # Create robot config for FK computation
    robot_config = create_excavator_config()

    print("\n" + "="*60)
    print("  READY TO RECEIVE COMMANDS")
    print("  Start streaming from the GUI!")
    print("="*60 + "\n")

    # Start receiving UDP data
    server.start_receiving()

    last_position = None
    last_rotation = 0.0
    last_print_time = time.time()
    packets_received = 0
    last_packet_count = 0

    try:
        while True:
            # Get latest UDP packet
            data = server.get_latest()

            if data:
                # Decode 8 bytes to position and rotation
                try:
                    position, rotation_y = decode_bytes_to_position(data)
                    x, y, z = position

                    # Send to controller with rotation from GUI
                    controller.give_pose(position, rotation_y)

                    last_position = position
                    last_rotation = rotation_y
                    packets_received += 1

                except Exception as e:
                    print(f"Decode error: {e}")
                    continue

            # Read current joint positions via FK (independent of receiving commands)
            try:
                # Read live IMU data
                raw_imu_quats = hardware.read_imu_data()
                slew_quat = hardware.read_slew_quaternion()

                if raw_imu_quats is not None and len(raw_imu_quats) >= 3:
                    # Combine slew + 3 IMUs to get all 4 joints
                    all_quaternions = [slew_quat] + raw_imu_quats

                    # Apply IMU mounting offset corrections
                    corrected_quats = apply_imu_offsets(np.array(all_quaternions), robot_config)

                    # Project to rotation axes
                    projected_quats = project_to_rotation_axes(corrected_quats, robot_config)

                    # Compute forward kinematics to get joint positions
                    joint_positions = get_joint_positions(projected_quats, robot_config)

                    # Encode and send back to GUI (24 bytes for 4 joint positions)
                    encoded_joints = encode_joint_positions_to_bytes(joint_positions)
                    server.send(encoded_joints)

            except Exception as e:
                # Don't print FK errors continuously, just skip this cycle
                pass

            # Print status every 1 second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                # Get connection stats
                stats = server.get_connection_stats()

                if last_position:
                    # Calculate packet rate
                    new_packets = packets_received - last_packet_count
                    packet_rate = new_packets / (current_time - last_print_time)

                    # Get current pose from controller
                    actual_pos, actual_rot = controller.get_pose()

                    print(f"[{packet_rate:.1f} Hz] "
                          f"Target: [{last_position[0]:+.3f}, {last_position[1]:+.3f}, {last_position[2]:+.3f}]m "
                          f"Rot: {last_rotation:+.1f}° | "
                          f"Actual: [{actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f}]m "
                          f"Rot: {actual_rot:+.1f}° | "
                          f"Packets: {packets_received}")

                    last_packet_count = packets_received
                else:
                    if stats['is_connected']:
                        print(f"Waiting for position data... (connected)")
                    else:
                        print(f"No data received (connection lost)")

                last_print_time = current_time

            time.sleep(0.01)  # 100Hz check rate

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

        print(f"\n{'='*60}")
        print("FINAL STATUS:")
        print(f"  Position: [{final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}]m")
        print(f"  Rotation Y: {final_rot_y:+.2f}°")
        print(f"  Total packets received: {packets_received}")
        print(f"{'='*60}")
        print("Done.")


if __name__ == "__main__":
    main()
