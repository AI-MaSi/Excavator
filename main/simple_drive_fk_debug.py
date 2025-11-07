from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface
from modules.diff_ik import (
    create_excavator_config,
    get_all_poses,
    extract_axis_rotation,
)
import time
import numpy as np
import argparse

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description='Simple drive controller with FK debugging')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode: prints pulse widths (us) for all commanded channels')
args = parser.parse_args()

DEBUG_PULSE_WIDTHS = args.debug

# ============================================================
# INITIALIZE HARDWARE & NETWORK
# ============================================================
server = UDPSocket(local_id=2)
server.setup("192.168.0.132", 8080, num_inputs=9, num_outputs=0, is_server=True)

hardware = HardwareInterface(
    config_file="configuration_files/servo_config.yaml",
    pump_variable=False,
    toggle_channels=True,
    input_rate_threshold=5
)

print("Waiting for hardware to be ready...")
while not hardware.is_hardware_ready():
    time.sleep(0.1)
print("Hardware ready!")

# Create robot kinematic config for FK computation
robot_config = create_excavator_config()

try:
    # Enable low-overhead perf metrics
    hardware.set_perf_enabled(True)
    hardware.reset_perf_stats()
except Exception:
    pass

if DEBUG_PULSE_WIDTHS:
    print("\n" + "="*60)
    print("  DEBUG MODE ENABLED")
    print("  Pulse widths (μs) will be printed for all active channels")
    print("="*60 + "\n")

print("\n" + "="*60)
print("  FK DEBUG MODE")
print("  Printing Forward Kinematics results every 2 seconds")
print("  This will help verify IMU0 (slew) is being read correctly")
print("="*60 + "\n")

# ============================================================
# MAIN CONTROL LOOP
# ============================================================
if server.handshake():
    server.start_receiving()
    print("UDP connection established, starting control loop...\n")

    # Button state tracking
    button_1_prev = 0.0
    button_2_prev = 0.0
    button_threshold = 0.5

    last_fk_print_time = time.time()
    FK_PRINT_INTERVAL = 0.1  # Print FK results every 0.1 seconds (10 Hz)

    # Loop timing
    CONTROL_FREQUENCY = 100  # Hz
    loop_period = 1.0 / CONTROL_FREQUENCY
    next_run_time = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        data = server.get_latest()

        if data:
            # Convert to float [-1.0 to 1.0]
            float_data = [round((value / 127.0), 2) for value in data]

            # Map controller inputs
            right_rl = float_data[8]        # scoop
            right_ud = float_data[7]        # lift
            left_rl = float_data[6]         # rotate
            left_ud = float_data[5]         # tilt
            right_paddle = float_data[4]    # right track
            left_paddle = float_data[3]     # left track
            button_1 = float_data[1]
            button_2 = float_data[2]

            # Button 1: Reload config
            if button_1 > button_threshold and button_1_prev <= button_threshold:
                print("\n[Button 1] Reloading configuration...")
                hardware.reload_config()

            # Button 2: Toggle pump
            if button_2 > button_threshold and button_2_prev <= button_threshold:
                print("\n[Button 2] Toggling pump...")
                if hardware.pwm_controller:
                    hardware.pwm_controller.set_pump(not hardware.pwm_controller.pump_enabled)

            button_1_prev = button_1
            button_2_prev = button_2

            # Build name-based command mapping
            named = {
                'scoop': right_rl,
                'lift_boom': right_ud,
                'rotate': left_rl,
                'tilt_boom': left_ud,
                'trackR': right_paddle,
                'trackL': left_paddle,
            }

            # Send commands to hardware
            hardware.send_named_pwm_commands(named, unset_to_zero=True)

            # Debug: Print pulse widths for all commanded channels
            if DEBUG_PULSE_WIDTHS and hardware.pwm_controller:
                dbg = "[PULSE DEBUG] "
                parts = []
                for ch in ['rotate', 'lift_boom', 'tilt_boom', 'scoop']:
                    val = named.get(ch, 0.0)
                    pu = hardware.pwm_controller.compute_pulse(ch, val)
                    pu_txt = f"{pu:.1f}us" if pu is not None else "n/a"
                    parts.append(f"{ch}={val:+.2f} -> {pu_txt}")
                print(dbg + " | ".join(parts))

            # Periodic FK computation and printing
            current_time = time.time()
            if (current_time - last_fk_print_time >= FK_PRINT_INTERVAL):
                last_fk_print_time = current_time

                # Read raw IMU quaternions - NO FALLBACKS, will raise on error (with safety reset)
                raw_imu_quats = hardware.read_imu_data()  # Returns 3 IMUs [boom, arm, bucket] or raises
                slew_quat = hardware.read_slew_quaternion()  # Returns slew (IMU0) or raises

                # Combine slew + 3 joint IMUs: [slew, boom, arm, bucket]
                full_quats = [slew_quat] + list(raw_imu_quats)
                full_quats_array = np.array(full_quats, dtype=np.float32)

                print("\n" + "="*70)
                print("  FORWARD KINEMATICS DEBUG OUTPUT")
                print("="*70)

                # Print raw quaternions
                print("\nRAW IMU QUATERNIONS [w, x, y, z]:")
                print(f"  IMU0 (Slew):   {slew_quat}")
                print(f"  IMU1 (Boom):   {raw_imu_quats[0]}")
                print(f"  IMU2 (Arm):    {raw_imu_quats[1]}")
                print(f"  IMU3 (Bucket): {raw_imu_quats[2]}")

                # Compute FK using diff_ik pipeline
                joint_positions, joint_orientations, ee_position, ee_orientation = get_all_poses(
                    full_quats_array, robot_config
                )

                print("\nJOINT POSITIONS (meters) [x, y, z]:")
                joint_names_pos = ["Slew", "Boom", "Arm", "Bucket"]
                for i, name in enumerate(joint_names_pos):
                    if i < len(joint_positions):
                        pos = joint_positions[i]
                        print(f"  {name:12s}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")

                print("\nJOINT ORIENTATIONS (quaternions) [w, x, y, z]:")
                joint_names = ["Slew", "Boom", "Arm", "Bucket"]
                for i, name in enumerate(joint_names):
                    if i < len(joint_orientations):
                        quat = joint_orientations[i]
                        print(f"  {name:12s}: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")

                print("\nJOINT ANGLES (radians and degrees):")
                # Joint 0: Slew rotation (Z-axis)
                slew_angle = float(extract_axis_rotation(joint_orientations[0], robot_config.rotation_axes[0]))
                print(f"  Slew (Z):      {slew_angle:7.4f} rad  ({np.degrees(slew_angle):7.2f}°)")

                # Joint 1: Boom pitch (Y-axis)
                boom_angle = float(extract_axis_rotation(joint_orientations[1], robot_config.rotation_axes[1]))
                print(f"  Boom (Y):      {boom_angle:7.4f} rad  ({np.degrees(boom_angle):7.2f}°)")

                # Joint 2: Arm pitch (Y-axis)
                arm_angle = float(extract_axis_rotation(joint_orientations[2], robot_config.rotation_axes[2]))
                print(f"  Arm (Y):       {arm_angle:7.4f} rad  ({np.degrees(arm_angle):7.2f}°)")

                # Joint 3: Bucket pitch (Y-axis)
                bucket_angle = float(extract_axis_rotation(joint_orientations[3], robot_config.rotation_axes[3]))
                print(f"  Bucket (Y):    {bucket_angle:7.4f} rad  ({np.degrees(bucket_angle):7.2f}°)")

                print("\nEND-EFFECTOR (Bucket Tip):")
                print(f"  Position:    [{ee_position[0]:7.4f}, {ee_position[1]:7.4f}, {ee_position[2]:7.4f}] m")
                print(f"  Orientation: [{ee_orientation[0]:7.4f}, {ee_orientation[1]:7.4f}, {ee_orientation[2]:7.4f}, {ee_orientation[3]:7.4f}] [w,x,y,z]")

                # Extract Y-axis rotation (pitch) from end-effector orientation
                ee_y_rotation = float(extract_axis_rotation(ee_orientation, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
                print(f"  Y-Rotation:  {ee_y_rotation:7.4f} rad  ({np.degrees(ee_y_rotation):7.2f}°)")

                print("="*70 + "\n")

        else:
            # No data received (safety handled internally)
            pass

        # Adaptive sleep to maintain frequency
        next_run_time += loop_period
        sleep_time = next_run_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # If we're behind, reset timing to prevent drift
            next_run_time = time.perf_counter()

else:
    print("UDP handshake failed!")
    hardware.reset()

# Cleanup on exit
print("\nShutting down...")
hardware.reset(reset_pump=True)
