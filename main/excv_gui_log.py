#!/usr/bin/env python3
"""
Well not gui but it reads values sent from client GUI haha

  # Normal operation with INFO level (default)
  python excv_gui_log.py
  # Debug mode - see all internal details
  python excv_gui_log.py --log-level DEBUG
  # Quiet mode - only warnings and errors
  python excv_gui_log.py --log-level WARNING
  # Performance monitoring mode (compact output)
  python excv_gui_log.py --perf
  # Jacobian debugging with detailed IK logging
  python excv_gui_log.py --jac
  # Jacobian debugging with full DEBUG output
  python excv_gui_log.py --log-level DEBUG --jac

"""

import time
import logging
import argparse
import numpy as np
from modules.udp_socket import UDPSocket
from modules.excavator_controller import ExcavatorController, ControllerConfig
from modules.diff_ik_V2 import (
    create_excavator_config,
    get_joint_positions,
    apply_imu_offsets,
    extract_axis_rotation,
    project_to_rotation_axes,
    compute_jacobian,
    get_pose,
)
# Load IRL pathing config exclusively from configuration_files
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_cfg_dir = os.path.abspath(os.path.join(_here, 'configuration_files'))
_cfg_file = os.path.join(_cfg_dir, 'pathing_config.py')
if _cfg_dir not in sys.path:
    sys.path.append(_cfg_dir)
if not os.path.isfile(_cfg_file):
    raise ImportError("configuration_files/pathing_config.py not found (copy required for IRL)")
from pathing_config import DEFAULT_CONFIG

# Enable Jacobian-level debug logging via env var
DEBUG_JAC = str(os.getenv('EXCV_DEBUG_JAC', '0')).lower() not in ('0', '', 'false', 'no')


def decode_bytes_to_position(bytes_list):
    """
    Decode 9 bytes back to position, rotation values, and reload flag [x, y, z, rot_y, reload_flag].

    This matches the encoding in excavator_gui.py.

    Args:
        bytes_list: List of 9 signed bytes [-128, 127]

    Returns:
        Tuple: ([x, y, z], rot_y, reload_flag) - position in meters, rotation in degrees, reload flag
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

    # Extract reload flag (9th byte) - normalized to [-1, 1]
    reload_flag = bytes_list[8] / 127.0 if len(bytes_list) >= 9 else 0.0

    return ([x, y, z], rot_y, reload_flag)


def encode_joint_positions_to_bytes(joint_positions):
    """
    Encode joint positions to bytes (2 bytes per coordinate, 3 coords per joint).

    Args:
        joint_positions: List of positions [[x,y,z], ...]
                        For excavator with EE: [boom_mount, arm_mount, bucket_mount, tool_mount, ee_actual]
                        Returns 30 bytes for 5 positions

    Returns:
        List of signed bytes (6 bytes per position)
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
    # CLI args for log level and performance debug
    parser = argparse.ArgumentParser(description="GUI receiver with configurable logging")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for controller and modules (default: INFO)"
    )
    parser.add_argument("--jac", action="store_true", help="Enable Jacobian-level debug logging")
    parser.add_argument("--perf", action="store_true", help="Enable clean performance metrics display (loop time, headroom, CPU)")
    args, _ = parser.parse_known_args()

    # Setup main application logger
    app_logger = logging.getLogger("excv_gui_log")
    app_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)

    global DEBUG_JAC
    DEBUG_JAC = args.jac

    # Performance debug mode uses compact output
    DEBUG_PERF = args.perf

    if not DEBUG_PERF:
        app_logger.info("="*60)
        app_logger.info("  GUI POSITION RECEIVER WITH LIVE VISUALIZATION")
        app_logger.info("  Receiving position commands from excavator_gui.py")
        app_logger.info("  Sending back joint positions for live arm display")
        app_logger.info("="*60)

    # Setup UDP server to receive 9 bytes (position + rotation + reload_flag) and send 30 bytes (joint positions + EE)
    server = UDPSocket(local_id=2, max_age_seconds=0.5)
    server.setup("192.168.0.132", 8080, num_inputs=9, num_outputs=30, is_server=True)

    if not DEBUG_PERF:
        app_logger.info("\nWaiting for GUI connection...")
    if not server.handshake(timeout=30.0):
        if not DEBUG_PERF:
            app_logger.error("Handshake failed!")
        return

    if not DEBUG_PERF:
        app_logger.info("✓ Connected to GUI!")

    # Create hardware interface
    if not DEBUG_PERF:
        app_logger.info("\nInitializing hardware...")
    from modules.hardware_interface import HardwareInterface

    # Set log level based on mode: WARNING in perf mode, user-specified otherwise
    hw_log_level = "WARNING" if DEBUG_PERF else args.log_level.upper()
    hardware = HardwareInterface(log_level=hw_log_level)

    if not DEBUG_PERF:
        app_logger.info("Waiting for hardware to be ready...")
    while not hardware.is_hardware_ready():
        time.sleep(0.1)
    if not DEBUG_PERF:
        app_logger.info("✓ Hardware ready!")

    # Create controller
    if not DEBUG_PERF:
        app_logger.info("\nInitializing controller (includes numba warmup)...")

    # Enable perf tracking when in DEBUG_PERF mode
    # Set log level: WARNING in perf mode, DEBUG if --jac, otherwise user-specified
    ctrl_log_level = "WARNING" if DEBUG_PERF else ("DEBUG" if DEBUG_JAC else args.log_level.upper())
    controller = ExcavatorController(
        hardware,
        config=None,
        enable_perf_tracking=DEBUG_PERF,
        log_level=ctrl_log_level,
    )
    # TODO: this
    #controller.ik_controller.cfg.enable_frame_transform = False

    # Start the background control loop
    controller.start()
    if not DEBUG_PERF:
        app_logger.info("Waiting for controller to initialize...")
    time.sleep(5.0)

    # Create robot config for FK computation
    robot_config = create_excavator_config()

    if not DEBUG_PERF:
        app_logger.info("\n" + "="*60)
        app_logger.info("  READY TO RECEIVE COMMANDS")
        app_logger.info("  Start streaming from the GUI!")
        # Brief IK config context
        try:
            jw = DEFAULT_CONFIG.ik_params.get('joint_weights') if hasattr(DEFAULT_CONFIG, 'ik_params') else None
            app_logger.info(f"  IK: method={getattr(DEFAULT_CONFIG,'ik_method',None)} ignore_axes={getattr(DEFAULT_CONFIG,'ignore_axes',None)} joint_weights={jw}")
        except Exception:
            pass
        app_logger.info("="*60 + "\n")

    # Start receiving UDP data
    server.start_receiving()

    last_position = None
    last_rotation = 0.0
    last_time = time.perf_counter()

    last_print_time = time.time()
    packets_received = 0
    last_packet_count = 0

    # Performance debug mode: print every 2s instead of 1s
    perf_print_interval = 2.0 if DEBUG_PERF else 1.0

    # Network timing tracking for debug mode
    network_times = []
    max_network_samples = 1000

    # Edge detection for reload config flag
    reload_flag_prev = 0.0
    flag_threshold = 0.5

    # Jacobian debug state (populated when DEBUG_JAC)
    prev_joint_angles = None  # radians per joint
    prev_ee_pos = None        # meters xyz
    dbg_singular_values = None
    dbg_cond = None
    dbg_dq = None
    dbg_ee_pred = None
    dbg_ee_actual = None
    dbg_Jpos = None
    dbg_Jang = None
    dbg_joint_deg = None
    dbg_dq_deg = None

    # Target loop timing from pathing_config (DEFAULT_CONFIG)
    target_hz = getattr(DEFAULT_CONFIG, "update_frequency", 100.0)
    try:
        target_dt = 1.0 / max(1e-3, float(target_hz))
    except Exception:
        target_dt = 0.01  # Fallback to 100 Hz if config is malformed
    next_tick = time.perf_counter()

    try:
        while True:
            # Get latest UDP packet
            if DEBUG_PERF:
                net_start = time.perf_counter()

            data = server.get_latest()

            if data:
                # Decode 9 bytes to position, rotation, and reload flag
                try:
                    position, rotation_y, reload_flag = decode_bytes_to_position(data)
                    x, y, z = position

                    # Handle reload config request (rising edge detection)
                    if reload_flag > flag_threshold and reload_flag_prev <= flag_threshold:
                        try:
                            if not DEBUG_PERF:
                                app_logger.info("Reload config requested by client GUI...")
                            servo_ok = hardware.reload_config()
                            general_ok = hardware.reload_general_config()
                            if not DEBUG_PERF:
                                if servo_ok or general_ok:
                                    app_logger.info(f"✓ Config reloaded (servo={servo_ok}, general={general_ok})")
                                else:
                                    app_logger.info("Config reload: no changes detected")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Config reload failed: {e}")

                    reload_flag_prev = reload_flag

                    # Store target; actual command is rate-limited below
                    last_position = position
                    last_rotation = rotation_y
                    packets_received += 1

                except Exception as e:
                    if not DEBUG_PERF:
                        app_logger.error(f"Decode error: {e}")
                    continue

            # Send latest target directly; controller handles smoothing internally
            try:
                now_t = time.perf_counter()
                dt = max(0.0, min(now_t - last_time, 0.2))
                last_time = now_t

                if last_position is not None:
                    controller.give_pose(
                        np.array(last_position, dtype=np.float32),
                        float(last_rotation),
                    )
            except Exception:
                pass

            # Read current joint positions via FK (independent of receiving commands)
            try:
                # Read live IMU data
                raw_imu_quats = hardware.read_imu_data()
                slew_quat = hardware.read_slew_quaternion()

                if raw_imu_quats is not None and len(raw_imu_quats) >= 3:
                    # Combine slew + 3 IMUs to get all 4 joints
                    all_quaternions = [slew_quat] + raw_imu_quats

                    # New diff_ik API expects FULL quats; it applies IMU offsets and base propagation.
                    full_quats = np.array(all_quaternions, dtype=np.float32)

                    # Compute forward kinematics to get joint positions
                    joint_positions = get_joint_positions(full_quats, robot_config)

                    # Get actual end-effector position (with ee_offset applied)
                    ee_position, _ = get_pose(full_quats, robot_config)

                    # Jacobian-level diagnostics (optional)
                    if DEBUG_JAC:
                        try:
                            J = compute_jacobian(full_quats, robot_config)
                            J_pos = J[0:3, :]
                            J_ang = J[3:6, :]
                            s = np.linalg.svd(J_pos, compute_uv=False)
                            dbg_singular_values = s.astype(np.float32)
                            s_nonzero = s[s > 1e-9]
                            dbg_cond = float(np.max(s_nonzero) / np.min(s_nonzero)) if len(s_nonzero) > 0 else float('inf')
                            dbg_Jpos = J_pos.astype(np.float32)
                            dbg_Jang = J_ang.astype(np.float32)

                            # Joint angles via axis-twist projection
                            axes = np.array(robot_config.rotation_axes, dtype=np.float32)
                            projected = project_to_rotation_axes(full_quats, axes)
                            joint_angles = np.array([
                                extract_axis_rotation(q, axis) for q, axis in zip(projected, robot_config.rotation_axes)
                            ], dtype=np.float32)

                            # EE position for measured delta
                            ee_pos, _ = get_pose(full_quats, robot_config)

                            dbg_joint_deg = np.degrees(joint_angles).astype(np.float32)
                            if prev_joint_angles is not None:
                                dbg_dq = (joint_angles - prev_joint_angles).astype(np.float32)
                                dbg_dq_deg = np.degrees(dbg_dq).astype(np.float32)
                                dbg_ee_pred = (J_pos @ dbg_dq).astype(np.float32)
                            else:
                                dbg_dq = None
                                dbg_dq_deg = None
                                dbg_ee_pred = None

                            if prev_ee_pos is not None:
                                dbg_ee_actual = (ee_pos - prev_ee_pos).astype(np.float32)
                            else:
                                dbg_ee_actual = None

                            prev_joint_angles = joint_angles
                            prev_ee_pos = ee_pos.astype(np.float32)
                        except Exception:
                            # Keep running even if diagnostics fail
                            pass

                    # Encode and send back to GUI (30 bytes: 4 joint positions + 1 EE position)
                    all_positions = list(joint_positions) + [ee_position]
                    encoded_joints = encode_joint_positions_to_bytes(all_positions)
                    server.send(encoded_joints)

                    # Track network timing
                    if DEBUG_PERF:
                        net_end = time.perf_counter()
                        network_times.append((net_end - net_start) * 1000.0)  # Convert to ms
                        if len(network_times) > max_network_samples:
                            network_times.pop(0)

            except Exception as e:
                # Don't print FK errors continuously, just skip this cycle
                pass

            # Print status every 1-2 seconds (depending on mode)
            current_time = time.time()
            if current_time - last_print_time >= perf_print_interval:
                # Get connection stats
                stats = server.get_connection_stats()

                if last_position:
                    # Calculate packet rate
                    new_packets = packets_received - last_packet_count
                    packet_rate = new_packets / (current_time - last_print_time)

                    # Get current pose from controller
                    actual_pos, actual_rot = controller.get_pose()

                    # Performance debug mode: clean output with stage breakdown
                    if DEBUG_PERF:
                        perf_stats = controller.get_performance_stats()

                        # Main line: overall loop performance
                        line = (f"Loop: {perf_stats['avg_loop_time_ms']:.2f}ms "
                                f"(min={perf_stats['min_loop_time_ms']:.2f} max={perf_stats['max_loop_time_ms']:.2f}) | "
                                f"CPU: {perf_stats['cpu_usage_pct']:.1f}% | "
                                f"Headroom: {perf_stats['avg_headroom_ms']:.2f}ms | "
                                f"Rate: {perf_stats['actual_hz']:.1f}Hz")

                        # Stage breakdown: Sensors, IK/FK, PWM, Network
                        stages = []

                        # Sensors (IMU + Encoder)
                        sensor_avg = perf_stats.get('avg_sensor_ms', 0)
                        sensor_min = perf_stats.get('min_sensor_ms', 0)
                        sensor_max = perf_stats.get('max_sensor_ms', 0)
                        if sensor_avg > 0:
                            stages.append(f"Sensors: {sensor_avg:.2f}ms ({sensor_min:.2f}-{sensor_max:.2f})")

                        # IK/FK
                        ik_avg = perf_stats.get('avg_ik_fk_ms', 0)
                        ik_min = perf_stats.get('min_ik_fk_ms', 0)
                        ik_max = perf_stats.get('max_ik_fk_ms', 0)
                        if ik_avg > 0:
                            stages.append(f"IK/FK: {ik_avg:.2f}ms ({ik_min:.2f}-{ik_max:.2f})")

                        # PWM Control
                        pwm_avg = perf_stats.get('avg_pwm_ms', 0)
                        pwm_min = perf_stats.get('min_pwm_ms', 0)
                        pwm_max = perf_stats.get('max_pwm_ms', 0)
                        if pwm_avg > 0:
                            stages.append(f"PWM: {pwm_avg:.2f}ms ({pwm_min:.2f}-{pwm_max:.2f})")

                        # Network
                        if network_times:
                            net_avg = np.mean(network_times)
                            net_min = np.min(network_times)
                            net_max = np.max(network_times)
                            stages.append(f"Net: {net_avg:.2f}ms ({net_min:.2f}-{net_max:.2f})")

                        if stages:
                            line += " | " + " | ".join(stages)

                        # Add packet rate at the end
                        line += f" | Pkt: {packet_rate:.1f}Hz"

                        # Optionally append joint velocity and limiter info
                        vel_line = ""
                        try:
                            last_vel = perf_stats.get('last_joint_vel_degps', [])
                            vel_cap = perf_stats.get('effective_vel_cap_degps', [])
                            vel_lim_on = perf_stats.get('ik_vel_lim_enabled', False)
                            if last_vel:
                                # Expect order: [slew, boom, arm, bucket]
                                names = ['slew', 'boom', 'arm', 'bucket']
                                parts = []
                                for n, v in zip(names, last_vel):
                                    parts.append(f"{n}={v:+.1f}")
                                vel_line = " | Vel(deg/s): " + ", ".join(parts)
                                if vel_lim_on and vel_cap:
                                    cap_parts = []
                                    for n, c in zip(names, vel_cap):
                                        cap_parts.append(f"{n}={c:.1f}")
                                    vel_line += " | Cap(deg/s): " + ", ".join(cap_parts)
                        except Exception:
                            pass

                        print(line + vel_line)
                    else:
                        print(f"[{packet_rate:.1f} Hz] "
                              f"Target: [{last_position[0]:+.3f}, {last_position[1]:+.3f}, {last_position[2]:+.3f}]m "
                              f"Rot: {last_rotation:+.1f}° | "
                              f"Actual: [{actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f}]m "
                              f"Rot: {actual_rot:+.1f}° | "
                              f"Packets: {packets_received}")

                    # Optional: print compact Jacobian summary once per second (not in DEBUG_PERF)
                    if DEBUG_JAC and not DEBUG_PERF and dbg_singular_values is not None and dbg_Jpos is not None:
                        try:
                            def _fmt_vec(v, n=3):
                                return ", ".join(f"{float(x):+.3f}" for x in list(v)[:n])
                            sv = dbg_singular_values
                            print(f"  [JAC] s=[{_fmt_vec(sv, n=min(3, len(sv)))}] cond={dbg_cond:.2f}")
                            if dbg_dq is not None:
                                print(f"  [JAC] dq_deg=[{_fmt_vec(dbg_dq_deg, n=4)}]; ee_pred=[{_fmt_vec(dbg_ee_pred, n=3)}] m")
                            if dbg_ee_actual is not None:
                                print(f"  [JAC] ee_act=[{_fmt_vec(dbg_ee_actual, n=3)}] m")
                            jp = dbg_Jpos
                            print(f"  [Jpos] r0=[{_fmt_vec(jp[0], n=4)}] | r1=[{_fmt_vec(jp[1], n=4)}] | r2=[{_fmt_vec(jp[2], n=4)}]")
                            if dbg_Jang is not None:
                                ja = dbg_Jang
                                print(f"  [Jang] r3=[{_fmt_vec(ja[0], n=4)}] | r4=[{_fmt_vec(ja[1], n=4)}] | r5=[{_fmt_vec(ja[2], n=4)}]")

                            if dbg_joint_deg is not None:
                                slew_deg = float(dbg_joint_deg[0]) if len(dbg_joint_deg) > 0 else 0.0
                                dyaw = float(dbg_dq_deg[0]) if (dbg_dq_deg is not None and len(dbg_dq_deg) > 0) else 0.0
                                dt_print = max(1e-6, current_time - last_print_time)
                                yaw_rate = dyaw / dt_print
                                print(
                                    f"  [JNT] deg=[{_fmt_vec(dbg_joint_deg, n=4)}]; "
                                    f"[SLEW] yaw={slew_deg:+.1f} deg dyaw={dyaw:+.2f} deg dt={dt_print:.3f}s rate={yaw_rate:+.2f} deg/s"
                                )

                            if last_position:
                                pos_err = np.array(last_position, dtype=np.float32) - np.asarray(actual_pos, dtype=np.float32)
                                print(f"  [ERR] pos=[{_fmt_vec(pos_err, n=3)}] m")
                        except Exception:
                            pass
                    last_packet_count = packets_received
                else:
                    if not DEBUG_PERF:
                        if stats['is_connected']:
                            app_logger.info(f"Waiting for position data... (connected)")
                        else:
                            app_logger.warning(f"No data received (connection lost)")

                last_print_time = current_time

            # Deadline-based timing using configured target frequency
            next_tick += target_dt
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we overrun, reset the schedule to avoid drift
                next_tick = now

    except KeyboardInterrupt:
        if not DEBUG_PERF:
            app_logger.info("\n\nInterrupted by user")

    except Exception as e:
        if not DEBUG_PERF:
            app_logger.error(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    finally:
        # Clean shutdown
        if not DEBUG_PERF:
            app_logger.info("\nStopping controller...")
        controller.stop()
        hardware.reset(reset_pump=True)

        # Show final status
        final_pos, final_rot_y = controller.get_pose()

        if not DEBUG_PERF:
            app_logger.info(f"\n{'='*60}")
            app_logger.info("FINAL STATUS:")
            app_logger.info(f"  Position: [{final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}]m")
            app_logger.info(f"  Rotation Y: {final_rot_y:+.2f}°")
            app_logger.info(f"  Total packets received: {packets_received}")
            app_logger.info(f"{'='*60}")
            app_logger.info("Done.")


if __name__ == "__main__":
    main()
