#!/usr/bin/env python3
"""
Well not gui but it reads values sent from client GUI haha

  # Normal operation with INFO level (default)
  python excv_gui_log.py
  # Debug mode - see all internal details
  python excv_gui_log.py --log-level DEBUG
  # Quiet mode - only warnings and errors
  python excv_gui_log.py --log-level WARNING
  # Performance monitoring mode (perf-only, compact output)
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
import sys
import os
from pathlib import Path
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from modules.udp_socket import UDPSocket
from modules.excavator_controller import ExcavatorController, ControllerConfig
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO
from modules.diff_ik_V2 import (
    create_excavator_config,
    get_joint_positions,
    apply_imu_offsets,
    extract_axis_rotation,
    project_to_rotation_axes,
    compute_jacobian,
    get_pose,
)
import yaml
_here = os.path.dirname(os.path.abspath(__file__))
_cfg_dir = os.path.abspath(os.path.join(_here, 'configuration_files'))


def _load_control_config():
    """Load control configuration YAML file for IK config."""
    config_path = Path(_cfg_dir) / "control_config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_CONTROL_CONFIG = _load_control_config()

# Enable Jacobian-level debug logging via env var
DEBUG_JAC = str(os.getenv('EXCV_DEBUG_JAC', '0')).lower() not in ('0', '', 'false', 'no')


def decode_bytes_to_position(bytes_list):
    """
    Decode 9 bytes back to position, rotation values, and control flags.

    Layout:
        [x_high, x_low, y_high, y_low, z_high, z_low, rot_high, rot_low, control_flags]

    control_flags bitmask:
        bit0: reload config
        bit1: pause controller
        bit2: resume controller

    Args:
        bytes_list: List of 9 signed bytes [-128, 127]

    Returns:
        Tuple: ([x, y, z], rot_y, control_flags) - position in meters, rotation in degrees, control flags (int)
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

    # Extract control flags (9th byte)
    control_flags = to_unsigned_byte(bytes_list[8]) if len(bytes_list) >= 9 else 0

    return ([x, y, z], rot_y, control_flags)


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


def decode_bytes_to_direct_commands(bytes_list):
    """
    Decode 9 bytes as direct valve commands (same packet layout, all 4 int16
    fields carry normalized [-1, 1] values scaled by SCALE_POS).

    Returns:
        List of 4 floats [slew, boom, arm, bucket] in range [-1, 1].
    """
    SCALE_POS = 13107.0

    def to_unsigned_byte(b):
        return b if b >= 0 else b + 256

    bytes_unsigned = [to_unsigned_byte(b) for b in bytes_list]

    raw_ints = []
    for i in range(0, 8, 2):
        val = (bytes_unsigned[i] << 8) | bytes_unsigned[i + 1]
        if val >= 32768:
            val -= 65536
        raw_ints.append(val)

    return [v / SCALE_POS for v in raw_ints]


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
    parser.add_argument("--perf", action="store_true", help="Enable perf-only display (loop time, jitter, headroom, CPU)")
    parser.add_argument("--rt-priority", type=int, default=75, help="RT priority for main thread (default: 75, 0=disable)")
    parser.add_argument("--imu-priority", type=int, default=70, help="RT priority for IMU thread (default: 70)")
    parser.add_argument("--adc-priority", type=int, default=70, help="RT priority for ADC thread (default: 70)")
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
    PERF_ONLY = bool(DEBUG_PERF)

    if not DEBUG_PERF:
        app_logger.info("="*60)
        app_logger.info("  GUI POSITION RECEIVER WITH LIVE VISUALIZATION")
        app_logger.info("  Receiving position commands from excavator_gui.py")
        app_logger.info("  Sending back joint positions for live arm display")
        app_logger.info("="*60)
        app_logger.info(f"RT priorities: Control={args.rt_priority}, IMU={args.imu_priority}, ADC={args.adc_priority}")
        app_logger.info("ADC: Slew encoder @ 100Hz with 2x oversample")

    # Setup UDP server to receive 9 bytes and send 32 bytes (5 positions + EE rot_y)
    server = UDPSocket(local_id=2, max_age_seconds=0.5)
    server.setup("192.168.0.132", 8080, num_inputs=9, num_outputs=32, is_server=True)

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
    hardware = HardwareInterface(
        log_level=hw_log_level,
        pump_variable=True,
        cleanup_disable_osc=False,
        adc_channels=["Slew encoder rot"],
        adc_sample_hz=200.0,  # 200Hz for 200Hz control loop
        imu_rt_priority=args.imu_priority,
        adc_rt_priority=args.adc_priority,
    )

    if not DEBUG_PERF:
        app_logger.info("Waiting for hardware to be ready...")
    while not hardware.is_hardware_ready():
        time.sleep(0.1)
    if not DEBUG_PERF:
        app_logger.info("✓ Hardware ready!")

    # Create controller
    if not DEBUG_PERF:
        app_logger.info("\nInitializing controller (includes numba warmup)...")

    # Local buffers used in perf/telemetry modes
    network_times = []
    last_print_time = time.time()

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
    # Warmup then reset perf stats when perf modes are active
    warmup_time = 5.0 if DEBUG_PERF else 5.0
    if warmup_time > 0:
        time.sleep(warmup_time)

    # Apply RT priority to main thread after warmup
    if args.rt_priority > 0:
        if not DEBUG_PERF:
            app_logger.info(f"Applying RT priority SCHED_FIFO-{args.rt_priority} to main thread...")
        if not apply_rt_to_thread(priority=args.rt_priority, policy=SCHED_FIFO, quiet=True):
            if not DEBUG_PERF:
                app_logger.warning("Failed to apply RT priority (run as root for RT scheduling)")

    if DEBUG_PERF:
        # Drop warmup samples so perf stats start clean after numba/hardware settles
        try:
            controller.reset_performance_stats()
        except Exception:
            pass
        try:
            if hasattr(hardware, "reset_perf_stats"):
                hardware.reset_perf_stats()
        except Exception:
            pass
        # Reset local timing buffers
        last_print_time = time.time()
        network_times.clear()

    # Create robot config for FK computation
    robot_config = create_excavator_config()

    if not DEBUG_PERF:
        app_logger.info("\n" + "="*60)
        app_logger.info("  READY TO RECEIVE COMMANDS")
        app_logger.info("  Start streaming from the GUI!")
        # Brief IK config context (from control_config.yaml)
        try:
            ik_cfg = _CONTROL_CONFIG.get('ik', {})
            jw = ik_cfg.get('params', {}).get('joint_weights')
            app_logger.info(f"  IK: method={ik_cfg.get('method')} ignore_axes={ik_cfg.get('ignore_axes')} joint_weights={jw}")
        except Exception:
            pass
        app_logger.info("="*60 + "\n")

    # Start receiving UDP data
    server.start_receiving()

    last_position = None
    last_rotation = 0.0
    last_time = time.perf_counter()

    # Initialize print timer (may be reset after perf warmup)
    packets_received = 0
    last_packet_count = 0

    # Performance debug mode: print every 2s instead of 1s
    perf_print_interval = 2.0 if DEBUG_PERF else 1.0

    # Network timing tracking for debug mode
    max_network_samples = 1000

    # Control flag bits (must match client_gui_3d.py)
    FLAG_RELOAD = 1 << 0
    FLAG_PAUSE = 1 << 1
    FLAG_RESUME = 1 << 2
    FLAG_DIRECT = 1 << 3

    # Edge detection for control flags
    reload_prev = False
    pause_prev = False
    resume_prev = False
    is_paused = False

    # Direct control mode state
    is_direct_mode = False
    direct_prev = False
    last_direct_commands = [0, 0, 0, 0]

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

    target_hz = 100.0
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
                # Decode 9 bytes to position, rotation, and control flags
                try:
                    position, rotation_y, control_flags = decode_bytes_to_position(data)
                    x, y, z = position

                    # Mask to valid flag bits (4 bits: reload, pause, resume, direct)
                    control_flags &= 0x0F

                    reload_now = bool(control_flags & FLAG_RELOAD)
                    pause_now = bool(control_flags & FLAG_PAUSE)
                    resume_now = bool(control_flags & FLAG_RESUME)

                    # Handle reload config request (rising edge detection)
                    if reload_now and not reload_prev:
                        try:
                            if not DEBUG_PERF:
                                app_logger.info("Reload config requested by client GUI...")
                            servo_ok = hardware.reload_config()
                            if not DEBUG_PERF:
                                if servo_ok:
                                    app_logger.info("✓ PWM config reloaded")
                                else:
                                    app_logger.info("Config reload: no changes detected")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Config reload failed: {e}")

                    # Handle pause/resume requests (rising edge detection)
                    if pause_now and not pause_prev:
                        try:
                            controller.pause()
                            is_paused = True
                            if not DEBUG_PERF:
                                app_logger.info("Controller paused by GUI")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Pause failed: {e}")

                    if resume_now and not resume_prev:
                        try:
                            controller.resume()
                            is_paused = False
                            if not DEBUG_PERF:
                                app_logger.info("Controller resumed by GUI")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Resume failed: {e}")

                    # Handle direct mode flag (edge detection)
                    direct_now = bool(control_flags & FLAG_DIRECT)
                    if direct_now and not direct_prev:
                        try:
                            controller.enter_direct_mode()
                            is_direct_mode = True
                            if not DEBUG_PERF:
                                app_logger.info("Entered direct control mode")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Enter direct mode failed: {e}")
                    elif not direct_now and direct_prev:
                        try:
                            controller.exit_direct_mode()
                            is_direct_mode = False
                            if not DEBUG_PERF:
                                app_logger.info("Exited direct control mode")
                        except Exception as e:
                            if not DEBUG_PERF:
                                app_logger.error(f"Exit direct mode failed: {e}")

                    reload_prev = reload_now
                    pause_prev = pause_now
                    resume_prev = resume_now
                    direct_prev = direct_now

                    if is_direct_mode:
                        # Decode as direct valve commands
                        last_direct_commands = decode_bytes_to_direct_commands(data)
                    else:
                        # Store target; actual command is rate-limited below
                        last_position = position
                        last_rotation = rotation_y
                    packets_received += 1

                except Exception as e:
                    if not DEBUG_PERF:
                        app_logger.error(f"Decode error: {e}")
                    continue

            # Send commands to controller (direct or IK mode)
            try:
                now_t = time.perf_counter()
                dt = max(0.0, min(now_t - last_time, 0.2))
                last_time = now_t

                if not is_paused:
                    if is_direct_mode:
                        slew_v, boom_v, arm_v, bucket_v = last_direct_commands
                        controller.give_direct_commands({
                            'rotate': slew_v,
                            'lift_boom': boom_v,
                            'tilt_boom': arm_v,
                            'scoop': bucket_v,
                        })
                    elif last_position is not None:
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

                    # Get actual end-effector position and orientation
                    ee_position, ee_quat = get_pose(full_quats, robot_config)
                    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    ee_rot_y_deg = float(np.degrees(extract_axis_rotation(ee_quat, y_axis)))

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

                    # Encode and send back to GUI (32 bytes: 5 positions + EE rot_y)
                    all_positions = list(joint_positions) + [ee_position]
                    encoded_joints = encode_joint_positions_to_bytes(all_positions)
                    # Append EE rot_y as int16 (2 bytes) using SCALE_ROT
                    SCALE_ROT = 182.0
                    rot_int = max(-32768, min(32767, int(round(ee_rot_y_deg * SCALE_ROT))))
                    if rot_int < 0:
                        rot_int = (1 << 16) + rot_int
                    rh = (rot_int >> 8) & 0xFF
                    rl = rot_int & 0xFF
                    to_sb = lambda b: b if b < 128 else b - 256
                    encoded_joints.extend([to_sb(rh), to_sb(rl)])
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

                        # Main line: overall loop performance (with jitter)
                        loop_std = perf_stats.get('std_loop_time_ms', 0.0)
                        loop_max = perf_stats.get('max_loop_time_ms', 0.0)
                        loop_p95 = perf_stats.get('jitter_p95_ms', 0.0)
                        loop_p99 = perf_stats.get('jitter_p99_ms', 0.0)
                        loop_step = perf_stats.get('max_step_ms', 0.0)
                        line = (f"Loop: {perf_stats['avg_loop_time_ms']:.2f}ms "
                                f"(min={perf_stats['min_loop_time_ms']:.2f} max={loop_max:.2f}) | "
                                f"Jitter: std={loop_std:.2f} p95={loop_p95:.2f} p99={loop_p99:.2f} maxΔ={loop_step:.2f}ms | "
                                f"CPU: {perf_stats['cpu_usage_pct']:.1f}% | "
                                f"Headroom: {perf_stats['avg_headroom_ms']:.2f}ms")

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
                        if network_times and not PERF_ONLY:
                            net_avg = np.mean(network_times)
                            net_min = np.min(network_times)
                            net_max = np.max(network_times)
                            stages.append(f"Net: {net_avg:.2f}ms ({net_min:.2f}-{net_max:.2f})")

                        if stages:
                            line += " | " + " | ".join(stages)

                        # Recent overrun summary
                        over_cnt = perf_stats.get('overrun_count_recent', 0)
                        over_pct = perf_stats.get('overrun_pct_recent', 0.0)
                        over_win = perf_stats.get('overrun_window_sec', 0.0)
                        line += f" | Overruns: {int(over_cnt)} ({over_pct:.1f}% over {over_win:.1f}s)"

                        # Add packet rate at the end (skip in perf-only mode)
                        if not PERF_ONLY:
                            line += f" | Pkt: {packet_rate:.1f}Hz"

                        # Optionally append joint velocity and limiter info
                        vel_line = ""
                        if not PERF_ONLY:
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
        reset_to_normal(quiet=True)
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
