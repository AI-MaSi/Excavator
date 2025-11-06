from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

# ============================================================
# DATA COLLECTION SETTINGS
# ============================================================
# BUTTON CONTROLS:
# - Button 0: Start/Stop data logging (toggles on/off, saves on stop)
# - Button 1: Reload configuration file
# - Button 2: Toggle hydraulic pump on/off
#
# NOTE: Save strategy uses PAUSE approach (optimized for Raspberry Pi):
# - Stops logging
# - Pauses hardware (PWM reset, pump stays on)
# - Saves data synchronously (10-30 seconds)
# - Resumes logging
# This avoids threading/GIL issues on Pi and keeps control loop clean.

# NOTE: Pressure reading is currently DISABLED for initial testing
# When enabled: ADC channels 1-7 only (channel 8 is slew encoder)
# Max ADC read rate: ~20-25Hz realistic (8 channels @ 240SPS / 8)

DATA_COLLECTION_ENABLED = False
TARGET_DURATION_MINUTES = 60  # Target: 60 minutes of driving
AUTO_SAVE_INTERVAL_MINUTES = 10  # Auto-save every N minutes (0 = disabled)
SAMPLING_FREQUENCY = 100  # Hz (matches control loop)
PRESSURE_SAMPLING_FREQUENCY = 20  # Hz (ADC hardware limit, when enabled)
ENABLE_PRESSURE_LOGGING = False  # Set True to enable pressure sensor logging
OUTPUT_DIR = Path("hydraulic_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description='Simple drive controller with optional data logging')
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

if DEBUG_PULSE_WIDTHS:
    print("\n" + "="*60)
    print("  DEBUG MODE ENABLED")
    print("  Pulse widths (μs) will be printed for all active channels")
    print("  Use this to find where valve movement actually starts")
    print("="*60 + "\n")

# ============================================================
# DATA COLLECTION SETUP
# ============================================================
class DataLogger:
    """Collects hydraulic actuator data for LSTM training.

    Data format (Egli-style approach):
    - 100Hz: valve commands, joint positions (IMU pitch), joint velocities, pressures
    - Joint order: [lift, tilt, scoop] (matching IMU order: boom, arm, bucket)
    - Saves as CSV for easy post-processing
    """

    def __init__(self):
        self.start_time = None
        self.is_logging = False

        # Data storage (100Hz) - all lists of same length
        self.timestamps = []
        self.valve_commands = []     # Shape: (N, 3) - [lift, tilt, scoop]
        self.joint_positions = []    # Shape: (N, 3) - [lift, tilt, scoop] pitch angles (rad)
        self.joint_velocities = []   # Shape: (N, 3) - [lift, tilt, scoop] (rad/s)
        self.joint_pressures = []    # Shape: (N, 3) - [lift, tilt, scoop] averaged chambers (V)

        # Previous joint positions for velocity calculation
        self.prev_joint_positions = None
        self.prev_timestamp = None

        # Pressure data (~20Hz, will be held to 100Hz)
        self.last_pressure_reading = None  # Last valid pressure reading
        self.pressure_sample_interval = 1.0 / PRESSURE_SAMPLING_FREQUENCY  # ~50ms
        self.last_pressure_sample_time = None

        # Performance tracking
        self.loop_times = []
        self.compute_times = []
        self.timing_violations = 0
        self.loop_count = 0
        self.last_loop_time = None

    def quaternion_to_pitch(self, quat):
        """Convert quaternion [w,x,y,z] to pitch angle in radians."""
        if quat is None or len(quat) != 4:
            return 0.0
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Pitch (rotation about Y axis)
        pitch = np.arctan2(2.0 * (w*y - z*x), 1.0 - 2.0 * (x*x + y*y))
        return pitch

    def start(self):
        """Start data collection."""
        self.start_time = time.time()
        self.is_logging = True
        print(f"\n{'='*60}")
        print(f"  DATA COLLECTION STARTED")
        print(f"  Target duration: {TARGET_DURATION_MINUTES} minutes")
        print(f"  IMU/Command sampling: {SAMPLING_FREQUENCY} Hz")
        if ENABLE_PRESSURE_LOGGING:
            print(f"  Pressure sampling: {PRESSURE_SAMPLING_FREQUENCY} Hz (channels 1-7)")
        else:
            print(f"  Pressure sampling: DISABLED (testing mode)")
        print(f"{'='*60}\n")

    def log_sample(self, commands, hardware_interface):
        """Log a single data sample for LSTM training.

        Args:
            commands: List of 8 PWM commands [scoop, lift, extra1, rotate, tilt, extra2, track_r, track_l]
            hardware_interface: Hardware interface object

        Returns:
            Compute time in seconds (for performance tracking)
        """
        if not self.is_logging:
            return 0.0

        compute_start = time.perf_counter()
        current_time = time.time() - self.start_time

        # ========== 100Hz Data (every sample) ==========
        # Read IMU pitch angles directly (CSV stream provides pitch)
        imu_pitches = hardware_interface.read_imu_pitch()  # List of 3 pitches [rad]

        # Extract valve commands: [lift, tilt, scoop] (reordered to match IMU order)
        # Original: commands[0]=scoop, commands[1]=lift, commands[4]=tilt
        valve_cmds = [
            commands[1],  # lift (boom IMU)
            commands[4],  # tilt (arm IMU)
            commands[0],  # scoop (bucket IMU)
        ]

        # Use streamed pitch directly (radians)
        if imu_pitches and len(imu_pitches) == 3:
            joint_pos = [float(imu_pitches[0]), float(imu_pitches[1]), float(imu_pitches[2])]
        else:
            joint_pos = [0.0, 0.0, 0.0]  # Fallback if IMU read fails

        # Compute joint velocities (numerical derivative)
        if self.prev_joint_positions is not None and self.prev_timestamp is not None:
            dt = current_time - self.prev_timestamp
            if dt > 0.001:  # Avoid division by very small numbers
                joint_vel = [
                    (joint_pos[i] - self.prev_joint_positions[i]) / dt
                    for i in range(3)
                ]
            else:
                joint_vel = [0.0, 0.0, 0.0]
        else:
            joint_vel = [0.0, 0.0, 0.0]  # First sample has no velocity

        # Store for next iteration
        self.prev_joint_positions = joint_pos.copy()
        self.prev_timestamp = current_time

        # ========== 20Hz Pressure Data (every ~50ms) ==========
        if ENABLE_PRESSURE_LOGGING:
            should_sample_pressure = False
            if self.last_pressure_sample_time is None:
                should_sample_pressure = True
            elif (current_time - self.last_pressure_sample_time) >= self.pressure_sample_interval:
                should_sample_pressure = True

            if should_sample_pressure:
                # Read pressure sensors from ADC (channels 1-7 only, skip channel 8)
                try:
                    pressure_readings = hardware_interface.adc.read_sensors()

                    # Extract and average chamber pressures per joint:
                    # Joint order: [lift, tilt, scoop]
                    lift_retract = pressure_readings.get("LiftBoom retract ps", 0.0)
                    lift_extend = pressure_readings.get("LiftBoom extend ps", 0.0)
                    tilt_retract = pressure_readings.get("TiltBoom retract ps", 0.0)
                    tilt_extend = pressure_readings.get("TiltBoom extend ps", 0.0)
                    scoop_extend = pressure_readings.get("Scoop extend ps", 0.0)
                    scoop_retract = pressure_readings.get("Scoop retract ps", 0.0)

                    # Average the two chambers for each joint
                    joint_pressures = [
                        (lift_retract + lift_extend) / 2.0,   # lift
                        (tilt_retract + tilt_extend) / 2.0,   # tilt
                        (scoop_extend + scoop_retract) / 2.0, # scoop
                    ]

                    self.last_pressure_reading = joint_pressures
                    self.last_pressure_sample_time = current_time

                except Exception as e:
                    print(f"Warning: Pressure read failed: {e}")
                    if self.last_pressure_reading is None:
                        self.last_pressure_reading = [0.0, 0.0, 0.0]

            # Store pressure data (held from last reading if not sampled this cycle)
            if self.last_pressure_reading:
                pressures = self.last_pressure_reading.copy()
            else:
                pressures = [0.0, 0.0, 0.0]
        else:
            # Pressure logging disabled - store dummy data
            pressures = [0.0, 0.0, 0.0]

        # Store all data for this sample
        self.timestamps.append(current_time)
        self.valve_commands.append(valve_cmds)
        self.joint_positions.append(joint_pos)
        self.joint_velocities.append(joint_vel)
        self.joint_pressures.append(pressures)

        compute_end = time.perf_counter()
        return compute_end - compute_start

    def record_loop_time(self, loop_duration, compute_time):
        """Record main loop timing (called from main loop).

        Args:
            loop_duration: Time from last iteration start to this iteration start (seconds)
            compute_time: Actual compute time excluding sleep (seconds)
        """
        target_period = 1.0 / SAMPLING_FREQUENCY  # 0.01s for 100Hz

        self.loop_times.append(loop_duration)
        self.compute_times.append(compute_time)
        self.loop_count += 1
        self.last_loop_time = loop_duration

        # Check for timing violations (compute exceeded target period)
        if compute_time > target_period:
            self.timing_violations += 1

        # Keep only last 1000 samples (rolling window)
        if len(self.loop_times) > 1000:
            self.loop_times.pop(0)
            self.compute_times.pop(0)

    def get_elapsed_time(self):
        """Get elapsed time in minutes."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 60.0

    def get_sample_count(self):
        """Get number of samples collected."""
        return len(self.timestamps)

    def get_debug_info(self, hardware_interface):
        """Get current debug information for live monitoring.

        Returns:
            Dict with current joint angles (deg), loop timing, and data stats
        """
        debug_info = {}

        # Get current joint angles in degrees (from streamed pitch)
        imu_pitches = hardware_interface.read_imu_pitch()
        if imu_pitches and len(imu_pitches) == 3:
            lift_pitch = np.degrees(imu_pitches[0])
            tilt_pitch = np.degrees(imu_pitches[1])
            scoop_pitch = np.degrees(imu_pitches[2])
            debug_info['imu_angles'] = {
                'lift': lift_pitch,
                'tilt': tilt_pitch,
                'scoop': scoop_pitch
            }
        else:
            debug_info['imu_angles'] = None

        # Loop performance
        if self.loop_times and self.compute_times:
            loop_times_ms = np.array(self.loop_times) * 1000.0
            compute_times_ms = np.array(self.compute_times) * 1000.0
            actual_hz = 1.0 / np.mean(self.loop_times)
            target_period_ms = 1000.0 / SAMPLING_FREQUENCY  # 10ms for 100Hz

            # Calculate headroom (time available for sleep)
            avg_headroom_ms = target_period_ms - np.mean(compute_times_ms)

            # Calculate CPU usage (compute time / total time)
            cpu_usage_pct = (np.mean(compute_times_ms) / target_period_ms) * 100.0

            # Violation percentage
            violation_pct = (self.timing_violations / self.loop_count * 100) if self.loop_count > 0 else 0.0

            debug_info['loop_time_ms'] = {
                'avg_loop': np.mean(loop_times_ms),
                'avg_compute': np.mean(compute_times_ms),
                'max_compute': np.max(compute_times_ms),
                'headroom': avg_headroom_ms,
                'cpu_usage_pct': cpu_usage_pct,
                'actual_hz': actual_hz,
                'target_hz': SAMPLING_FREQUENCY,
                'violation_pct': violation_pct
            }
            # Check if keeping up (within 5% of target and headroom positive)
            debug_info['keeping_up'] = actual_hz >= (SAMPLING_FREQUENCY * 0.95) and avg_headroom_ms > 0
        else:
            debug_info['loop_time_ms'] = None
            debug_info['keeping_up'] = True

        # Data collection stats
        debug_info['samples_collected'] = len(self.timestamps)
        debug_info['elapsed_minutes'] = self.get_elapsed_time()

        return debug_info

    def save(self):
        """Save collected data to CSV for LSTM training.

        Saves in format:
        timestamp, valve_cmd_0, valve_cmd_1, valve_cmd_2,
                   joint_pos_0, joint_pos_1, joint_pos_2,
                   joint_vel_0, joint_vel_1, joint_vel_2,
                   pressure_0, pressure_1, pressure_2

        Note: This blocks execution during save. Use save_with_pause()
        to automatically pause hardware during save.
        """
        if not self.timestamps:
            print("No data to save!")
            return None

        print(f"\n{'='*60}")
        print(f"  SAVING DATA...")
        print(f"{'='*60}")

        save_start = time.perf_counter()

        # Convert to numpy arrays
        timestamps = np.array(self.timestamps)
        valve_cmds = np.array(self.valve_commands)  # (N, 3)
        joint_pos = np.array(self.joint_positions)   # (N, 3)
        joint_vel = np.array(self.joint_velocities)  # (N, 3)
        pressures = np.array(self.joint_pressures)   # (N, 3)

        # Build CSV data with proper column names
        # Format: timestamp, valve_cmd_0..2, joint_pos_0..2, joint_vel_0..2, pressure_0..2
        import pandas as pd

        data_dict = {
            'timestamp': timestamps,
            'valve_cmd_0': valve_cmds[:, 0],  # lift
            'valve_cmd_1': valve_cmds[:, 1],  # tilt
            'valve_cmd_2': valve_cmds[:, 2],  # scoop
            'joint_pos_0': joint_pos[:, 0],   # lift (rad)
            'joint_pos_1': joint_pos[:, 1],   # tilt (rad)
            'joint_pos_2': joint_pos[:, 2],   # scoop (rad)
            'joint_vel_0': joint_vel[:, 0],   # lift (rad/s)
            'joint_vel_1': joint_vel[:, 1],   # tilt (rad/s)
            'joint_vel_2': joint_vel[:, 2],   # scoop (rad/s)
            'pressure_0': pressures[:, 0],    # lift (V)
            'pressure_1': pressures[:, 1],    # tilt (V)
            'pressure_2': pressures[:, 2],    # scoop (V)
        }

        df = pd.DataFrame(data_dict)

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"lstm_training_data_{timestamp_str}.csv"

        # Save CSV
        df.to_csv(filename, index=False)

        save_time = time.perf_counter() - save_start

        # Print statistics
        print(f"\nSaved {len(timestamps)} samples in {save_time:.1f}s")
        self.print_statistics(df)
        print(f"\n✓ Data saved to: {filename}")
        print(f"{'='*60}\n")

        return filename

    def save_with_pause(self, hardware_interface):
        """Save data with hardware paused (recommended approach for Pi).

        This method:
        1. Stops logging
        2. Pauses hardware (PWM reset, pump stays on)
        3. Saves data synchronously
        4. Resumes logging

        Args:
            hardware_interface: Hardware interface to pause during save

        Returns:
            Filename of saved data
        """
        if not self.timestamps:
            print("No data to save!")
            return None

        # Stop logging
        was_logging = self.is_logging
        self.is_logging = False

        # Pause hardware (stop all movement, keep pump running)
        print("\n[PAUSE] Stopping machine for save...")
        hardware_interface.reset(reset_pump=False)

        # Give hardware time to settle
        time.sleep(0.2)

        # Save data (blocking - but that's OK, we're paused)
        filename = self.save()

        # Resume logging if it was active
        if was_logging:
            print("[RESUME] Restarting data logging...")
            self.is_logging = True
            print("✓ Ready to continue driving!\n")

        return filename

    def print_statistics(self, df):
        """Print data collection statistics.

        Args:
            df: Pandas DataFrame with collected data
        """
        duration_sec = df['timestamp'].iloc[-1]
        duration_min = duration_sec / 60.0
        num_samples = len(df)

        print(f"\n  === DATA STATISTICS ===")
        print(f"  Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
        print(f"  Samples: {num_samples}")
        print(f"  Actual sampling rate: {num_samples / duration_sec:.1f} Hz")
        print(f"  Format: CSV (ready for LSTM training)")

        # Valve command statistics
        print(f"\n  Valve Commands (3 joints: lift, tilt, scoop):")
        for i, label in enumerate(['Lift', 'Tilt', 'Scoop']):
            col = f'valve_cmd_{i}'
            ch_data = df[col].values
            active_pct = (np.abs(ch_data) > 0.1).sum() / len(ch_data) * 100
            print(f"    {label}: range [{ch_data.min():.2f}, {ch_data.max():.2f}], "
                  f"active {active_pct:.1f}% of time")

        # Joint position statistics
        print(f"\n  Joint Positions (pitch angles in radians):")
        for i, label in enumerate(['Lift', 'Tilt', 'Scoop']):
            col = f'joint_pos_{i}'
            pos_data = df[col].values
            print(f"    {label}: range [{pos_data.min():.3f}, {pos_data.max():.3f}] rad "
                  f"({np.degrees(pos_data.min()):.1f}° to {np.degrees(pos_data.max()):.1f}°)")

        # Joint velocity statistics
        print(f"\n  Joint Velocities (rad/s):")
        for i, label in enumerate(['Lift', 'Tilt', 'Scoop']):
            col = f'joint_vel_{i}'
            vel_data = df[col].values
            print(f"    {label}: max {np.abs(vel_data).max():.3f} rad/s "
                  f"({np.degrees(np.abs(vel_data).max()):.1f}°/s)")

        # Pressure statistics
        if ENABLE_PRESSURE_LOGGING:
            print(f"\n  Cylinder Pressures (averaged chambers, V):")
            for i, label in enumerate(['Lift', 'Tilt', 'Scoop']):
                col = f'pressure_{i}'
                press_data = df[col].values
                print(f"    {label}: range [{press_data.min():.3f}, {press_data.max():.3f}] V, "
                      f"mean {press_data.mean():.3f} V")
        else:
            print(f"\n  Cylinder Pressures: DISABLED (dummy zeros)")

        print(f"\n  Joint Order: 0=Lift, 1=Tilt, 2=Scoop")


# Initialize data logger
logger = DataLogger()

# ============================================================
# MAIN CONTROL LOOP
# ============================================================
if server.handshake():
    server.start_receiving()
    print("UDP connection established, starting control loop...")

    # Button state tracking
    button_0_prev = 0.0
    button_1_prev = 0.0
    button_2_prev = 0.0
    button_threshold = 0.5

    # Auto-start logging
    if DATA_COLLECTION_ENABLED:
        logger.start()

    last_status_time = time.time()
    last_debug_time = time.time()
    last_auto_save_time = time.time()
    DEBUG_INTERVAL = 2.0  # Print debug info every 2 seconds

    # Loop timing tracking (use perf_counter for accurate timing)
    loop_period = 1.0 / SAMPLING_FREQUENCY  # 0.01s for 100Hz
    next_run_time = time.perf_counter()
    last_loop_start = next_run_time

    while True:
        loop_start = time.perf_counter()
        compute_time = 0.0  # Initialize here for all loop paths

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
            button_0 = float_data[0]
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

            # Button 0: Start/Stop logging
            if button_0 > button_threshold and button_0_prev <= button_threshold:
                if not logger.is_logging:
                    print("\n[Button 0] Starting data collection...")
                    logger.start()
                else:
                    print("\n[Button 0] Stopping data collection and saving...")
                    logger.save_with_pause(hardware)
                    logger.is_logging = False

            button_0_prev = button_0
            button_1_prev = button_1
            button_2_prev = button_2

            # Build name-based command mapping (more friendly to students, no need to mess around with indexes)
            named = {
                'scoop': right_rl,
                'lift_boom': right_ud,
                'rotate': left_rl,
                'tilt_boom': left_ud,
                'trackR': right_paddle,
                'trackL': left_paddle,
            }

            # Send commands to hardware (zero unspecified channels per call)
            hardware.send_named_pwm_commands(named, unset_to_zero=True) # unset_to_zero is True by default, but here for clarity

            # Debug: Print pulse widths for all commanded channels
            if DEBUG_PULSE_WIDTHS and hardware.pwm_controller:
                debug_str = "[PULSE DEBUG] "
                pulse_info = []
                for name, value in named.items():
                    # Get the pulse width that will be used (or was just used)
                    pulse = hardware.pwm_controller.compute_pulse(name, value)
                    if pulse is not None:
                        # Only show channels with non-zero commands to reduce clutter
                        if abs(value) > 0.01:
                            pulse_info.append(f"{name}={value:+.2f} → {pulse:.1f}μs")

                if pulse_info:
                    print(debug_str + " | ".join(pulse_info))

            # Log data
            if logger.is_logging:
                # Keep original list for logging shape/compatibility
                commands_list = [right_rl, right_ud, 0, left_rl, left_ud, 0, right_paddle, left_paddle]
                compute_time = logger.log_sample(commands_list, hardware)

            # Lightweight debug output (every 2 seconds)
            current_time = time.time()
            if logger.is_logging and (current_time - last_debug_time >= DEBUG_INTERVAL):
                last_debug_time = current_time
                debug = logger.get_debug_info(hardware)

                # Build compact debug string
                debug_str = f"[DEBUG] "

                # Joint angles
                if debug['imu_angles']:
                    angles = debug['imu_angles']
                    debug_str += f"Joints: Lift={angles['lift']:+6.1f}° Tilt={angles['tilt']:+6.1f}° Scoop={angles['scoop']:+6.1f}° | "

                # Loop timing
                if debug['loop_time_ms']:
                    timing = debug['loop_time_ms']
                    status_icon = "✓" if debug['keeping_up'] else "⚠"
                    debug_str += f"Loop: {timing['actual_hz']:.1f}Hz | "
                    debug_str += f"Compute: {timing['avg_compute']:.2f}ms (headroom {timing['headroom']:.2f}ms) {status_icon}"

                    # Add violation warning if any
                    if timing['violation_pct'] > 0:
                        debug_str += f" [violations: {timing['violation_pct']:.1f}%]"
                    debug_str += " | "

                # Data stats
                debug_str += f"Samples: {debug['samples_collected']}"

                print(debug_str)

            # Auto-save check (periodic paused saves)
            if logger.is_logging and AUTO_SAVE_INTERVAL_MINUTES > 0:
                if (current_time - last_auto_save_time) >= (AUTO_SAVE_INTERVAL_MINUTES * 60):
                    last_auto_save_time = current_time
                    print(f"\n[AUTO-SAVE] Periodic save triggered ({AUTO_SAVE_INTERVAL_MINUTES} min interval)")
                    logger.save_with_pause(hardware)  # Pause machine during save

            # Extended status updates (every 30 seconds)
            if current_time - last_status_time >= 30.0:
                last_status_time = current_time

                if logger.is_logging:
                    elapsed_min = logger.get_elapsed_time()
                    samples = logger.get_sample_count()
                    progress = (elapsed_min / TARGET_DURATION_MINUTES) * 100

                    print(f"\n[STATUS] Time: {elapsed_min:.1f}/{TARGET_DURATION_MINUTES} min "
                          f"({progress:.1f}%) | Samples: {samples}")

                    # Check if target duration reached
                    if elapsed_min >= TARGET_DURATION_MINUTES:
                        print(f"\n{'='*60}")
                        print(f"  TARGET DURATION REACHED ({TARGET_DURATION_MINUTES} minutes)")
                        print(f"  Stopping data collection...")
                        print(f"{'='*60}")
                        logger.save_with_pause(hardware)  # Pause and save
                        print("\nData collection complete! You can continue driving or exit.")

                # Print input rate
                if hardware.pwm_controller:
                    avg_rate = hardware.pwm_controller.get_average_input_rate()
                    print(f"Average input rate: {avg_rate:.2f}Hz")

        else:
            # No data received (safety handled internally)
            pass

        # Record actual loop timing (iteration-to-iteration)
        if logger.is_logging:
            loop_duration = loop_start - last_loop_start
            if loop_duration > 0.001:  # Skip first iteration
                logger.record_loop_time(loop_duration, compute_time)
        last_loop_start = loop_start

        # Adaptive sleep to maintain exact frequency (from excavator_controller.py)
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
if logger.get_sample_count() > 0:
    print("Final save...")
    logger.is_logging = False  # Stop logging
    hardware.reset(reset_pump=False)  # Pause hardware
    logger.save()  # Save synchronously
hardware.reset(reset_pump=True)  # Final cleanup with pump off
