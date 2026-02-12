from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.perf_tracker import LoopPerfTracker
import time
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description="Simple drive logging with optional perf metrics")
parser.add_argument("--perf", action="store_true", help="Show performance metrics in status output")
args = parser.parse_args()

# ============================================================
# DATA COLLECTION SETTINGS
# ============================================================
# BUTTON CONTROLS:
# - Button 0: Start/Stop data logging (toggles on/off, saves on stop)
# - Button 1: Toggle IMU debug display (10Hz, requires logging to be active)
# - Button 2: Toggle hydraulic pump on/off
# - Button 3: Reload configuration file
#
# NOTE: Save strategy uses PAUSE approach (optimized for Raspberry Pi):
# - Stops logging
# - Pauses hardware (PWM reset, pump stays on)
# - Saves data synchronously
# - Resumes logging
# This avoids threading/GIL issues on Pi and keeps control loop clean.

# NOTE: Pressure reading runs at ~20Hz (ADC channels 1-7 only).

DATA_COLLECTION_ENABLED = False
TARGET_DURATION_MINUTES = 60  # Target: 60 minutes of driving
AUTO_SAVE_INTERVAL_MINUTES = 10  # Auto-save every N minutes (0 = disabled)
SAMPLING_FREQUENCY = 100  # Hz (logging rate)
PWM_UPDATE_FREQUENCY = 100  # Hz (PWM update rate - higher for smoother ramps)
PRESSURE_SAMPLING_FREQUENCY = 20  # Hz (ADC hardware limit, when enabled)
ENABLE_PRESSURE_LOGGING = True  # Enable pressure sensor logging (ADC)
STATUS_PRINT_INTERVAL = 5.0  # seconds
IMU_DISPLAY_INTERVAL = 0.1  # seconds (10 Hz)
OUTPUT_DIR = Path("hydraulic_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# INITIALIZE HARDWARE & NETWORK
# ============================================================
server = UDPSocket(local_id=2)
# server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)
server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)

hardware = HardwareInterface(
    config_file="configuration_files/servo_config_200.yaml",
    pump_variable=False,
    toggle_channels=True,
    input_rate_threshold=5,
    stale_timeout_s=0.5,  # Reject commands older than 500ms, reset PWM if no commands for 500ms
    adc_channels=[
        "LiftBoom retract ps",
        "LiftBoom extend ps",
        "TiltBoom retract ps",
        "TiltBoom extend ps",
        "Scoop extend ps",
        "Scoop retract ps",
        "Pump ps",
    ],
    adc_sample_hz=PRESSURE_SAMPLING_FREQUENCY,
    # Subsystem enable flags (all default to True)
    enable_pwm=True,
    enable_imu=True,
    enable_adc=True,
    # Keep PWM oscillator running on cleanup (outputs stay at center, prevents servo faults)
    cleanup_disable_osc=False,
)

print("Waiting for hardware to be ready...")
try:
    while not hardware.is_hardware_ready():
        time.sleep(0.1)
    print("Hardware ready!")
except HardwareFaultError as e:
    print(f"\n*** HARDWARE FAULT: {e.subsystem} ***")
    print(f"Reason: {e.reason}")
    print("\nCheck your configuration and hardware connections.")
    hardware.shutdown()
    raise SystemExit(1)

# Enable perf tracking if --perf flag is set
main_loop_perf = LoopPerfTracker(enabled=args.perf)
if args.perf:
    try:
        hardware.set_perf_enabled(True)
        hardware.reset_perf_stats()
        print("[PERF] Performance metrics enabled")
    except Exception as e:
        print(f"[PERF] Failed to enable: {e}")

# ============================================================
# DATA COLLECTION SETUP
# ============================================================
class DataLogger:
    """Collects hydraulic actuator data for analysis and model training.

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

        self.joint_pressures_extend = []   # Shape: (N, 3) - [lift, tilt, scoop] extend chamber (V)
        self.joint_pressures_retract = []  # Shape: (N, 3) - [lift, tilt, scoop] retract chamber (V)
        self.joint_pressure_ts = []  # ADC sample timestamps (seconds since start)

        # Previous joint positions for velocity calculation
        self.prev_joint_positions = None
        self.prev_timestamp = None

        # Pressure data (~20Hz, will be held to 100Hz)
        self.last_pressure_reading_extend = None  # Last valid extend pressure reading
        self.last_pressure_reading_retract = None  # Last valid retract pressure reading
        self.pressure_sample_interval = 1.0 / PRESSURE_SAMPLING_FREQUENCY  # ~50ms
        self.last_pressure_sample_time = None
        self.last_pressure_wall_ts = None  # wallclock timestamp from ADC thread

        # Performance tracking
        self.loop_times = []
        self.compute_times = []
        self.timing_violations = 0
        self.loop_count = 0
        self.last_loop_time = None
        self.last_joint_pos = [np.nan, np.nan, np.nan]

    def quaternion_to_pitch(self, quat):
        """Convert quaternion [w,x,y,z] to pitch angle in radians."""
        if quat is None or len(quat) < 4:
            return np.nan
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Pitch (rotation about Y axis)
        return np.arctan2(2.0 * (w * y - z * x), 1.0 - 2.0 * (x * x + y * y))

    def start(self):
        """Start data collection."""
        # Reset buffers so each logging session produces a clean, monotonic timeline.
        self.start_time = time.time()
        self.is_logging = True
        self.timestamps = []
        self.valve_commands = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_pressures_extend = []
        self.joint_pressures_retract = []
        self.joint_pressure_ts = []
        self.prev_joint_positions = None
        self.prev_timestamp = None
        self.last_pressure_reading_extend = None
        self.last_pressure_reading_retract = None
        self.last_pressure_sample_time = None
        self.last_pressure_wall_ts = None
        self.loop_times = []
        self.compute_times = []
        self.timing_violations = 0
        self.loop_count = 0
        self.last_loop_time = None
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
        """Log a single data sample for later usage.

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
        # Read IMU quaternions and compute pitch angles (radians)
        imu_quats = hardware_interface.read_imu_data()  # List of 3 quats [w,x,y,z]

        # Extract valve commands: [lift, tilt, scoop] (reordered to match IMU order)
        # Original: commands[0]=scoop, commands[1]=lift, commands[4]=tilt
        valve_cmds = [
            commands[1],  # lift (boom IMU)
            commands[4],  # tilt (arm IMU)
            commands[0],  # scoop (bucket IMU)
        ]

        # Convert quaternions to pitch angles (radians)
        if imu_quats and len(imu_quats) >= 3:
            joint_pos = [
                float(self.quaternion_to_pitch(imu_quats[0])),
                float(self.quaternion_to_pitch(imu_quats[1])),
                float(self.quaternion_to_pitch(imu_quats[2])),
            ]
        else:
            joint_pos = [np.nan, np.nan, np.nan]  # Fallback if IMU read fails

        # Compute joint velocities (numerical derivative)
        if self.prev_joint_positions is not None and self.prev_timestamp is not None:
            dt = current_time - self.prev_timestamp
            if dt > 0.001:  # Avoid division by very small numbers
                joint_vel = [
                    (joint_pos[i] - self.prev_joint_positions[i]) / dt
                    for i in range(3)
                ]
            else:
                joint_vel = [np.nan, np.nan, np.nan]
        else:
            joint_vel = [np.nan, np.nan, np.nan]  # First sample has no velocity

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
                    adc_snapshot = hardware_interface.get_latest_adc_snapshot()
                    pressure_readings = adc_snapshot.get("readings", {})
                    self.last_pressure_wall_ts = adc_snapshot.get("timestamp", None)

                    # Extract chamber pressures per joint:
                    # Joint order: [lift, tilt, scoop]
                    lift_retract = pressure_readings.get("LiftBoom retract ps", 0.0)
                    lift_extend = pressure_readings.get("LiftBoom extend ps", 0.0)
                    tilt_retract = pressure_readings.get("TiltBoom retract ps", 0.0)
                    tilt_extend = pressure_readings.get("TiltBoom extend ps", 0.0)
                    scoop_extend = pressure_readings.get("Scoop extend ps", 0.0)
                    scoop_retract = pressure_readings.get("Scoop retract ps", 0.0)

                    joint_pressures_extend = [
                        lift_extend,   # lift
                        tilt_extend,   # tilt
                        scoop_extend,  # scoop
                    ]
                    joint_pressures_retract = [
                        lift_retract,   # lift
                        tilt_retract,   # tilt
                        scoop_retract,  # scoop
                    ]

                    self.last_pressure_reading_extend = joint_pressures_extend
                    self.last_pressure_reading_retract = joint_pressures_retract
                    self.last_pressure_sample_time = current_time

                except Exception as e:
                    print(f"Warning: Pressure read failed: {e}")
                    if self.last_pressure_reading_extend is None:
                        self.last_pressure_reading_extend = [np.nan, np.nan, np.nan]
                    if self.last_pressure_reading_retract is None:
                        self.last_pressure_reading_retract = [np.nan, np.nan, np.nan]

            # Store pressure data (held from last reading if not sampled this cycle)
            if self.last_pressure_reading_extend:
                pressures_extend = self.last_pressure_reading_extend.copy()
            else:
                pressures_extend = [np.nan, np.nan, np.nan]
            if self.last_pressure_reading_retract:
                pressures_retract = self.last_pressure_reading_retract.copy()
            else:
                pressures_retract = [np.nan, np.nan, np.nan]
            if self.last_pressure_wall_ts is not None:
                pressure_ts = max(0.0, self.last_pressure_wall_ts - self.start_time)
            else:
                pressure_ts = np.nan
        else:
            # Pressure logging disabled - store dummy data
            pressures_extend = [np.nan, np.nan, np.nan]
            pressures_retract = [np.nan, np.nan, np.nan]
            pressure_ts = np.nan

        # Store all data for this sample
        self.timestamps.append(current_time)
        self.valve_commands.append(valve_cmds)
        self.joint_positions.append(joint_pos)
        self.joint_velocities.append(joint_vel)
        self.joint_pressures_extend.append(pressures_extend)
        self.joint_pressures_retract.append(pressures_retract)
        self.joint_pressure_ts.append(pressure_ts)
        self.last_joint_pos = joint_pos

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

    def save(self):
        """Save collected data to CSV.

        Saves in format:
        timestamp, valve_cmd_0, valve_cmd_1, valve_cmd_2,
                   joint_pos_0, joint_pos_1, joint_pos_2,
                   joint_vel_0, joint_vel_1, joint_vel_2,
                   pext_lift, pext_tilt, pext_scoop,
                   pret_lift, pret_tilt, pret_scoop,
                   pressure_ts

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
        pressures_extend = np.array(self.joint_pressures_extend)   # (N, 3)
        pressures_retract = np.array(self.joint_pressures_retract)  # (N, 3)
        pressure_ts = np.array(self.joint_pressure_ts)  # (N,)

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
            'pext_lift': pressures_extend[:, 0],   # lift (V)
            'pext_tilt': pressures_extend[:, 1],   # tilt (V)
            'pext_scoop': pressures_extend[:, 2],  # scoop (V)
            'pret_lift': pressures_retract[:, 0],  # lift (V)
            'pret_tilt': pressures_retract[:, 1],  # tilt (V)
            'pret_scoop': pressures_retract[:, 2], # scoop (V)
            'pressure_ts': pressure_ts,       # wallclock seconds since start for ADC sample
        }

        df = pd.DataFrame(data_dict)

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"drive_log_{timestamp_str}.csv"

        # Save CSV
        df.to_csv(filename, index=False)

        save_time = time.perf_counter() - save_start

        # Print statistics
        print(f"\nSaved {len(timestamps)} samples in {save_time:.1f}s")
        self.print_statistics(df)
        print(f"\n[OK] Data saved to: {filename}")
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
            print("[RESUME] Ready to continue driving!\n")

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
        print(f"  Format: CSV")

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
                  f"({np.degrees(pos_data.min()):.1f} deg to {np.degrees(pos_data.max()):.1f} deg)")

        # Joint velocity statistics
        print(f"\n  Joint Velocities (rad/s):")
        for i, label in enumerate(['Lift', 'Tilt', 'Scoop']):
            col = f'joint_vel_{i}'
            vel_data = df[col].values
            print(f"    {label}: max {np.abs(vel_data).max():.3f} rad/s "
                  f"({np.degrees(np.abs(vel_data).max()):.1f} deg/s)")

        # Pressure statistics
        if ENABLE_PRESSURE_LOGGING:
            print(f"\n  Cylinder Pressures (extend/retract, V):")
            for label in ['lift', 'tilt', 'scoop']:
                col_extend = f'pext_{label}'
                col_retract = f'pret_{label}'
                extend_data = df[col_extend].values
                retract_data = df[col_retract].values
                print(f"    {label.capitalize()} extend: range [{extend_data.min():.3f}, {extend_data.max():.3f}] V, "
                      f"mean {extend_data.mean():.3f} V")
                print(f"    {label.capitalize()} retract: range [{retract_data.min():.3f}, {retract_data.max():.3f}] V, "
                      f"mean {retract_data.mean():.3f} V")

            # Observed ADC sample rate using unique ADC timestamps
            try:
                pressure_ts = df['pressure_ts'].values
                valid_ts = pressure_ts[pressure_ts > 0]
                unique_ts = np.unique(valid_ts)
                if len(unique_ts) > 1:
                    pressure_duration = unique_ts[-1] - unique_ts[0]
                    observed_hz = len(unique_ts) / pressure_duration if pressure_duration > 0 else float("nan")
                    print(f"  Observed pressure sampling: {observed_hz:.1f} Hz "
                          f"(target {PRESSURE_SAMPLING_FREQUENCY} Hz)")
            except Exception as e:
                print(f"  Pressure sampling stats unavailable: {e}")
        else:
            print(f"\n  Cylinder Pressures: DISABLED (dummy zeros)")

        print(f"\n  Joint Order: 0=Lift, 1=Tilt, 2=Scoop")


# Initialize data logger
logger = DataLogger()

# ============================================================
# MAIN CONTROL LOOP
# ============================================================
if server.handshake(timeout=30.0):
    server.start_receiving()
    print("UDP connection established, starting control loop...")

    # Button state tracking
    button_0_prev = 0.0
    button_1_prev = 0.0
    button_2_prev = 0.0
    button_3_prev = 0.0
    button_threshold = 0.5

    # Auto-start logging
    if DATA_COLLECTION_ENABLED:
        logger.start()

    last_status_time = time.time()
    last_auto_save_time = time.time()
    last_imu_display_time = time.time()
    display_imu_angles = False

    # Loop timing tracking (use perf_counter for accurate timing)
    loop_period = 1.0 / PWM_UPDATE_FREQUENCY  # 0.005s for 200Hz PWM updates
    next_run_time = time.perf_counter()
    last_loop_start = next_run_time

    # Log every Nth iteration to maintain SAMPLING_FREQUENCY for data
    log_divider = PWM_UPDATE_FREQUENCY // SAMPLING_FREQUENCY  # 2 for 200/100
    log_counter = 0

    try:
        while True:
                loop_start = time.perf_counter()
                main_loop_perf.tick_start()

                # Get latest controller data as floats in [-1, 1]
                float_data = server.get_latest_floats()

                # Capture monotonic timestamp for stale command detection
                command_ts = time.monotonic() if float_data else None

                if float_data:

                    # Map controller inputs
                    right_rl = float_data[9]        # scoop
                    right_ud = float_data[8]        # lift
                    left_rl = float_data[7]         # rotate
                    left_ud = float_data[6]         # tilt
                    right_paddle = float_data[5]    # right track
                    left_paddle = float_data[4]     # left track
                    button_0 = float_data[0]
                    button_1 = float_data[1]
                    button_2 = float_data[2]
                    button_3 = float_data[3]

                    # Button 1: Toggle IMU debug display (requires logging active)
                    if button_1 > button_threshold and button_1_prev <= button_threshold:
                        if not logger.is_logging:
                            # IMU data only updates when logging - display would show stale/NaN values
                            print("\n[Button 1] Cannot enable IMU display - start logging first (Button 0)")
                        else:
                            display_imu_angles = not display_imu_angles
                            state = "ON" if display_imu_angles else "OFF"
                            print(f"\n[Button 1] IMU angle display {state} (10Hz)")

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

                    # Button 3: Reload configuration file
                    if button_3 > button_threshold and button_3_prev <= button_threshold:
                        print("\n[Button 3] Reloading configuration...")
                        if hardware.reload_config():
                            print("[Button 3] Configuration reloaded successfully")
                        else:
                            print("[Button 3] Configuration reload failed")

                    button_0_prev = button_0
                    button_1_prev = button_1
                    button_2_prev = button_2
                    button_3_prev = button_3

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
                    # Pass command_ts for stale detection - if command is older than stale_timeout_s, it will be rejected
                    # unset_to_zero is True by default, but here for clarity
                    hardware.send_named_pwm_commands(named, unset_to_zero=True, command_ts=command_ts)

                    # Log data at SAMPLING_FREQUENCY (every Nth PWM iteration)
                    if logger.is_logging:
                        log_counter += 1
                        if log_counter >= log_divider:
                            log_counter = 0
                            # Keep original list for logging shape/compatibility
                            commands_list = [right_rl, right_ud, 0, left_rl, left_ud, 0, right_paddle, left_paddle]
                            logger.log_sample(commands_list, hardware)

                    current_time = time.time()

                    # Auto-save check (periodic paused saves)
                    if logger.is_logging and AUTO_SAVE_INTERVAL_MINUTES > 0:
                        if (current_time - last_auto_save_time) >= (AUTO_SAVE_INTERVAL_MINUTES * 60):
                            last_auto_save_time = current_time
                            print(f"\n[AUTO-SAVE] Periodic save triggered ({AUTO_SAVE_INTERVAL_MINUTES} min interval)")
                            logger.save_with_pause(hardware)  # Pause machine during save

                    # Basic status updates (every 5 seconds while driving)
                    if current_time - last_status_time >= STATUS_PRINT_INTERVAL:
                        last_status_time = current_time
                        elapsed_min = logger.get_elapsed_time()
                        samples = logger.get_sample_count()
                        logging_state = "ON" if logger.is_logging else "OFF"
                        if logger.is_logging and AUTO_SAVE_INTERVAL_MINUTES > 0:
                            remaining = max(0.0, (AUTO_SAVE_INTERVAL_MINUTES * 60) - (current_time - last_auto_save_time))
                            remaining_str = f"{remaining:.1f}s"
                        else:
                            remaining_str = "n/a"

                        print(f"[STATUS] Logging={logging_state} | Time={elapsed_min:.1f} min | Samples={samples} | Next save in {remaining_str}")

                        # Show perf metrics if --perf flag is set
                        if args.perf:
                            try:
                                perf = hardware.get_perf_stats()
                                loop_stats = main_loop_perf.get_stats()
                                if perf:
                                    imu = perf.get('imu', {})
                                    adc = perf.get('adc', {})
                                    pwm = perf.get('pwm', {})

                                    imu_hz = imu.get('hz', 0)
                                    imu_jitter = imu.get('std_interval_ms', 0)
                                    adc_hz = adc.get('hz', 0)
                                    adc_jitter = adc.get('std_interval_ms', 0)
                                    pwm_hz = pwm.get('hz', 0)
                                    pwm_jitter = pwm.get('loop_std_ms', 0)

                                    # Main loop stats
                                    loop_hz = loop_stats.get('hz', 0)
                                    loop_proc = loop_stats.get('proc_avg_ms', 0)
                                    loop_headroom = loop_stats.get('headroom_avg_ms', 0)
                                    loop_violations = loop_stats.get('violation_pct', 0)

                                    print(f"[PERF]  IMU={imu_hz:.0f}Hz (j={imu_jitter:.2f}ms) | "
                                          f"ADC={adc_hz:.0f}Hz (j={adc_jitter:.2f}ms) | "
                                          f"PWM={pwm_hz:.0f}Hz (j={pwm_jitter:.2f}ms)")
                                    print(f"[LOOP]  {loop_hz:.0f}Hz | proc={loop_proc:.2f}ms | "
                                          f"headroom={loop_headroom:.2f}ms | violations={loop_violations:.1f}%")
                            except Exception:
                                pass

                    # IMU angle display (10Hz when enabled)
                    if display_imu_angles and (current_time - last_imu_display_time >= IMU_DISPLAY_INTERVAL):
                        last_imu_display_time = current_time
                        joint_pos = logger.last_joint_pos
                        if joint_pos and len(joint_pos) >= 3:
                            angles_deg = np.degrees(joint_pos[:3])
                            print(f"[IMU] Lift={angles_deg[0]:+6.1f} deg | Tilt={angles_deg[1]:+6.1f} deg | Scoop={angles_deg[2]:+6.1f} deg")

                else:
                    # No data received (safety handled internally)
                    pass

                # Record actual loop timing (iteration-to-iteration)
                if logger.is_logging:
                    loop_duration = loop_start - last_loop_start
                    if loop_duration > 0.001:  # Skip first iteration
                        loop_compute_time = time.perf_counter() - loop_start
                        logger.record_loop_time(loop_duration, loop_compute_time)
                last_loop_start = loop_start

                # Track main loop compute time (before sleep)
                main_loop_perf.tick_end(target_period_s=loop_period)

                # Adaptive sleep to maintain exact frequency (from excavator_controller.py)
                next_run_time += loop_period
                sleep_time = next_run_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # If we're behind, reset timing to prevent drift
                    next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        # Cleanup always runs
        print("\nShutting down...")
        if logger.get_sample_count() > 0:
            print("Final save...")
            logger.is_logging = False
            hardware.reset(reset_pump=False)
            logger.save()
        hardware.reset(reset_pump=True)
        try:
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")

else:
    print("UDP handshake failed!")
    hardware.reset()
    try:
        hardware.shutdown()
    except Exception:
        pass
