from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

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
# INITIALIZE HARDWARE & NETWORK
# ============================================================
server = UDPSocket(local_id=2)
server.setup("192.168.0.132", 8080, num_inputs=9, num_outputs=0, is_server=True)

hardware = HardwareInterface(
    config_file="configuration_files/linear_config.yaml",
    pump_variable=False,
    toggle_channels=True,
    input_rate_threshold=5
)

print("Waiting for hardware to be ready...")
while not hardware.is_hardware_ready():
    time.sleep(0.1)
print("Hardware ready!")

# ============================================================
# DATA COLLECTION SETUP
# ============================================================
class DataLogger:
    """Collects hydraulic actuator data during manual driving.

    Collects at mixed rates:
    - 100Hz: IMU orientations + valve commands
    - ~20Hz: Chamber pressures (6 channels + pump pressure)
    """

    def __init__(self):
        self.start_time = None
        self.is_logging = False

        # Data storage (100Hz)
        self.timestamps = []
        self.valve_commands = []  # Focus on 3 hydraulic actuators: [scoop, lift, tilt]
        self.imu_quaternions = []  # 3 IMUs (boom, arm, bucket) - each is [w,x,y,z]

        # Pressure data (~20Hz, will be upsampled/held to 100Hz)
        self.chamber_pressures = []  # 6 channels (2 per actuator: extend/retract)
        self.pump_pressure = []  # Pump supply pressure

        # Pressure sampling control
        self.pressure_sample_interval = 1.0 / PRESSURE_SAMPLING_FREQUENCY  # ~50ms
        self.last_pressure_sample_time = None
        self.last_pressure_reading = None

        # Performance tracking (inspired by excavator_controller.py)
        self.loop_times = []          # Total loop period (iteration to iteration)
        self.compute_times = []       # Actual data collection time (without sleep)
        self.timing_violations = 0    # Count of times compute exceeded target
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
        """Log a single data sample.

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
        # Read IMU data
        imu_data = hardware_interface.read_imu_data()  # List of 3 quaternions [w,x,y,z]

        # Extract only the 3 hydraulic actuator commands (indices 0, 1, 4)
        # [scoop, lift, tilt] - we don't care about tracks or center rotation
        hydraulic_commands = [commands[0], commands[1], commands[4]]

        # Store high-rate data
        self.timestamps.append(current_time)
        self.valve_commands.append(hydraulic_commands.copy())
        self.imu_quaternions.append(imu_data if imu_data else [None, None, None])

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
                    pressure_readings = hardware_interface.adc.read_raw()

                    # Extract chamber pressures (6 channels):
                    # Channels 1-6: [lift_retract, lift_extend, tilt_retract, tilt_extend, scoop_extend, scoop_retract]
                    # Channel 7: Pump pressure
                    # Channel 8: SKIP (slew encoder, not a pressure sensor)
                    chamber_data = [
                        pressure_readings.get("LiftBoom retract ps", 0.0),
                        pressure_readings.get("LiftBoom extend ps", 0.0),
                        pressure_readings.get("TiltBoom retract ps", 0.0),
                        pressure_readings.get("TiltBoom extend ps", 0.0),
                        pressure_readings.get("Scoop extend ps", 0.0),
                        pressure_readings.get("Scoop retract ps", 0.0),
                    ]
                    pump_data = pressure_readings.get("Pump ps", 0.0)

                    self.last_pressure_reading = (chamber_data, pump_data)
                    self.last_pressure_sample_time = current_time

                except Exception as e:
                    print(f"Warning: Pressure read failed: {e}")
                    if self.last_pressure_reading is None:
                        self.last_pressure_reading = ([0.0] * 6, 0.0)

            # Store pressure data (held from last reading if not sampled this cycle)
            if self.last_pressure_reading:
                self.chamber_pressures.append(self.last_pressure_reading[0].copy())
                self.pump_pressure.append(self.last_pressure_reading[1])
            else:
                self.chamber_pressures.append([0.0] * 6)
                self.pump_pressure.append(0.0)
        else:
            # Pressure logging disabled - store dummy data
            self.chamber_pressures.append([0.0] * 6)
            self.pump_pressure.append(0.0)

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
            Dict with current IMU angles (deg), loop timing, and data stats
        """
        debug_info = {}

        # Get current IMU angles in degrees
        imu_data = hardware_interface.read_imu_data()
        if imu_data and len(imu_data) == 3:
            boom_pitch = np.degrees(self.quaternion_to_pitch(imu_data[0]))
            arm_pitch = np.degrees(self.quaternion_to_pitch(imu_data[1]))
            bucket_pitch = np.degrees(self.quaternion_to_pitch(imu_data[2]))
            debug_info['imu_angles'] = {
                'boom': boom_pitch,
                'arm': arm_pitch,
                'bucket': bucket_pitch
            }
        else:
            debug_info['imu_angles'] = None

        # Loop performance (following excavator_controller.py pattern)
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
        """Save collected data to disk synchronously.

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

        # Convert lists to numpy arrays
        data_dict = {
            'timestamps': np.array(self.timestamps),
            'valve_commands': np.array(self.valve_commands),
            'chamber_pressures': np.array(self.chamber_pressures),
            'pump_pressure': np.array(self.pump_pressure),
            'sampling_frequency': SAMPLING_FREQUENCY,
            'pressure_sampling_frequency': PRESSURE_SAMPLING_FREQUENCY,
            'collection_date': datetime.now().isoformat(),
            'command_labels': ['scoop', 'lift', 'tilt'],
            'pressure_labels': ['lift_retract', 'lift_extend', 'tilt_retract',
                               'tilt_extend', 'scoop_extend', 'scoop_retract'],
        }

        # Process IMU data (handle potential None values)
        imu_array = []
        for sample in self.imu_quaternions:
            if sample and all(q is not None for q in sample):
                imu_array.append([q for q in sample])
            else:
                # Fill with identity quaternions if data missing
                imu_array.append([
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0, 0.0])
                ])
        data_dict['imu_quaternions'] = np.array(imu_array)

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"manual_drive_{timestamp_str}.npz"

        # Save (this is the slow part - blocking)
        np.savez_compressed(filename, **data_dict)

        save_time = time.perf_counter() - save_start

        # Print statistics
        print(f"\nSaved {len(self.timestamps)} samples in {save_time:.1f}s")
        self.print_statistics(data_dict)
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

    def print_statistics(self, data):
        """Print data collection statistics."""
        duration_min = data['timestamps'][-1] / 60.0
        num_samples = len(data['timestamps'])

        print(f"\n  === DATA STATISTICS ===")
        print(f"  Duration: {duration_min:.2f} minutes ({data['timestamps'][-1]:.1f} seconds)")
        print(f"  Samples: {num_samples}")
        print(f"  Actual sampling rate: {num_samples / data['timestamps'][-1]:.1f} Hz")

        # Command statistics (3 hydraulic actuators)
        commands = data['valve_commands']
        print(f"\n  Valve Commands (3 hydraulic actuators: scoop, lift, tilt):")
        print(f"    Shape: {commands.shape}")
        print(f"    Range: [{commands.min():.2f}, {commands.max():.2f}]")
        print(f"    Mean: {commands.mean():.2f}")
        for i, label in enumerate(['Scoop', 'Lift', 'Tilt']):
            ch_data = commands[:, i]
            active_pct = (np.abs(ch_data) > 0.1).sum() / len(ch_data) * 100
            print(f"      {label}: active {active_pct:.1f}% of time")

        # IMU statistics
        imu = data['imu_quaternions']
        print(f"\n  IMU Quaternions (3 IMUs: boom, arm, bucket):")
        print(f"    Shape: {imu.shape}")

        # Pressure statistics
        pressures = data['chamber_pressures']
        print(f"\n  Chamber Pressures (6 channels):")
        print(f"    Shape: {pressures.shape}")
        print(f"    Range: [{pressures.min():.3f}, {pressures.max():.3f}] V")
        print(f"    Mean: {pressures.mean():.3f} V")

        pump = data['pump_pressure']
        print(f"\n  Pump Pressure:")
        print(f"    Range: [{pump.min():.3f}, {pump.max():.3f}] V")
        print(f"    Mean: {pump.mean():.3f} V")


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

            # Build name-based command mapping (students-friendly, no indexing)
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

                # IMU angles
                if debug['imu_angles']:
                    angles = debug['imu_angles']
                    debug_str += f"IMUs: Boom={angles['boom']:+6.1f}° Arm={angles['arm']:+6.1f}° Bucket={angles['bucket']:+6.1f}° | "

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
