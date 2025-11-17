"""
Robot-side PID tuner (100 Hz) for real-time PID gain and setpoint tuning.

Uses:
- modules.hardware_interface.HardwareInterface (with background IMU/ADC threads)
- modules.pid.PIDController
- modules.udp_socket.UDPSocket for UDP comms (normalized floats)

Protocol (UDPSocket, angles use 2-byte int16 for better resolution):
- Client -> Robot (num_inputs = 8): [setpoint_hi, setpoint_lo, kp_norm, ki_norm, kd_norm, joint_norm, reload_flag, pump_toggle_flag]
  - setpoint_hi, setpoint_lo: setpoint angle as 2-byte int16, range ±90° (±π/2)
  - kp_norm, ki_norm, kd_norm: map to ranges via ranges below
  - joint_norm: maps to joint id {0..3} via round((n+1)/2*3)
  - reload_flag: 1.0 to trigger config reload (both servo & general), else 0.0
  - pump_toggle_flag: 1.0 to toggle pump state, else 0.0

- Robot -> Client (num_outputs = 3): [angle_hi, angle_lo, pwm_norm]
  - angle_hi, angle_lo: measured angle as 2-byte int16, range ±90° (±π/2)
  - pwm_norm: last PWM command in [-1, 1]

Joint mapping:
  0: 'rotate'     (slew, measured from ADC encoder)
  1: 'lift_boom'  (boom, measured from IMU[0] Y-axis angle)
  2: 'tilt_boom'  (arm, measured from IMU[1] Y-axis angle)
  3: 'scoop'      (bucket, measured from IMU[2] Y-axis angle)

Note: Hardware interface uses exception-based error handling with automatic safety reset.
"""

import math
import numpy as np
import time
from typing import Optional, Tuple

from modules.pid import PIDController
from modules.udp_socket import UDPSocket
from modules.quaternion_math import y_deg_from_quat
from modules.diff_ik_V2 import (
    create_excavator_config,
    apply_imu_offsets,
    project_to_rotation_axes,
    propagate_base_rotation,
    extract_axis_rotation,
)
from modules.hardware_interface import HardwareInterface


# Gain ranges for mapping [-1,1] -> [min, max]
KP_RANGE = (0.0, 20.0)
KI_RANGE = (0.0, 5.0)
KD_RANGE = (0.0, 5.0)


def norm_to_range(n: float, lo: float, hi: float) -> float:
    n = max(-1.0, min(1.0, float(n)))
    return (n + 1.0) * 0.5 * (hi - lo) + lo


def range_to_norm(x: float, lo: float, hi: float) -> float:
    # Inverse of norm_to_range
    if hi <= lo:
        return 0.0
    return ((x - lo) / (hi - lo)) * 2.0 - 1.0


def joint_norm_to_id(n: float) -> int:
    # Map [-1,1] -> {0,1,2,3}
    idx = int(round((max(-1.0, min(1.0, n)) + 1.0) * 0.5 * 3.0))
    return max(0, min(3, idx))


def rad_to_norm(angle_rad: float) -> float:
    # Map [-pi/2, pi/2] (±90°) to [-1, 1] for better resolution
    return max(-1.0, min(1.0, angle_rad / (math.pi / 2.0)))


def norm_to_rad(n: float) -> float:
    return max(-1.0, min(1.0, n)) * (math.pi / 2.0)


def pack_int16(value: float) -> tuple:
    """Pack normalized float [-1,1] into 2 signed bytes for int16 resolution."""
    # Map to int16 range
    i16 = int(max(-32768, min(32767, round(value * 32767))))
    # Split into high and low bytes as signed int8
    hi = (i16 >> 8) & 0xFF
    lo = i16 & 0xFF
    # Convert to signed int8 range
    hi = hi - 256 if hi > 127 else hi
    lo = lo - 256 if lo > 127 else lo
    return (hi, lo)


def unpack_int16(hi: int, lo: int) -> float:
    """Unpack 2 signed bytes back to normalized float [-1,1]."""
    # Convert signed int8 to unsigned
    hi_u = hi + 256 if hi < 0 else hi
    lo_u = lo + 256 if lo < 0 else lo
    # Reconstruct int16
    i16 = (hi_u << 8) | lo_u
    # Convert to signed int16
    if i16 > 32767:
        i16 -= 65536
    # Normalize to [-1, 1]
    return max(-1.0, min(1.0, i16 / 32767.0))


class RobotPIDTuner:
    def __init__(self, host: str = "192.168.0.132", port: int = 8080):
        # Hardware
        self.hw = HardwareInterface()

        # Robot config for IMU offset correction and axis projection
        self.robot_config = create_excavator_config()

        # PID per joint (we keep separate integrators for each)
        # Map logical joints -> PWM channel names from configuration_files/linear_config.yaml
        # 0: rotate (slew), 1: lift_boom (boom), 2: tilt_boom (arm), 3: scoop (bucket)
        self.joint_names = ['rotate', 'lift_boom', 'tilt_boom', 'scoop']
        self.pid_controllers = [PIDController(kp=2.0, ki=0.0, kd=0.0) for _ in self.joint_names]

        # Current tuning state
        self.active_joint_id = 0
        # Per-joint setpoints initialized on first valid measurement to avoid jumps
        self.setpoints_rad = [0.0 for _ in self.joint_names]
        self._setpoint_initialized = [False for _ in self.joint_names]
        self.last_pwm = 0.0

        # Edge detection for one-shot commands
        self._reload_flag_prev = 0.0
        self._pump_toggle_flag_prev = 0.0
        self._flag_threshold = 0.5

        # UDP setup (server)
        self.sock = UDPSocket(local_id=2, max_age_seconds=0.5)
        # Expect 8 inputs (setpoint as 2 bytes + 4 single bytes + 2 flags), send 3 outputs (angle as 2 bytes + pwm)
        self.sock.setup(host=host, port=port, num_inputs=8, num_outputs=3, is_server=True)
        print("Waiting for client handshake...")
        if not self.sock.handshake(timeout=10.0):
            print("Handshake failed; continuing to wait for data anyway.")
        self.sock.start_receiving()

        # Debug mapping and available PWM channels
        try:
            available = self.hw.get_pwm_channel_names(include_pump=False)
            print(f"PWM channels available: {available}")
        except Exception as e:
            print(f"Could not read PWM channel names: {e}")

        print(f"Joint mapping: 0->'{self.joint_names[0]}', 1->'{self.joint_names[1]}', 2->'{self.joint_names[2]}', 3->'{self.joint_names[3]}'")

        # Debug cadence
        self._loop_count = 0
        self._debug_every = 10  # print every N cycles (at 100Hz -> ~10 Hz)

    def _read_joint_angle(self, joint_id: int) -> Optional[float]:
        """Read joint angle in radians using full quaternion processing pipeline.

        Args:
            joint_id: Joint index (0=slew, 1=boom, 2=arm, 3=bucket)

        Returns:
            Joint angle in radians, or None if sensors not ready
        """
        try:
            # Read IMU data and slew quaternion (may raise on error)
            imu_quats = self.hw.read_imu_data()
            slew_quat = self.hw.read_slew_quaternion()
            if not imu_quats or len(imu_quats) < 3:
                return None

            # Combine into 4-joint quaternion array [slew, boom, arm, bucket]
            all_quats = np.array([slew_quat] + imu_quats[:3], dtype=np.float32)

            # Apply full quaternion processing pipeline
            corrected = apply_imu_offsets(all_quats, self.robot_config)
            projected = project_to_rotation_axes(corrected, self.robot_config.rotation_axes)
            propagated = propagate_base_rotation(projected, self.robot_config)

            # Extract angle about this joint's rotation axis
            axis = self.robot_config.rotation_axes[joint_id]
            angle = float(extract_axis_rotation(propagated[joint_id], axis))
            return angle
        except Exception as e:
            # Hardware errors (sensors not ready, etc.) are caught and logged
            # Safety decorator has already reset hardware if needed
            if self._loop_count % 100 == 0:  # Log occasionally to avoid spam
                print(f"Sensor read error: {e}")
            return None

    def _apply_pwm(self, joint_id: int, pwm: float) -> bool:
        name = self.joint_names[joint_id]
        # Explicitly zero all other channels using controller's zeroing behavior
        return self.hw.send_named_pwm_commands(
            {name: float(max(-1.0, min(1.0, pwm)))}, unset_to_zero=True
        )

    def _update_tuning_from_inputs(self, inputs) -> None:
        # inputs: [setpoint_hi, setpoint_lo, kp_norm, ki_norm, kd_norm, joint_norm, reload_flag, pump_toggle_flag]
        if inputs is None or len(inputs) < 6:
            return

        setpoint_hi, setpoint_lo, kp_n, ki_n, kd_n, joint_n = inputs[:6]

        self.active_joint_id = joint_norm_to_id(joint_n)
        # Convert floats back to int8 bytes, then unpack 2-byte setpoint angle
        setpoint_hi_int = int(round(setpoint_hi * 127.0))
        setpoint_lo_int = int(round(setpoint_lo * 127.0))
        setpoint_n = unpack_int16(setpoint_hi_int, setpoint_lo_int)
        # Update setpoint for active joint only (explicit from client)
        self.setpoints_rad[self.active_joint_id] = norm_to_rad(setpoint_n)

        # Update PID gains for active joint only
        pid = self.pid_controllers[self.active_joint_id]
        pid.kp = norm_to_range(kp_n, *KP_RANGE)
        pid.ki = norm_to_range(ki_n, *KI_RANGE)
        pid.kd = norm_to_range(kd_n, *KD_RANGE)

        # Handle one-shot commands (if provided) with edge detection
        if len(inputs) >= 8:
            reload_flag = inputs[6]
            pump_toggle_flag = inputs[7]

            # Reload config if requested (rising edge only)
            if reload_flag > self._flag_threshold and self._reload_flag_prev <= self._flag_threshold:
                try:
                    print("Reload config requested by client...")
                    # Try both servo and general config
                    servo_ok = self.hw.reload_config()
                    general_ok = self.hw.reload_general_config()
                    if servo_ok or general_ok:
                        print(f"Config reloaded (servo={servo_ok}, general={general_ok})")
                    else:
                        print("Config reload: no changes")
                except Exception as e:
                    print(f"Config reload failed: {e}")

            # Toggle pump if requested (rising edge only)
            if pump_toggle_flag > self._flag_threshold and self._pump_toggle_flag_prev <= self._flag_threshold:
                try:
                    print("Pump toggle requested by client...")
                    if self.hw.pwm_controller:
                        self.hw.pwm_controller.set_pump(not self.hw.pwm_controller.pump_enabled)
                        print(f"Pump {'ON' if self.hw.pwm_controller.pump_enabled else 'OFF'}.")
                    else:
                        print("Pump toggle failed: PWM controller not available")
                except Exception as e:
                    print(f"Pump toggle failed: {e}")

            # Update previous flag states
            self._reload_flag_prev = reload_flag
            self._pump_toggle_flag_prev = pump_toggle_flag

    def run(self):
        # 100 Hz control loop
        period = 1.0 / 100.0
        next_t = time.perf_counter()
        print("PID tuner running at 100 Hz. Press Ctrl+C to stop.")

        try:
            while True:
                # Update tuning from latest client inputs (if fresh)
                latest = self.sock.get_latest_floats()
                if latest is not None:
                    self._update_tuning_from_inputs(latest)

                # Measure angle
                angle = self._read_joint_angle(self.active_joint_id)

                # Compute PWM via PID (if measurement available)
                if angle is not None:
                    # Initialize setpoint on first valid measurement to avoid jumps
                    if not self._setpoint_initialized[self.active_joint_id]:
                        self.setpoints_rad[self.active_joint_id] = angle
                        self.pid_controllers[self.active_joint_id].reset()
                        self._setpoint_initialized[self.active_joint_id] = True

                    pid = self.pid_controllers[self.active_joint_id]
                    pwm = pid.compute(self.setpoints_rad[self.active_joint_id], angle)
                    self.last_pwm = pwm
                    send_ok = self._apply_pwm(self.active_joint_id, pwm)

                    # Send feedback to client: pack angle as 2 bytes
                    angle_norm = rad_to_norm(angle)
                    angle_hi, angle_lo = pack_int16(angle_norm)
                    self.sock.send_floats([
                        angle_hi / 127.0,  # Convert back to normalized for UDPSocket
                        angle_lo / 127.0,
                        float(max(-1.0, min(1.0, pwm)))
                    ])

                    # Periodic debug print
                    self._loop_count += 1
                    if (self._loop_count % self._debug_every) == 0:
                        pid = self.pid_controllers[self.active_joint_id]
                        print(
                            f"joint={self.active_joint_id}('{self.joint_names[self.active_joint_id]}') "
                            f"angle={angle:+.3f}rad setpoint={self.setpoints_rad[self.active_joint_id]:+.3f}rad "
                            f"kp={pid.kp:.3f} ki={pid.ki:.3f} kd={pid.kd:.3f} pwm={pwm:+.3f} "
                            f"pwm_ready={self.hw.pwm_ready} send_ok={send_ok}"
                        )
                else:
                    # No measurement; drive all channels to zero for safety
                    try:
                        self.hw.reset(reset_pump=False)
                    except Exception:
                        pass

                # Sleep to maintain loop rate
                next_t += period
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    # Missed deadline; reset next_t to avoid drift
                    next_t = time.perf_counter()

        except KeyboardInterrupt:
            print("\nStopping PID tuner...")
        finally:
            # Clean shutdown: reset PWM, stop background threads, close serial
            try:
                print("Resetting hardware...")
                self.hw.reset(reset_pump=True)
                time.sleep(0.1)  # Allow reset command to complete
                print("Shutting down background threads...")
                self.hw.shutdown()
            except Exception as e:
                print(f"Shutdown error: {e}")
            print("PID tuner stopped.")


if __name__ == "__main__":
    import sys
    host = "192.168.0.132"
    port = 8080
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            port = int(sys.argv[2])
        except ValueError:
            pass
    RobotPIDTuner(host=host, port=port).run()
