"""
Robot-side PID tuner (100 Hz) for real-time PID gain and setpoint tuning.

Uses:
- modules.PCA9685_controller.PWMController via modules.hardware_interface.HardwareInterface
- modules.pid.PIDController
- modules.udp_socket.UDPSocket for UDP comms (normalized floats)

Protocol (UDPSocket, normalized to [-1, 1]):
- Client -> Robot (num_inputs = 5): [setpoint_norm, kp_norm, ki_norm, kd_norm, joint_norm]
  - setpoint_norm: setpoint angle in radians scaled by pi (angle_rad/pi)
  - kp_norm, ki_norm, kd_norm: map to ranges via ranges below
  - joint_norm: maps to joint id {0..3} via round((n+1)/2*3)

- Robot -> Client (num_outputs = 2): [measured_angle_norm, pwm_norm]
  - measured_angle_norm: measured angle_rad / pi
  - pwm_norm: last PWM command in [-1, 1]

Joint mapping:
  0: 'slew'   (measured from ADC encoder)
  1: 'boom'   (measured from IMU[0] Y-axis angle)
  2: 'arm'    (measured from IMU[1] Y-axis angle)
  3: 'bucket' (measured from IMU[2] Y-axis angle)

Note: IMU-based angles assume primary rotation about the Y axis.
"""

import math
import numpy as np
import time
from typing import Optional, Tuple

from modules.pid import PIDController
from modules.udp_socket import UDPSocket
from modules.quaternion_math import y_deg_from_quat
from modules.diff_ik import (
    create_excavator_config,
    apply_imu_offsets,
    project_to_rotation_axes,
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
    # Map [-pi, pi] (and beyond clamped) to [-1, 1]
    return max(-1.0, min(1.0, angle_rad / math.pi))


def norm_to_rad(n: float) -> float:
    return max(-1.0, min(1.0, n)) * math.pi


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

        # UDP setup (server)
        self.sock = UDPSocket(local_id=2, max_age_seconds=0.5)
        # Expect 5 inputs, send 2 outputs
        self.sock.setup(host=host, port=port, num_inputs=5, num_outputs=2, is_server=True)
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
        # Returns angle in radians if available, else None
        if joint_id == 0:
            # Slew from ADC encoder (already calibrated radians)
            return float(self.hw.read_slew_angle())

        # IMU-based angles with mounting offsets and axis projection
        imu_quats = self.hw.read_imu_data()
        slew_quat = self.hw.read_slew_quaternion()
        if not imu_quats or len(imu_quats) < 3:
            return None

        # Combine into 4-joint quaternion array [slew, boom, arm, bucket]
        all_quats = [slew_quat] + imu_quats[:3]

        try:
            corrected = apply_imu_offsets(np.array(all_quats, dtype=np.float32), self.robot_config)
            # Project to rotation axes to isolate each joint's principal rotation
            projected = project_to_rotation_axes(corrected, self.robot_config)

            if joint_id == 0:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            angle = float(extract_axis_rotation(projected[joint_id], axis))
            return angle
        except Exception:
            # Fallback to simple Y extraction if anything fails
            imu_idx = max(0, min(2, joint_id - 1))
            y_deg = y_deg_from_quat(imu_quats[imu_idx])
            return math.radians(y_deg)

    def _apply_pwm(self, joint_id: int, pwm: float) -> bool:
        name = self.joint_names[joint_id]
        # Explicitly zero all other channels using controller's zeroing behavior
        return self.hw.send_named_pwm_commands(
            {name: float(max(-1.0, min(1.0, pwm)))}, unset_to_zero=True
        )

    def _update_tuning_from_inputs(self, inputs) -> None:
        # inputs: [setpoint_norm, kp_norm, ki_norm, kd_norm, joint_norm]
        if inputs is None or len(inputs) < 5:
            return

        setpoint_n, kp_n, ki_n, kd_n, joint_n = inputs[:5]

        self.active_joint_id = joint_norm_to_id(joint_n)
        # Update setpoint for active joint only (explicit from client)
        self.setpoints_rad[self.active_joint_id] = norm_to_rad(setpoint_n)

        # Update PID gains for active joint only
        pid = self.pid_controllers[self.active_joint_id]
        pid.kp = norm_to_range(kp_n, *KP_RANGE)
        pid.ki = norm_to_range(ki_n, *KI_RANGE)
        pid.kd = norm_to_range(kd_n, *KD_RANGE)

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

                    # Send feedback to client
                    self.sock.send_floats([
                        rad_to_norm(angle),
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
                        self.hw.send_named_pwm_commands({}, unset_to_zero=True)
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
            print("Stopping PID tuner...")
        finally:
            try:
                self.hw.reset(reset_pump=False)
            except Exception:
                pass


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
