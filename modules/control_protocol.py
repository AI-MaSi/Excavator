import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Sequence, Tuple


MAGIC = b"EXCV"
PROTOCOL_VERSION = 1


class MessageType(IntEnum):
    COMMAND = 1
    TELEMETRY = 2


class ControlMode(IntEnum):
    IK = 1
    DIRECT = 2


class CommandFlags(IntEnum):
    RELOAD = 1 << 0
    PAUSE = 1 << 1
    RESUME = 1 << 2


class TelemetryFlags(IntEnum):
    PAUSED = 1 << 0
    DIRECT_MODE = 1 << 1
    HW_READY = 1 << 2
    SLEW_FUSION_ENABLED = 1 << 3
    SLEW_FUSION_ACTIVE = 1 << 4


COMMAND_STRUCT = struct.Struct("<4sBBBBII8f")
TELEMETRY_STRUCT = struct.Struct("<4sBBBBII13f15f")

COMMAND_PACKET_SIZE = COMMAND_STRUCT.size
TELEMETRY_PACKET_SIZE = TELEMETRY_STRUCT.size


def _now_ms() -> int:
    return int(time.time() * 1000) & 0xFFFFFFFF


def _bytes_to_signed_list(payload: bytes) -> List[int]:
    return [b if b < 128 else b - 256 for b in payload]


def _signed_list_to_bytes(values: Sequence[int], expected_size: int) -> bytes:
    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes, got {len(values)}")
    return bytes((v if v >= 0 else v + 256) & 0xFF for v in values)


@dataclass
class PoseTarget:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rot_y_deg: float = 0.0


@dataclass
class DirectCommand:
    slew: float = 0.0
    boom: float = 0.0
    arm: float = 0.0
    bucket: float = 0.0


@dataclass
class ControlCommand:
    sequence: int = 0
    timestamp_ms: int = 0
    mode: ControlMode = ControlMode.IK
    pose: PoseTarget = field(default_factory=PoseTarget)
    direct: DirectCommand = field(default_factory=DirectCommand)
    reload_config: bool = False
    pause: bool = False
    resume: bool = False

    def flags(self) -> int:
        flags = 0
        if self.reload_config:
            flags |= int(CommandFlags.RELOAD)
        if self.pause:
            flags |= int(CommandFlags.PAUSE)
        if self.resume:
            flags |= int(CommandFlags.RESUME)
        return flags


@dataclass
class RobotTelemetry:
    sequence: int = 0
    timestamp_ms: int = 0
    mode: ControlMode = ControlMode.IK
    paused: bool = False
    hardware_ready: bool = False
    slew_fusion_enabled: bool = False
    slew_fusion_active: bool = False
    measured_pose: PoseTarget = field(default_factory=PoseTarget)
    target_pose: PoseTarget = field(default_factory=PoseTarget)
    joint_angles_deg: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    joint_positions: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    slew_fusion_gyro_z_degps: float = 0.0

    def flags(self) -> int:
        flags = 0
        if self.paused:
            flags |= int(TelemetryFlags.PAUSED)
        if self.mode == ControlMode.DIRECT:
            flags |= int(TelemetryFlags.DIRECT_MODE)
        if self.hardware_ready:
            flags |= int(TelemetryFlags.HW_READY)
        if self.slew_fusion_enabled:
            flags |= int(TelemetryFlags.SLEW_FUSION_ENABLED)
        if self.slew_fusion_active:
            flags |= int(TelemetryFlags.SLEW_FUSION_ACTIVE)
        return flags


def encode_command_message(command: ControlCommand) -> List[int]:
    payload = COMMAND_STRUCT.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(MessageType.COMMAND),
        int(command.mode),
        command.flags(),
        int(command.sequence) & 0xFFFFFFFF,
        int(command.timestamp_ms or _now_ms()) & 0xFFFFFFFF,
        float(command.pose.x),
        float(command.pose.y),
        float(command.pose.z),
        float(command.pose.rot_y_deg),
        float(command.direct.slew),
        float(command.direct.boom),
        float(command.direct.arm),
        float(command.direct.bucket),
    )
    return _bytes_to_signed_list(payload)


def decode_command_message(values: Sequence[int]) -> ControlCommand:
    payload = _signed_list_to_bytes(values, COMMAND_PACKET_SIZE)
    magic, version, msg_type, mode, flags, seq, timestamp_ms, *floats = COMMAND_STRUCT.unpack(payload)
    if magic != MAGIC:
        raise ValueError("Invalid command packet magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported command packet version {version}")
    if msg_type != int(MessageType.COMMAND):
        raise ValueError(f"Unexpected command packet type {msg_type}")

    pose = PoseTarget(*floats[0:4])
    direct = DirectCommand(*floats[4:8])
    return ControlCommand(
        sequence=int(seq),
        timestamp_ms=int(timestamp_ms),
        mode=ControlMode(mode),
        pose=pose,
        direct=direct,
        reload_config=bool(flags & int(CommandFlags.RELOAD)),
        pause=bool(flags & int(CommandFlags.PAUSE)),
        resume=bool(flags & int(CommandFlags.RESUME)),
    )


def encode_telemetry_message(telemetry: RobotTelemetry) -> List[int]:
    flat_positions: List[float] = []
    positions = telemetry.joint_positions[:5]
    for pos in positions:
        flat_positions.extend((float(pos[0]), float(pos[1]), float(pos[2])))
    while len(flat_positions) < 15:
        flat_positions.extend((0.0, 0.0, 0.0))

    payload = TELEMETRY_STRUCT.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(MessageType.TELEMETRY),
        int(telemetry.mode),
        telemetry.flags(),
        int(telemetry.sequence) & 0xFFFFFFFF,
        int(telemetry.timestamp_ms or _now_ms()) & 0xFFFFFFFF,
        float(telemetry.measured_pose.x),
        float(telemetry.measured_pose.y),
        float(telemetry.measured_pose.z),
        float(telemetry.measured_pose.rot_y_deg),
        float(telemetry.target_pose.x),
        float(telemetry.target_pose.y),
        float(telemetry.target_pose.z),
        float(telemetry.target_pose.rot_y_deg),
        float(telemetry.joint_angles_deg[0]),
        float(telemetry.joint_angles_deg[1]),
        float(telemetry.joint_angles_deg[2]),
        float(telemetry.joint_angles_deg[3]),
        float(telemetry.slew_fusion_gyro_z_degps),
        *flat_positions,
    )
    return _bytes_to_signed_list(payload)


def decode_telemetry_message(values: Sequence[int]) -> RobotTelemetry:
    payload = _signed_list_to_bytes(values, TELEMETRY_PACKET_SIZE)
    unpacked = TELEMETRY_STRUCT.unpack(payload)
    magic, version, msg_type, mode, flags, seq, timestamp_ms = unpacked[:7]
    if magic != MAGIC:
        raise ValueError("Invalid telemetry packet magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported telemetry packet version {version}")
    if msg_type != int(MessageType.TELEMETRY):
        raise ValueError(f"Unexpected telemetry packet type {msg_type}")

    floats = unpacked[7:]
    measured_pose = PoseTarget(*floats[0:4])
    target_pose = PoseTarget(*floats[4:8])
    joint_angles = tuple(float(v) for v in floats[8:12])
    slew_gyro_z = float(floats[12])
    pos_vals = floats[13:28]
    positions = []
    for idx in range(0, len(pos_vals), 3):
        positions.append((float(pos_vals[idx]), float(pos_vals[idx + 1]), float(pos_vals[idx + 2])))

    return RobotTelemetry(
        sequence=int(seq),
        timestamp_ms=int(timestamp_ms),
        mode=ControlMode(mode),
        paused=bool(flags & int(TelemetryFlags.PAUSED)),
        hardware_ready=bool(flags & int(TelemetryFlags.HW_READY)),
        slew_fusion_enabled=bool(flags & int(TelemetryFlags.SLEW_FUSION_ENABLED)),
        slew_fusion_active=bool(flags & int(TelemetryFlags.SLEW_FUSION_ACTIVE)),
        measured_pose=measured_pose,
        target_pose=target_pose,
        joint_angles_deg=joint_angles,  # type: ignore[arg-type]
        joint_positions=tuple(positions[:5]),
        slew_fusion_gyro_z_degps=slew_gyro_z,
    )
