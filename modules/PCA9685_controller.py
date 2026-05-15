"""
Valve testing PWM controller with:
- Derived PWM period from actual PCA9685 frequency
- Simple deadband: compresses command range to skip dead zone (linear throughout!)
- Optional dither (per-channel) to prevent valve stiction
- Optional per-channel ramp/slew limiting to smooth step inputs

Maintains linearity by packing commands into the working range around the dead zone.
"""

import atexit
import threading
import time
import logging
import yaml
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path

from .perf_tracker import LoopPerfTracker


PCA9685_I2C_BUS = 1
PCA9685_I2C_ADDRESS = 0x40


def _open_smbus(bus_number: int = PCA9685_I2C_BUS):
    """Open Raspberry Pi I2C via smbus2/smbus for low-overhead direct writes."""
    try:
        import smbus2
        return smbus2.SMBus(bus_number)
    except Exception:
        import smbus as smbus2  # type: ignore
        return smbus2.SMBus(bus_number)


class PWMControllerError(Exception):
    """Base exception for PWM controller failures."""


class PWMConfigError(PWMControllerError):
    """Base exception for PWM configuration failures."""


class PWMConfigLoadError(PWMConfigError):
    """Raised when a PWM config file cannot be loaded or parsed."""


class PWMConfigValidationError(PWMConfigError):
    """Raised when a PWM config file is syntactically valid but invalid semantically."""


class PWMHardwareIOError(PWMControllerError):
    """Raised when a low-level PWM hardware operation fails."""


# ============================================================================
# Direct I2C PWM Writer
# ============================================================================

class DirectPWMWriter:
    """Direct I2C writer for PCA9685 using pre-allocated buffer.

    Writes only configured channels in a single I2C transaction, eliminating
    per-channel allocation and lock overhead from the Adafruit library.

    """

    LED0_ON_L = 0x06  # First PWM register (channel 0)
    MODE1 = 0x00
    MODE2 = 0x01
    PRE_SCALE = 0xFE

    # MODE1 bits
    SLEEP = 0x10
    RESTART = 0x80
    AI = 0x20  # Auto-increment

    def __init__(self, i2c_bus, address: int = PCA9685_I2C_ADDRESS,
                 min_channel: int = 0, max_channel: int = 15,
                 frequency: int = 200):
        """Initialize the direct PWM writer.

        Args:
            i2c_bus: The I2C bus instance
            address: PCA9685 I2C address (default 0x40)
            min_channel: Lowest channel number to write (0-15)
            max_channel: Highest channel number to write (0-15)
            frequency: PWM frequency in Hz (default 200)
        """
        if not (0 <= min_channel <= max_channel < 16):
            raise ValueError(
                f"Invalid PWM channel range [{min_channel}, {max_channel}] - expected 0 <= min <= max < 16"
            )
        self._i2c = i2c_bus
        self._addr = address
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1
        self._lock = threading.Lock()
        self._uses_busio = all(hasattr(i2c_bus, name) for name in ("try_lock", "writeto", "unlock"))
        self._uses_smbus = all(hasattr(i2c_bus, name) for name in ("write_byte_data", "write_i2c_block_data"))
        if not (self._uses_busio or self._uses_smbus):
            raise TypeError(
                "i2c_bus must be either busio.I2C-like (try_lock/writeto/unlock) "
                "or SMBus-like (write_byte_data/write_i2c_block_data)"
            )

        # Pre-allocated buffer: 1 byte register addr + 4 bytes per channel
        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)  # Start register
        self._duty_cycles = [0] * 16  # Full array; only min_ch..max_ch are written

        self._init_chip(frequency)

    def _write_reg(self, reg: int, value: int):
        """Write a single byte to a register."""
        try:
            if self._uses_smbus:
                with self._lock:
                    self._i2c.write_byte_data(self._addr, reg, value)
                return

            while not self._i2c.try_lock():
                time.sleep(0)  # Yield to scheduler to avoid busy-waiting on I2C lock.
            try:
                self._i2c.writeto(self._addr, bytes([reg, value]))
            finally:
                self._i2c.unlock()
        except Exception as exc:
            raise PWMHardwareIOError(
                f"Failed to write PCA9685 register 0x{reg:02X} at address 0x{self._addr:02X}: {exc}"
            ) from exc

    def _init_chip(self, frequency: int):
        """Initialize PCA9685 and set PWM frequency."""
        # Reset - sleep mode, auto-increment enabled
        self._write_reg(self.MODE1, self.SLEEP | self.AI)
        self._write_reg(self.MODE2, 0x04)  # OUTDRV = totem pole

        # Set frequency (prescale = 25MHz / (4096 * freq) - 1)
        prescale = int(25_000_000 / (4096 * frequency) - 1)
        prescale = max(3, min(255, prescale))  # Valid range: 3-255
        self._write_reg(self.PRE_SCALE, prescale)

        # Wake up
        self._write_reg(self.MODE1, self.AI)  # Clear sleep
        time.sleep(0.005)  # Wait for oscillator
        self._write_reg(self.MODE1, self.AI | self.RESTART)

        self.frequency = frequency

    def sleep(self):
        """Put chip to sleep - stops oscillator, outputs go low."""
        self._write_reg(self.MODE1, self.SLEEP | self.AI)

    def wake(self):
        """Wake chip from sleep."""
        self._write_reg(self.MODE1, self.AI)
        time.sleep(0.005)
        self._write_reg(self.MODE1, self.AI | self.RESTART)

    def set_channel(self, channel: int, duty_cycle: int):
        """Queue a duty cycle update (0-65535).

        Note: Only channels within [min_channel, max_channel] are written by flush().
        """
        if not 0 <= channel < 16:
            raise ValueError(f"PWM channel {channel} out of range [0, 15]")
        self._duty_cycles[channel] = duty_cycle

    def set_channel_range(self, min_channel: int, max_channel: int):
        """Update the channel range. For benchmarking only - allocates memory."""
        if not (0 <= min_channel <= max_channel < 16):
            raise ValueError(
                f"Invalid PWM channel range [{min_channel}, {max_channel}] - expected 0 <= min <= max < 16"
            )
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1
        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)

    def flush(self):
        """Write configured channels in a single I2C transaction."""
        buf = self._buf
        duty_cycles = self._duty_cycles
        min_ch = self._min_ch

        for i in range(self._num_channels):
            duty = duty_cycles[min_ch + i]
            # Convert 16-bit duty to PCA9685 ON/OFF register values
            if duty >= 0xFFF0:
                # Fully on: set bit 4 of ON_H
                on_val, off_val = 0x1000, 0
            elif duty < 0x0010:
                # Fully off: set bit 4 of OFF_H
                on_val, off_val = 0, 0x1000
            else:
                # Normal PWM: ON at 0, OFF at duty>>4 (12-bit)
                on_val = 0
                off_val = duty >> 4

            offset = 1 + i * 4
            buf[offset] = on_val & 0xFF
            buf[offset + 1] = (on_val >> 8) & 0xFF
            buf[offset + 2] = off_val & 0xFF
            buf[offset + 3] = (off_val >> 8) & 0xFF

        try:
            if self._uses_smbus:
                with self._lock:
                    data = buf[1:]
                    # SMBus block writes are limited to 32 data bytes. Keep
                    # chunks channel-aligned (8 channels * 4 bytes).
                    for offset in range(0, len(data), 32):
                        self._i2c.write_i2c_block_data(
                            self._addr,
                            buf[0] + offset,
                            list(data[offset:offset + 32]),
                        )
                return

            while not self._i2c.try_lock():
                time.sleep(0)  # Yield to scheduler to avoid busy-waiting on I2C lock.
            try:
                self._i2c.writeto(self._addr, buf)
            finally:
                self._i2c.unlock()
        except Exception as exc:
            raise PWMHardwareIOError(
                f"Failed to flush PCA9685 channel buffer at address 0x{self._addr:02X}: {exc}"
            ) from exc


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class ChannelConfig:
    """Configuration for a single PWM channel."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    direction: int  # +1 or -1; maps sign of input to physical pulse side
    center: Optional[float] = None
    deadzone: float = 0.0  # percent of full-scale input mapped to zero (rarely used here)
    affects_pump: bool = False
    toggleable: bool = False

    # Simple deadband (separate per sign) - jumps over dead zone by compressing command range
    # Offsets from center (us) where the working range starts for positive/negative inputs
    deadband_us_pos: float = 0.0
    deadband_us_neg: float = 0.0

    # Dither settings to prevent valve stiction
    dither_enable: bool = False
    dither_amp_us: float = 8.0  # vibration amplitude in microseconds
    dither_hz: float = 40.0  # vibration frequency
    dither_taper: bool = False  # fade dither in after the deadband edge
    dither_taper_us: float = 8.0  # distance after deadband edge before full dither

    # Slew rate limiting (microseconds per second) to soften command steps
    ramp_enable: bool = False
    ramp_limit: float = 0.0  # us/s; ignored when ramp_enable is False
    ramp_skip_deadband: bool = False  # if True, deadband jump is instant (slew only applies to usable range)

    # Symmetric gamma shaping (1.0 = linear). Applied to magnitude for both directions.
    gamma: float = 1.0

    def __post_init__(self):
        self.pulse_range = self.pulse_max - self.pulse_min
        if self.center is None:
            self.center = self.pulse_min + (self.pulse_range / 2)
        self.deadzone_threshold = self.deadzone / 100.0
        self._dither_omega = 2.0 * math.pi * float(self.dither_hz)
        self._dither_phase_offset = self.output_channel * 1.0471975512


@dataclass
class PumpConfig:
    """Configuration specific to pump control (not typically used in valve_testing)."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    static_pulse_us: float   # fixed pulse width (µs) used when auto mode is off
    base_pulse_us: float     # idle pulse (µs) in auto mode with no valve activity
    activity_gain_us: float  # extra µs added at full average valve activity (auto mode)


class PWMConstants:
    """Hardware and timing constants."""
    PWM_FREQUENCY_DEFAULT = 50  # Standard servo frequency
    MAX_CHANNELS = 16
    DUTY_CYCLE_MAX = 65535
    RAMP_DT_MAX = 0.05

    # Validation limits
    PULSE_MIN = 0
    PULSE_MAX = 4095
    PWM_FREQ_MIN = 30
    PWM_FREQ_MAX = 1000
    NORMALIZED_COMMAND_MIN = -1.0
    NORMALIZED_COMMAND_MAX = 1.0

    # Safety parameters
    DEFAULT_TIME_WINDOW = 0.5  # seconds; short enough to catch a stalled loop quickly


class PWMController:
    """Simple PWM controller with piecewise deadband and dither for valve testing."""

    def __init__(self, config_file: str, pump_auto_mode: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0,
                 default_unset_to_zero: bool = True, log_level: str = "INFO",
                 stale_timeout_s: float = 0.0, perf_enabled: bool = False,
                 cleanup_disable_osc: bool = True, pwm_frequency: Optional[int] = None):
        """Initialize PWM controller.

        Args:
            config_file: Path to YAML configuration file
            pump_auto_mode: Enable auto pump speed (scales with valve activity).
            toggle_channels: Enable toggleable channels
            input_rate_threshold: Input rate threshold for safety monitoring
            default_unset_to_zero: Default unset channels to zero
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            stale_timeout_s: Timeout for stale commands (0 = disabled)
            perf_enabled: Enable performance tracking (loop time, jitter, headroom)
            cleanup_disable_osc: If True, stop PCA9685 oscillator on cleanup (outputs go LOW).
                                 If False, keep oscillator running (outputs stay at center).
            pwm_frequency: PWM frequency in Hz (default: from PWMConstants.PWM_FREQUENCY_DEFAULT)
        """
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.PWMController")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self.pump_auto_mode = bool(pump_auto_mode)
        self.toggle_channels = toggle_channels
        self.pump_enabled = True
        self._pump_direct_us: Optional[float] = None  # set by set_pump_speed_us; None = auto/static
        self.pump_activity_sum = 0.0
        self.pump_activity_count = 0
        self._stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._last_command_ts = time.monotonic()

        # Rate monitoring (optional)
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking
        self.time_window = PWMConstants.DEFAULT_TIME_WINDOW

        # Threads/monitoring
        self.running = False
        self.monitor_thread = None
        self.last_input_time = time.time()

        # Load config
        self._load_config(config_file)

        # Hardware init - direct SMBus I2C, no Adafruit dependency
        self._i2c = _open_smbus(PCA9685_I2C_BUS)
        self._direct_writer = None
        self._pwm_period_us = 0.0
        self._rebuild_writer(pwm_frequency=pwm_frequency)

        # Current normalized values per channel
        self.values = [0.0] * PWMConstants.MAX_CHANNELS

        # Rate monitoring counter — incremented in update_named() under _lock,
        # read and reset by _monitor_loop each window. Avoids event coalescing.
        self._cmd_counter = 0
        self.rate_window_start = time.time()

        # Register simple cleanup and start monitoring
        atexit.register(self._simple_cleanup)
        self.reset()

        if not self.skip_rate_checking or self._stale_timeout_s > 0.0:
            self._start_monitoring()

        # Behavior defaults
        self._default_unset_to_zero = default_unset_to_zero

        # Performance tracking (lightweight, opt-in)
        self._perf_tracker = LoopPerfTracker(enabled=perf_enabled)

        # Cleanup behavior: whether to stop oscillator (outputs go LOW) or keep running (outputs stay at center)
        self._cleanup_disable_osc = cleanup_disable_osc

    def _load_config(self, config_file: str):
        config_path, channel_configs, pump_config, pwm_frequency = self._read_config_data(config_file)
        self._config_path = config_path
        self.channel_configs = channel_configs
        self.pump_config = pump_config
        self._config_pwm_frequency = pwm_frequency

    def _resolve_config_path(self, config_file: str) -> Path:
        path = Path(config_file)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path(__file__).resolve().parent.parent / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise PWMConfigLoadError(f"PWM config file '{config_file}' not found. Searched: {searched}")

    def _read_config_data(self, config_file: str) -> tuple[Path, Dict[str, ChannelConfig], Optional[PumpConfig], Any]:
        config_path = self._resolve_config_path(config_file)

        try:
            with config_path.open('r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise PWMConfigLoadError(f"Failed to parse YAML in '{config_path}': {exc}") from exc
        except OSError as exc:
            raise PWMConfigLoadError(f"Failed to read PWM config '{config_path}': {exc}") from exc

        if raw_config is None:
            raise PWMConfigLoadError(f"PWM config '{config_path}' is empty")
        if not isinstance(raw_config, dict):
            raise PWMConfigLoadError(
                f"PWM config '{config_path}' must contain a top-level mapping, got {type(raw_config).__name__}"
            )
        if 'CHANNEL_CONFIGS' not in raw_config:
            raise PWMConfigValidationError(
                f"PWM config '{config_path}' is missing required top-level key 'CHANNEL_CONFIGS'"
            )
        if not isinstance(raw_config['CHANNEL_CONFIGS'], dict):
            raise PWMConfigValidationError(
                f"PWM config '{config_path}': 'CHANNEL_CONFIGS' must be a mapping"
            )

        channel_configs, pump_config = self._parse_config(raw_config)
        pwm_frequency = raw_config.get('pwm_frequency', None)
        self._validate_config_data(channel_configs, pump_config, pwm_frequency)
        return config_path, channel_configs, pump_config, pwm_frequency

    def _parse_config(self, raw_config: Dict) -> tuple[Dict[str, ChannelConfig], Optional[PumpConfig]]:
        channel_configs: Dict[str, ChannelConfig] = {}
        pump_config = None
        config_errors: List[str] = []

        def _err(scope: str, key: str, msg: str) -> None:
            config_errors.append(f"{scope}: {key} {msg}")

        def _as_int(scope: str, key: str, value: Any) -> Optional[int]:
            if isinstance(value, bool):
                _err(scope, key, "must be int (got bool)")
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            _err(scope, key, f"must be int (got {type(value).__name__})")
            return None

        def _as_float(scope: str, key: str, value: Any) -> Optional[float]:
            if isinstance(value, bool):
                _err(scope, key, "must be number (got bool)")
                return None
            if isinstance(value, (int, float)):
                return float(value)
            _err(scope, key, f"must be number (got {type(value).__name__})")
            return None

        def _as_bool(scope: str, key: str, value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            _err(scope, key, f"must be bool (got {type(value).__name__})")
            return None

        required_pump_keys = [
            'output_channel', 'pulse_min', 'pulse_max', 'static_pulse_us', 'base_pulse_us', 'activity_gain_us'
        ]
        required_channel_keys = [
            'output_channel', 'pulse_min', 'pulse_max', 'direction', 'center',
            'deadzone', 'affects_pump', 'toggleable', 'deadband_us_pos', 'deadband_us_neg',
            'dither_enable', 'dither_amp_us', 'dither_hz', 'dither_taper', 'dither_taper_us',
            'ramp_enable', 'ramp_limit', 'ramp_skip_deadband', 'gamma'
        ]

        for name, cfg in raw_config['CHANNEL_CONFIGS'].items():
            if not isinstance(cfg, dict):
                config_errors.append(
                    f"Entry '{name}' in CHANNEL_CONFIGS must be a mapping (got {type(cfg).__name__})"
                )
                continue

            item_errors_before = len(config_errors)
            if name == 'pump':
                missing = [k for k in required_pump_keys if k not in cfg]
                if missing:
                    config_errors.append(
                        f"Pump config missing required keys: {', '.join(missing)}"
                    )
                    continue
                scope = "Pump"
                output_channel = _as_int(scope, 'output_channel', cfg['output_channel'])
                pulse_min = _as_int(scope, 'pulse_min', cfg['pulse_min'])
                pulse_max = _as_int(scope, 'pulse_max', cfg['pulse_max'])
                static_pulse_us = _as_float(scope, 'static_pulse_us', cfg['static_pulse_us'])
                base_pulse_us = _as_float(scope, 'base_pulse_us', cfg['base_pulse_us'])
                activity_gain_us = _as_float(scope, 'activity_gain_us', cfg['activity_gain_us'])
                if len(config_errors) != item_errors_before:
                    continue
                pump_config = PumpConfig(
                    output_channel=output_channel,
                    pulse_min=pulse_min,
                    pulse_max=pulse_max,
                    static_pulse_us=static_pulse_us,
                    base_pulse_us=base_pulse_us,
                    activity_gain_us=activity_gain_us,
                )
            else:
                missing = [k for k in required_channel_keys if k not in cfg]
                if missing:
                    config_errors.append(
                        f"Channel '{name}' missing required keys: {', '.join(missing)}"
                    )
                    continue
                scope = f"Channel '{name}'"
                output_channel = _as_int(scope, 'output_channel', cfg['output_channel'])
                pulse_min = _as_int(scope, 'pulse_min', cfg['pulse_min'])
                pulse_max = _as_int(scope, 'pulse_max', cfg['pulse_max'])
                direction = _as_int(scope, 'direction', cfg['direction'])
                center = self._normalize_none(cfg['center'])
                if center is not None:
                    center = _as_float(scope, 'center', center)
                deadzone = _as_float(scope, 'deadzone', cfg['deadzone'])
                affects_pump = _as_bool(scope, 'affects_pump', cfg['affects_pump'])
                toggleable = _as_bool(scope, 'toggleable', cfg['toggleable'])
                deadband_us_pos = _as_float(scope, 'deadband_us_pos', cfg['deadband_us_pos'])
                deadband_us_neg = _as_float(scope, 'deadband_us_neg', cfg['deadband_us_neg'])
                dither_enable = _as_bool(scope, 'dither_enable', cfg['dither_enable'])
                dither_amp_us = _as_float(scope, 'dither_amp_us', cfg['dither_amp_us'])
                dither_hz = _as_float(scope, 'dither_hz', cfg['dither_hz'])
                dither_taper = _as_bool(scope, 'dither_taper', cfg['dither_taper'])
                dither_taper_us = _as_float(scope, 'dither_taper_us', cfg['dither_taper_us'])
                ramp_enable = _as_bool(scope, 'ramp_enable', cfg['ramp_enable'])
                ramp_limit = _as_float(scope, 'ramp_limit', cfg['ramp_limit'])
                ramp_skip_deadband = _as_bool(scope, 'ramp_skip_deadband', cfg['ramp_skip_deadband'])
                gamma = _as_float(scope, 'gamma', cfg['gamma'])
                if len(config_errors) != item_errors_before:
                    continue
                channel_configs[name] = ChannelConfig(
                    output_channel=output_channel,
                    pulse_min=pulse_min,
                    pulse_max=pulse_max,
                    direction=direction,
                    center=center,
                    deadzone=deadzone,
                    affects_pump=affects_pump,
                    toggleable=toggleable,
                    # Simple deadband (separate per sign)
                    deadband_us_pos=float(deadband_us_pos),
                    deadband_us_neg=float(deadband_us_neg),
                    # Dither settings - opt-in only
                    dither_enable=dither_enable,
                    dither_amp_us=float(dither_amp_us),
                    dither_hz=float(dither_hz),
                    dither_taper=dither_taper,
                    dither_taper_us=float(dither_taper_us),
                    # Ramp/slew limiting - opt-in per channel
                    ramp_enable=ramp_enable,
                    ramp_limit=float(ramp_limit),
                    ramp_skip_deadband=ramp_skip_deadband,
                    # Symmetric gamma shaping
                    gamma=float(gamma),
                )

        if config_errors:
            raise PWMConfigValidationError("Invalid PWM config:\n- " + "\n- ".join(config_errors))

        return channel_configs, pump_config

    @staticmethod
    def _normalize_none(value: Any) -> Optional[Any]:
        none_values = [None, "None", "none", "null", "NONE", "Null", "NULL", "", "n/a", "N/A"]
        return None if value in none_values else value

    def _validate_config(self):
        self._validate_config_data(self.channel_configs, self.pump_config, self._config_pwm_frequency)

    def _validate_config_data(
        self,
        channel_configs: Dict[str, ChannelConfig],
        pump_config: Optional[PumpConfig],
        pwm_frequency: Any,
    ):
        errors = []
        used_outputs = {}

        # PWM frequency validation
        if pwm_frequency is None:
            errors.append("pwm_frequency: missing from servo config (required)")
        elif isinstance(pwm_frequency, bool) or not isinstance(pwm_frequency, (int, float)):
            errors.append(f"pwm_frequency: must be a number (got {type(pwm_frequency).__name__})")
        else:
            freq = int(pwm_frequency)
            if not PWMConstants.PWM_FREQ_MIN <= freq <= PWMConstants.PWM_FREQ_MAX:
                errors.append(f"pwm_frequency: {freq} Hz out of range "
                              f"({PWMConstants.PWM_FREQ_MIN}-{PWMConstants.PWM_FREQ_MAX} Hz)")

        if not channel_configs and pump_config is None:
            errors.append("CHANNEL_CONFIGS: must define at least one channel or a pump")

        for name, config in channel_configs.items():
            if config.direction not in [-1, 1]:
                errors.append(f"Channel '{name}': direction must be -1 or 1")

            if config.output_channel in used_outputs:
                errors.append(f"Channel '{name}': output {config.output_channel} already used")
            elif not 0 <= config.output_channel < PWMConstants.MAX_CHANNELS:
                errors.append(f"Channel '{name}': output must be 0-{PWMConstants.MAX_CHANNELS - 1}")
            else:
                used_outputs[config.output_channel] = name

            if not PWMConstants.PULSE_MIN <= config.pulse_min <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_min out of range")
            if not PWMConstants.PULSE_MIN <= config.pulse_max <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_max out of range")
            if config.pulse_min >= config.pulse_max:
                errors.append(f"Channel '{name}': pulse_min must be less than pulse_max")

            # Center sanity
            if config.center is not None and not (config.pulse_min <= float(config.center) <= config.pulse_max):
                errors.append(f"Channel '{name}': center must be within [pulse_min, pulse_max]")

            # Deadband and dither bounds
            rng = config.pulse_max - config.pulse_min
            # deadband_us_pos/neg should not exceed half of span and must be >=0
            if float(config.deadband_us_pos) < 0.0 or float(config.deadband_us_pos) > (rng * 0.5):
                errors.append(f"Channel '{name}': deadband_us_pos is unrealistic (0 .. {rng*0.5:.1f}us)")
            if float(config.deadband_us_neg) < 0.0 or float(config.deadband_us_neg) > (rng * 0.5):
                errors.append(f"Channel '{name}': deadband_us_neg is unrealistic (0 .. {rng*0.5:.1f}us)")
            # dither amplitude reasonable vs span
            if float(config.dither_amp_us) < 0.0 or float(config.dither_amp_us) > (rng * 0.25):
                errors.append(f"Channel '{name}': dither_amp_us is unrealistic (0 .. {rng*0.25:.1f}us)")
            # dither frequency sensible (allow 0 when disabled)
            if float(config.dither_hz) < 0.0 or float(config.dither_hz) > 200.0:
                errors.append(f"Channel '{name}': dither_hz must be within [0, 200]")
            # when enabled, require non-zero values
            if config.dither_enable:
                if float(config.dither_amp_us) <= 0.0:
                    errors.append(f"Channel '{name}': dither_amp_us must be > 0 when dither_enable is true")
                if float(config.dither_hz) <= 0.0:
                    errors.append(f"Channel '{name}': dither_hz must be > 0 when dither_enable is true")
                if config.dither_taper and float(config.dither_taper_us) <= 0.0:
                    errors.append(f"Channel '{name}': dither_taper_us must be > 0 when dither_taper is true")
            # ramp limits: enabled channels need a positive rate
            if config.ramp_enable and float(config.ramp_limit) <= 0.0:
                errors.append(f"Channel '{name}': ramp_limit must be > 0 when ramp_enable is true")
            # gamma shaping bounds (keep reasonable)
            if float(config.gamma) <= 0.0 or float(config.gamma) > 5.0:
                errors.append(f"Channel '{name}': gamma must be within (0, 5]")

        if pump_config:
            if pump_config.output_channel in used_outputs:
                errors.append(f"Pump: output {pump_config.output_channel} already used")
            elif not 0 <= pump_config.output_channel < PWMConstants.MAX_CHANNELS:
                errors.append(f"Pump: output must be 0-{PWMConstants.MAX_CHANNELS - 1}")

            if not PWMConstants.PULSE_MIN <= pump_config.pulse_min <= PWMConstants.PULSE_MAX:
                errors.append("Pump: pulse_min out of range")
            if not PWMConstants.PULSE_MIN <= pump_config.pulse_max <= PWMConstants.PULSE_MAX:
                errors.append("Pump: pulse_max out of range")
            if pump_config.pulse_min >= pump_config.pulse_max:
                errors.append("Pump: pulse_min must be less than pulse_max")

            if not (pump_config.pulse_min <= pump_config.static_pulse_us <= pump_config.pulse_max):
                errors.append("Pump: static_pulse_us must be within [pulse_min, pulse_max]")
            if not (pump_config.pulse_min <= pump_config.base_pulse_us <= pump_config.pulse_max):
                errors.append("Pump: base_pulse_us must be within [pulse_min, pulse_max]")
            if not 0.0 <= pump_config.activity_gain_us <= (pump_config.pulse_max - pump_config.pulse_min):
                errors.append("Pump: activity_gain_us must be within [0, pulse_max - pulse_min]")

        if errors:
            raise PWMConfigValidationError("Configuration validation failed:\n" + "\n".join(errors))

    def _build_writer_state(
        self,
        channel_configs: Optional[Dict[str, ChannelConfig]] = None,
        pump_config: Optional[PumpConfig] = None,
        pwm_frequency: Optional[int] = None,
    ) -> tuple[DirectPWMWriter, float, List[ChannelConfig], int, int]:
        channel_configs = self.channel_configs if channel_configs is None else channel_configs
        pump_config = self.pump_config if pump_config is None else pump_config

        all_channels = [cfg.output_channel for cfg in channel_configs.values()]
        if pump_config:
            all_channels.append(pump_config.output_channel)
        min_ch = min(all_channels) if all_channels else 0
        max_ch = max(all_channels) if all_channels else 15

        freq = int(pwm_frequency if pwm_frequency is not None else self._config_pwm_frequency)
        writer = DirectPWMWriter(
            self._i2c,
            min_channel=min_ch,
            max_channel=max_ch,
            frequency=freq,
        )
        pwm_period_us = 1e6 / float(writer.frequency)
        affects_pump_channels = [cfg for cfg in channel_configs.values() if cfg.affects_pump]
        return writer, pwm_period_us, affects_pump_channels, min_ch, max_ch

    def _rebuild_writer(self, pwm_frequency: Optional[int] = None) -> None:
        writer, pwm_period_us, affects_pump_channels, min_ch, max_ch = self._build_writer_state(
            pwm_frequency=pwm_frequency,
        )
        self._direct_writer = writer
        self._pwm_period_us = pwm_period_us
        self._affects_pump_channels = affects_pump_channels
        self.logger.info(
            f"DirectPWMWriter using channels {min_ch}-{max_ch} "
            f"({max_ch - min_ch + 1} channels, {1 + 4 * (max_ch - min_ch + 1)} bytes)"
        )

    def _start_monitoring(self):
        if (self.skip_rate_checking and self._stale_timeout_s <= 0.0) or (self.monitor_thread and self.monitor_thread.is_alive()):
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None

    def _monitor_loop(self):
        _stale_reset_sent = False
        while self.running:
            time.sleep(0.05)
            current_time = time.time()

            # Rate check: count actual update_named() calls per window.
            # _cmd_counter is incremented under _lock in update_named(), so read
            # and reset it atomically here to get an accurate rate regardless of
            # how many commands arrived between monitor wakeups.
            if not self.skip_rate_checking:
                elapsed = current_time - self.rate_window_start
                if elapsed >= self.time_window:
                    with self._lock:
                        count = self._cmd_counter
                        self._cmd_counter = 0
                    rate = count / elapsed if elapsed > 0 else 0.0
                    with self._lock:
                        self.is_safe_state = (rate >= self.input_rate_threshold)
                    self.rate_window_start = current_time

            # Stale command timeout: reset outputs if update_named() stops being called.
            # Uses its own flag so it fires exactly once per stale period.
            if self._stale_timeout_s > 0.0:
                if (time.monotonic() - self._last_command_ts) > self._stale_timeout_s:
                    if not _stale_reset_sent:
                        with self._lock:
                            self.reset(reset_pump=False)
                            self.is_safe_state = False
                        _stale_reset_sent = True
                    continue
                else:
                    _stale_reset_sent = False

    def update_named(self, commands: Dict[str, float], *, unset_to_zero: Optional[bool] = None,
                     command_ts: Optional[float] = None):
        # Performance tracking (before lock to capture full update time)
        self._perf_tracker.tick_start()

        duty_updates = []
        io_locked = False
        with self._lock:
            now_mono = time.monotonic()
            if command_ts is not None and self._stale_timeout_s > 0.0:
                if (now_mono - float(command_ts)) > self._stale_timeout_s:
                    return
            self._last_command_ts = now_mono

            if not self.skip_rate_checking:
                self._cmd_counter += 1
                if not self.is_safe_state:
                    return

            do_zero = self._default_unset_to_zero if unset_to_zero is None else unset_to_zero
            if do_zero:
                for cfg in self.channel_configs.values():
                    self.values[cfg.output_channel] = 0.0

            self.pump_activity_sum = 0.0
            self.pump_activity_count = 0
            for name, val in commands.items():
                cfg = self.channel_configs.get(name)
                if cfg is None:
                    continue
                try:
                    value = float(val)
                except Exception:
                    continue
                value = max(-1.0, min(1.0, value))
                self.values[cfg.output_channel] = value

            for cfg in self._affects_pump_channels:
                self.pump_activity_sum += abs(self.values[cfg.output_channel])
                self.pump_activity_count += 1

            now = time.monotonic()
            duty_updates = self._collect_channel_duties(now)
            pump_update = self._pump_duty_update()
            if pump_update is not None:
                duty_updates.append(pump_update)

            # Preserve write ordering but do not hold the PWM state lock during I2C.
            self._io_lock.acquire()
            io_locked = True

        try:
            self._flush_duty_updates(duty_updates)
        finally:
            if io_locked:
                self._io_lock.release()

        # Performance tracking (after lock released, captures full update cycle)
        self._perf_tracker.tick_end()

    def _collect_channel_duties(self, now: float) -> List[tuple[int, int]]:
        updates: List[tuple[int, int]] = []
        for name, config in self.channel_configs.items():
            if not self.toggle_channels and config.toggleable:
                continue

            value = self.values[config.output_channel]
            pulse = self._pulse_from_value(config, value, now, apply_ramp=True)
            duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
            duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
            updates.append((config.output_channel, duty_cycle))
        return updates

    def _flush_duty_updates(self, updates: List[tuple[int, int]]) -> None:
        for channel, duty_cycle in updates:
            self._direct_writer.set_channel(channel, duty_cycle)
        self._direct_writer.flush()

    def _update_channels(self):
        with self._lock:
            now = time.time()
            for name, config in self.channel_configs.items():
                if not self.toggle_channels and config.toggleable:
                    continue

                value = self.values[config.output_channel]
                pulse = self._pulse_from_value(config, value, now, apply_ramp=True)

                # Convert to duty and queue for batch write
                duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                self._direct_writer.set_channel(config.output_channel, duty_cycle)
            # Note: flush is called by update_named after _update_pump queues pump

    def _pulse_from_value(self, config: ChannelConfig, value: float, now: Optional[float] = None,
                          apply_ramp: bool = False) -> float:
        """Compute output pulse width (us) from normalized value using current config.

        Applies simple deadband by compressing command range into working area.
        Optional dither adds vibration to prevent valve stiction.
        Optional ramp limits slew rate so even deadband jumps are spread over time.
        """
        if now is None:
            now = time.time()

        # Enforce input deadzone locally so preview/compute_pulse() honors it too
        value = self._apply_deadzone(value, float(getattr(config, 'deadzone_threshold', 0.0)))
        # Apply symmetric gamma shaping for both directions
        value = self._apply_gamma(value, float(config.gamma))

        base_pulse = self._compute_base_pulse(config, value)
        if apply_ramp and config.ramp_enable:
            base_pulse = self._apply_ramp(config, base_pulse, now)

        dither_active = abs(base_pulse - float(config.center)) > 1e-6
        use_dither = (
            config.dither_enable
            and float(config.dither_amp_us) != 0.0
            and float(config.dither_hz) != 0.0
            and dither_active
        )
        if use_dither:
            dither_amp = self._dither_amplitude(config, base_pulse)
            base_pulse = self._clamp_base_for_dither(config, base_pulse, dither_amp)
            dither_amp = self._dither_amplitude(config, base_pulse)
            pulse = self._apply_dither(config, base_pulse, now, dither_amp)
        else:
            pulse = base_pulse

        # Final safety clamp: configured pulse limits are hard limits.
        pulse_min = float(config.pulse_min)
        pulse_max = float(config.pulse_max)
        pulse = max(pulse_min, min(pulse_max, pulse))
        return pulse

    def _compute_base_pulse(self, config: ChannelConfig, value: float) -> float:
        """Map normalized value to physical pulse without dither or ramp."""
        # Simple deadband by physical sign (value * direction):
        # - s > 0 => physical positive: jump to center + deadband_us_pos, then scale to pulse_max
        # - s < 0 => physical negative: jump to center - deadband_us_neg, then scale to pulse_min
        # - s == 0 => center
        s = float(value) * float(config.direction)
        if s == 0.0:
            return float(config.center)
        elif s > 0.0:
            base = float(config.center) + float(config.deadband_us_pos)
            working_range = float(config.pulse_max) - base
            return base + abs(float(value)) * working_range
        else:  # s < 0.0
            base = float(config.center) - float(config.deadband_us_neg)
            working_range = base - float(config.pulse_min)
            return base - abs(float(value)) * working_range

    def _dither_amplitude(self, config: ChannelConfig, pulse: float) -> float:
        amp = float(config.dither_amp_us)
        if not config.dither_taper:
            return amp

        center = float(config.center)
        if pulse > center:
            deadband_edge = center + float(config.deadband_us_pos)
            distance_from_edge = pulse - deadband_edge
        else:
            deadband_edge = center - float(config.deadband_us_neg)
            distance_from_edge = deadband_edge - pulse
        taper_us = max(float(config.dither_taper_us), 1e-6)
        taper_gain = max(0.0, min(1.0, distance_from_edge / taper_us))
        return amp * taper_gain

    def _clamp_base_for_dither(self, config: ChannelConfig, pulse: float, dither_amp: float) -> float:
        pulse_min = float(config.pulse_min)
        pulse_max = float(config.pulse_max)
        lower = pulse_min + max(0.0, dither_amp)
        upper = pulse_max - max(0.0, dither_amp)
        if lower > upper:
            return (pulse_min + pulse_max) * 0.5
        return max(lower, min(upper, pulse))

    def _apply_dither(self, config: ChannelConfig, pulse: float, now: float, dither_amp: float) -> float:
        # Dither to prevent valve stiction (only when actively commanding)
        # Per-channel phase offset using output_channel index to avoid perfect sync
        phase = config._dither_omega * now + config._dither_phase_offset
        dither = dither_amp * math.sin(phase)
        return pulse + dither

    def _apply_ramp(self, config: ChannelConfig, target_pulse: float, now: float) -> float:
        """Limit slew rate so large steps are spread over time."""
        state_container = getattr(self, "_channel_ramp_state", None)
        if state_container is None:
            self._channel_ramp_state = {}
            state_container = self._channel_ramp_state

        state = state_container.get(config.output_channel)
        if state is None:
            # Initialize state lazily if a channel was added later
            self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
            return target_pulse

        last_pulse, last_time, last_vel = state
        if not (config.ramp_enable and float(config.ramp_limit) > 0.0):
            self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
            return target_pulse

        # Deadband skip: slew limit only applies to usable range, not deadband jump
        center = float(config.center)
        if config.ramp_skip_deadband:
            pos_edge = center + float(config.deadband_us_pos)
            neg_edge = center - float(config.deadband_us_neg)

            # Transitioning from center to active: start from deadband edge
            if last_pulse == center and target_pulse != center:
                if target_pulse > center:
                    last_pulse = pos_edge
                else:
                    last_pulse = neg_edge
                # If target is exactly at the edge, we're done
                if target_pulse == last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse

            # Crossing from one active side to the other: snap through center
            elif last_pulse > center and target_pulse < center:
                # Was positive, going negative - snap to negative edge
                last_pulse = neg_edge
                if target_pulse >= last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse
            elif last_pulse < center and target_pulse > center:
                # Was negative, going positive - snap to positive edge
                last_pulse = pos_edge
                if target_pulse <= last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse

        dt_raw = max(0.0, now - last_time)
        # Clamp dt so a stalled loop cannot create a giant one-shot jump.
        dt = min(dt_raw, PWMConstants.RAMP_DT_MAX)
        if dt <= 0.0:
            self._channel_ramp_state[config.output_channel] = (last_pulse, now, 0.0)
            return last_pulse

        delta = target_pulse - last_pulse
        allowed_step = float(config.ramp_limit) * dt  # microseconds permitted in this interval
        if abs(delta) <= allowed_step:
            new_pulse = target_pulse
        else:
            new_pulse = last_pulse + allowed_step * (1 if delta > 0 else -1)

        # If ramping toward center and we'd enter deadband region, snap to center
        if config.ramp_skip_deadband and target_pulse == center:
            pos_edge = center + float(config.deadband_us_pos)
            neg_edge = center - float(config.deadband_us_neg)
            if neg_edge < new_pulse < pos_edge:
                new_pulse = center

        # Clamp to user-configured limits
        pulse_min = float(config.pulse_min)
        pulse_max = float(config.pulse_max)
        if new_pulse < pulse_min:
            new_pulse = pulse_min
        elif new_pulse > pulse_max:
            new_pulse = pulse_max
        self._channel_ramp_state[config.output_channel] = (new_pulse, now, 0.0)
        return new_pulse

    # Public helper for testers to preview the pulse for a value
    def compute_pulse(self, name: str, value: float, now: Optional[float] = None) -> Optional[float]:
        cfg = self.channel_configs.get(name)
        if cfg is None:
            return None
        value = max(-1.0, min(1.0, float(value)))
        return self._pulse_from_value(cfg, value, now)

    @staticmethod
    def _apply_deadzone(value: float, threshold: float) -> float:
        """Apply deadzone with rescaling so output uses full range after threshold.

        Values within [-threshold, +threshold] map to 0.
        Values outside are rescaled so that threshold->0 and 1.0->1.0.
        """
        if threshold <= 0.0:
            return value
        abs_val = abs(value)
        if abs_val < threshold:
            return 0.0
        if threshold >= 1.0:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs_val - threshold) / (1.0 - threshold)

    @staticmethod
    def _apply_gamma(value: float, gamma: float) -> float:
        """Apply symmetric gamma shaping to a normalized command."""
        if gamma == 1.0 or value == 0.0:
            return value
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs(value) ** gamma)

    def _update_pump(self, flush: bool = False):
        """Update pump channel. Set flush=True for standalone calls outside update_named."""
        with self._lock:
            update = self._pump_duty_update()
            if update is None:
                return
            if not flush:
                self._direct_writer.set_channel(update[0], update[1])
                return

        with self._io_lock:
            self._flush_duty_updates([update])

    def _pump_duty_update(self) -> Optional[tuple[int, int]]:
        if not self.pump_config:
            return None
        if not self.pump_enabled:
            pulse = float(self.pump_config.pulse_min)
        elif self._pump_direct_us is not None:
            # Direct control: caller owns the pulse, no auto logic applied
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), self._pump_direct_us))
        elif self.pump_auto_mode:
            # Auto mode: scale between base_pulse_us and base+activity_gain based on valve activity
            denom = max(1, self.pump_activity_count)
            avg_activity = self.pump_activity_sum / denom  # 0.0 - 1.0
            pulse = self.pump_config.base_pulse_us + (
                self.pump_config.activity_gain_us * avg_activity
            )
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), pulse))
        else:
            # Static mode: fixed speed from config
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), self.pump_config.static_pulse_us))

        duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
        duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
        return self.pump_config.output_channel, duty_cycle

    def reset(self, reset_pump: bool = True):
        duty_updates = []
        io_locked = False
        with self._lock:
            self.values = [0.0] * PWMConstants.MAX_CHANNELS
            self.pump_activity_sum = 0.0
            self.pump_activity_count = 0
            self._pump_direct_us = None
            for name, config in self.channel_configs.items():
                duty_cycle = int((config.center / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                duty_updates.append((config.output_channel, duty_cycle))
            self._init_ramp_state()
            if reset_pump and self.pump_config:
                duty_cycle = int((self.pump_config.pulse_min / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                duty_updates.append((self.pump_config.output_channel, duty_cycle))
            self.is_safe_state = False
            self.input_counter = 0
            self._io_lock.acquire()
            io_locked = True

        try:
            self._flush_duty_updates(duty_updates)
        finally:
            if io_locked:
                self._io_lock.release()

    def get_average_input_rate(self) -> float:
        current_time = time.time()
        elapsed = current_time - self.rate_window_start
        if elapsed <= 0:
            return 0.0
        rate = self.input_counter / elapsed
        if elapsed >= self.time_window:
            self.input_counter = 0
            self.rate_window_start = current_time
        return rate

    # =========================================================================
    # Performance Tracking API
    # =========================================================================

    def set_perf_enabled(self, enabled: bool) -> None:
        """Enable or disable performance tracking at runtime."""
        self._perf_tracker.set_enabled(enabled)

    def get_perf_stats(self) -> Dict[str, Any]:
        """Get PWM update performance statistics.

        Returns:
            Dictionary with timing stats (all in milliseconds):
            - hz: Measured update rate
            - loop_avg_ms, loop_std_ms, loop_min_ms, loop_max_ms: Update interval stats
            - proc_avg_ms, proc_std_ms, proc_min_ms, proc_max_ms: Processing time stats
            - samples: Number of samples collected
        """
        return self._perf_tracker.get_stats()

    def reset_perf_stats(self) -> None:
        """Reset performance statistics."""
        self._perf_tracker.reset()

    def set_pump_enabled(self, enabled: bool, flush: bool = True):
        with self._lock:
            self.pump_enabled = bool(enabled)
        if flush:
            self._update_pump(flush=True)

    def set_pump_auto(self, auto: bool):
        """Enable (True) or disable (False) valve-activity-based auto speed scaling."""
        with self._lock:
            self.pump_auto_mode = bool(auto)

    def set_pump_activity_gain_us(self, gain_us: float) -> float:
        """Set the auto-mode activity_gain_us at runtime. Returns the clamped value."""
        with self._lock:
            if not self.pump_config:
                return 0.0
            clamped = max(0.0, min(float(gain_us), float(self.pump_config.pulse_max - self.pump_config.pulse_min)))
            self.pump_config.activity_gain_us = clamped
            return clamped

    def get_pump_activity_gain_us(self) -> float:
        """Return current auto-mode activity_gain_us, or 0 if no pump configured."""
        with self._lock:
            return float(self.pump_config.activity_gain_us) if self.pump_config else 0.0

    def set_pump_speed_us(self, pulse_us: Optional[float], flush: bool = True):
        """Set pump speed directly in microseconds, bypassing auto/static logic.

        The pulse is clamped to [pulse_min, pulse_max] from config.
        Pass None to release direct control and return to auto/static mode.
        """
        with self._lock:
            if pulse_us is None:
                self._pump_direct_us = None
            else:
                if self.pump_config:
                    self._pump_direct_us = max(float(self.pump_config.pulse_min),
                                               min(float(self.pump_config.pulse_max), float(pulse_us)))
                else:
                    self._pump_direct_us = float(pulse_us)
        if flush:
            self._update_pump(flush=True)

    def disable_channels(self, disabled: bool):
        self.toggle_channels = not disabled

    def get_channel_names(self, include_pump: bool = True) -> List[str]:
        names = list(self.channel_configs.keys())
        if include_pump and self.pump_config:
            names.append('pump')
        return names

    def build_zero_commands(self, include_toggleable: bool = True, include_pump: bool = False) -> Dict[str, float]:
        commands: Dict[str, float] = {}
        for name, cfg in self.channel_configs.items():
            if include_toggleable or not cfg.toggleable:
                commands[name] = 0.0
        if include_pump and self.pump_config:
            commands['pump'] = 0.0
        return commands

    def reload_config(self, config_file: str) -> bool:
        config_path = None
        was_monitoring = self.running
        try:
            if was_monitoring:
                self._stop_monitoring()

            with self._lock:
                # Hold outputs neutral while the config file is parsed/swapped.
                # This blocks concurrent command writes until reload completes.
                self.reset(reset_pump=True)

                config_path, channel_configs, pump_config, pwm_frequency = self._read_config_data(config_file)
                writer_state = self._build_writer_state(
                    channel_configs=channel_configs,
                    pump_config=pump_config,
                    pwm_frequency=int(pwm_frequency),
                )
                self._config_path = config_path
                self.channel_configs = channel_configs
                self.pump_config = pump_config
                self._config_pwm_frequency = pwm_frequency
                self._direct_writer, self._pwm_period_us, self._affects_pump_channels, min_ch, max_ch = writer_state
                self.logger.info(
                    f"DirectPWMWriter using channels {min_ch}-{max_ch} "
                    f"({max_ch - min_ch + 1} channels, {1 + 4 * (max_ch - min_ch + 1)} bytes)"
                )
                self.reset(reset_pump=True)

            if was_monitoring:
                self._start_monitoring()
            return True
        except PWMConfigError as exc:
            location = config_path if config_path is not None else config_file
            self.logger.error(f"Configuration reload failed for '{location}': {exc}")
            if was_monitoring:
                self._start_monitoring()
            return False
        except Exception as exc:
            self.logger.error(f"Unexpected error reloading configuration '{config_file}': {exc}")
            if was_monitoring:
                self._start_monitoring()
            return False

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _init_ramp_state(self):
        now = time.time()
        self._channel_ramp_state: Dict[int, tuple[float, float, float]] = {}
        for cfg in self.channel_configs.values():
            self._channel_ramp_state[cfg.output_channel] = (float(cfg.center), now, 0.0)

    def _simple_cleanup(self):
        """Cleanup on exit: reset channels to center, optionally stop oscillator."""
        try:
            self._stop_monitoring()
            with self._lock:
                # Reset all channels to center (safe neutral position)
                self.reset(reset_pump=True)
                # Small delay for hardware to settle at center position
                time.sleep(0.05)
                # Optionally stop oscillator (outputs go LOW) or keep running (outputs stay at center)
                if self._cleanup_disable_osc:
                    self._direct_writer.sleep()
        except Exception as exc:
            self.logger.warning(f"PWM cleanup failed: {exc}")

    def set_cleanup_disable_osc(self, enabled: bool) -> None:
        """Configure whether oscillator stops on cleanup.

        Args:
            enabled: If True, stop oscillator on cleanup (outputs go LOW).
                     If False, keep oscillator running (outputs stay at center).
        """
        self._cleanup_disable_osc = enabled
