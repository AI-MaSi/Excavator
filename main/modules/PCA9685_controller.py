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
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path

from adafruit_pca9685 import PCA9685
import board
import busio


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

    # Dither settings to prevent valve stiction - DISABLED by default
    dither_enable: bool = False
    dither_amp_us: float = 8.0  # vibration amplitude in microseconds
    dither_hz: float = 40.0  # vibration frequency

    # Slew rate limiting (microseconds per second) to soften command steps
    ramp_enable: bool = False
    ramp_limit: float = 0.0  # us/s; ignored when ramp_enable is False

    # Symmetric gamma shaping (1.0 = linear). Applied to magnitude for both directions.
    gamma: float = 1.0

    def __post_init__(self):
        self.pulse_range = self.pulse_max - self.pulse_min
        if self.center is None:
            self.center = self.pulse_min + (self.pulse_range / 2)
        self.deadzone_threshold = self.deadzone / 100.0 * 2


@dataclass
class PumpConfig:
    """Configuration specific to pump control (not typically used in valve_testing)."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    idle: float
    multiplier: float
    # Manual pump via input channel removed in testing controller.


class PWMConstants:
    """Hardware and timing constants."""
    PWM_FREQUENCY_DEFAULT = 100  # 50 Hz, standard RC pulse rate
    MAX_CHANNELS = 16
    DUTY_CYCLE_MAX = 65535

    # Validation limits
    PULSE_MIN = 0
    PULSE_MAX = 4095
    PUMP_IDLE_MIN = -1.0
    PUMP_IDLE_MAX = 0.6
    PUMP_MULTIPLIER_MAX = 1.0

    # Safety parameters
    DEFAULT_TIME_WINDOW = 5  # seconds
    SAFE_STATE_THRESHOLD = 0.25


class PWMController:
    """Simple PWM controller with piecewise deadband and dither for valve testing."""

    def __init__(self, config_file: str, pump_variable: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0,
                 default_unset_to_zero: bool = True, log_level: str = "INFO"):
        """Initialize PWM controller.

        Args:
            config_file: Path to YAML configuration file
            pump_variable: Enable variable pump speed
            toggle_channels: Enable toggleable channels
            input_rate_threshold: Input rate threshold for safety monitoring
            default_unset_to_zero: Default unset channels to zero
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
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

        self.pump_variable = pump_variable
        self.toggle_channels = toggle_channels
        self.pump_enabled = True
        self.manual_pump_load = 0.0
        self.pump_variable_sum = 0.0
        self._pump_override_throttle: Optional[float] = None

        # Rate monitoring (optional)
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking
        self.time_window = PWMConstants.DEFAULT_TIME_WINDOW

        # Threads/monitoring
        self.running = False
        self.input_event = threading.Event()
        self.monitor_thread = None
        self.input_count = 0
        self.last_input_time = time.time()

        # Load config
        self._load_config(config_file)

        # Hardware init
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = PWMConstants.PWM_FREQUENCY_DEFAULT
        self._pwm_period_us = 1e6 / float(self.pca.frequency)

        # Current normalized values per channel
        self.values = [0.0] * PWMConstants.MAX_CHANNELS

        # Monitoring counters
        self.input_counter = 0
        self.rate_window_start = time.time()

        # Register simple cleanup and start monitoring
        atexit.register(self._simple_cleanup)
        self.reset()

        if not self.skip_rate_checking:
            self._start_monitoring()
        
        # Behavior defaults
        self._default_unset_to_zero = default_unset_to_zero

    def _load_config(self, config_file: str):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        self.channel_configs, self.pump_config = self._parse_config(raw_config)
        self._validate_config()

    def _parse_config(self, raw_config: Dict) -> tuple[Dict[str, ChannelConfig], Optional[PumpConfig]]:
        channel_configs: Dict[str, ChannelConfig] = {}
        pump_config = None

        for name, cfg in raw_config['CHANNEL_CONFIGS'].items():
            if name == 'pump':
                pump_config = PumpConfig(
                    output_channel=cfg['output_channel'],
                    pulse_min=cfg['pulse_min'],
                    pulse_max=cfg['pulse_max'],
                    idle=cfg['idle'],
                    multiplier=cfg['multiplier'],
                )
            else:
                channel_configs[name] = ChannelConfig(
                    output_channel=cfg['output_channel'],
                    pulse_min=cfg['pulse_min'],
                    pulse_max=cfg['pulse_max'],
                    direction=cfg['direction'],
                    center=self._normalize_none(cfg.get('center')),
                    deadzone=cfg.get('deadzone', 0.0),
                    affects_pump=cfg.get('affects_pump', False),
                    toggleable=cfg.get('toggleable', False),
                    # Simple deadband (separate per sign)
                    deadband_us_pos=float(cfg['deadband_us_pos']),
                    deadband_us_neg=float(cfg['deadband_us_neg']),
                    # Dither settings - opt-in only
                    dither_enable=cfg.get('dither_enable', False),
                    dither_amp_us=cfg.get('dither_amp_us', 8.0),
                    dither_hz=cfg.get('dither_hz', 40.0),
                    # Ramp/slew limiting - opt-in per channel
                    ramp_enable=cfg.get('ramp_enable', False),
                    ramp_limit=float(cfg.get('ramp_limit', 0.0)),
                    # Symmetric gamma shaping
                    gamma=float(cfg.get('gamma', 1.0)),
                )

        return channel_configs, pump_config

    @staticmethod
    def _normalize_none(value: Any) -> Optional[Any]:
        none_values = [None, "None", "none", "null", "NONE", "Null", "NULL", "", "n/a", "N/A"]
        return None if value in none_values else value

    def _validate_config(self):
        errors = []
        used_outputs = {}

        for name, config in self.channel_configs.items():
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
            # dither frequency sensible
            if float(config.dither_hz) <= 0.0 or float(config.dither_hz) > 200.0:
                errors.append(f"Channel '{name}': dither_hz must be within (0, 200]")
            # ramp limits: enabled channels need a positive rate
            if config.ramp_enable and float(config.ramp_limit) <= 0.0:
                errors.append(f"Channel '{name}': ramp_limit must be > 0 when ramp_enable is true")
            # gamma shaping bounds (keep reasonable)
            if float(config.gamma) <= 0.0 or float(config.gamma) > 5.0:
                errors.append(f"Channel '{name}': gamma must be within (0, 5]")

        if self.pump_config:
            if self.pump_config.output_channel in used_outputs:
                errors.append(f"Pump: output {self.pump_config.output_channel} already used")
            if not PWMConstants.PUMP_IDLE_MIN <= self.pump_config.idle <= PWMConstants.PUMP_IDLE_MAX:
                errors.append("Pump: idle out of range")
            if not 0 < self.pump_config.multiplier <= PWMConstants.PUMP_MULTIPLIER_MAX:
                errors.append("Pump: multiplier out of range")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    def _start_monitoring(self):
        if self.skip_rate_checking or (self.monitor_thread and self.monitor_thread.is_alive()):
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        self.input_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None

    def _monitor_loop(self):
        while self.running:
            if self.input_event.wait(timeout=1.0 / self.input_rate_threshold):
                self.input_event.clear()
                current_time = time.time()
                time_diff = current_time - self.last_input_time
                self.last_input_time = current_time
                if time_diff > 0:
                    current_rate = 1 / time_diff
                    if current_rate >= self.input_rate_threshold:
                        self.input_count += 1
                        required_count = int(self.input_rate_threshold * PWMConstants.SAFE_STATE_THRESHOLD)
                        if self.input_count >= required_count:
                            self.is_safe_state = True
                            self.input_count = 0
                    else:
                        self.input_count = 0
                    self.input_counter += 1
            else:
                if self.is_safe_state:
                    self.reset(reset_pump=False)
                    self.is_safe_state = False
                    self.input_count = 0

    def update_named(self, commands: Dict[str, float], *, unset_to_zero: Optional[bool] = None,
                     one_shot_pump_override: bool = True):
        if not self.skip_rate_checking:
            self.input_event.set()
            if not self.is_safe_state:
                return

        if 'pump' in commands and self.pump_config:
            try:
                pump_val = float(commands['pump'])
                self._pump_override_throttle = max(-1.0, min(1.0, pump_val))
            except Exception:
                pass

        do_zero = self._default_unset_to_zero if unset_to_zero is None else unset_to_zero
        if do_zero:
            for cfg in self.channel_configs.values():
                self.values[cfg.output_channel] = 0.0

        self.pump_variable_sum = 0.0
        for name, val in commands.items():
            cfg = self.channel_configs.get(name)
            if cfg is None:
                continue
            try:
                value = float(val)
            except Exception:
                continue
            value = max(-1.0, min(1.0, value))
            if abs(value) < cfg.deadzone_threshold:
                value = 0.0
            self.values[cfg.output_channel] = value

        for cfg in self.channel_configs.values():
            if cfg.affects_pump:
                self.pump_variable_sum += abs(self.values[cfg.output_channel])

        self._update_channels()
        self._update_pump()

        if one_shot_pump_override:
            self._pump_override_throttle = None

    def _update_channels(self):
        now = time.time()
        for name, config in self.channel_configs.items():
            if not self.toggle_channels and config.toggleable:
                continue

            value = self.values[config.output_channel]
            pulse = self._pulse_from_value(config, value, now, apply_ramp=True)

            # Convert to duty and push
            duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[config.output_channel].duty_cycle = duty_cycle

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
        if abs(value) < float(getattr(config, 'deadzone_threshold', 0.0)):
            value = 0.0
        # Apply symmetric gamma shaping for both directions
        value = self._apply_gamma(value, float(config.gamma))

        base_pulse = self._compute_base_pulse(config, value)
        if apply_ramp:
            base_pulse = self._apply_ramp(config, base_pulse, now)
        pulse = self._apply_dither(config, base_pulse, value, now)

        # Clamp to limits
        pulse = max(config.pulse_min, min(config.pulse_max, pulse))
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

    def _apply_dither(self, config: ChannelConfig, pulse: float, value: float, now: float) -> float:
        # Dither to prevent valve stiction (only when actively commanding)
        if config.dither_enable and abs(value) >= float(getattr(config, 'deadzone_threshold', 0.0)):
            # Per-channel phase offset using output_channel index to avoid perfect sync
            phase = 2.0 * 3.141592653589793 * config.dither_hz * now + (config.output_channel * 1.0471975512)
            dither = config.dither_amp_us * __import__('math').sin(phase)
            pulse += dither
        return pulse

    def _apply_ramp(self, config: ChannelConfig, target_pulse: float, now: float) -> float:
        """Limit slew rate so large steps are spread over time."""
        state_container = getattr(self, "_channel_ramp_state", None)
        if state_container is None:
            self._channel_ramp_state = {}
            state_container = self._channel_ramp_state

        state = state_container.get(config.output_channel)
        if state is None:
            # Initialize state lazily if a channel was added later
            self._channel_ramp_state[config.output_channel] = (target_pulse, now)
            return target_pulse

        last_pulse, last_time = state
        if not config.ramp_enable or float(config.ramp_limit) <= 0.0:
            self._channel_ramp_state[config.output_channel] = (target_pulse, now)
            return target_pulse

        dt_raw = max(0.0, now - last_time)
        # Clamp dt so a stalled loop cannot create a giant one-shot jump; allow up to 2x the prior interval.
        if self._last_ramp_dt > 0.0:
            dt = min(dt_raw, self._last_ramp_dt * 2.0)
        else:
            dt = dt_raw
        if dt <= 0.0:
            self._channel_ramp_state[config.output_channel] = (last_pulse, now)
            return last_pulse

        allowed_step = float(config.ramp_limit) * dt  # microseconds permitted in this interval
        delta = target_pulse - last_pulse
        if abs(delta) <= allowed_step:
            new_pulse = target_pulse
        else:
            new_pulse = last_pulse + allowed_step * (1 if delta > 0 else -1)

        self._channel_ramp_state[config.output_channel] = (new_pulse, now)
        # Remember unclamped dt to keep the clamp adaptive to the real loop cadence
        if dt_raw > 0.0:
            self._last_ramp_dt = dt_raw
        return new_pulse

    # Public helper for testers to preview the pulse for a value
    def compute_pulse(self, name: str, value: float, now: Optional[float] = None) -> Optional[float]:
        cfg = self.channel_configs.get(name)
        if cfg is None:
            return None
        value = max(-1.0, min(1.0, float(value)))
        return self._pulse_from_value(cfg, value, now)

    @staticmethod
    def _apply_gamma(value: float, gamma: float) -> float:
        """Apply symmetric gamma shaping to a normalized command."""
        if gamma == 1.0 or value == 0.0:
            return value
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs(value) ** gamma)

    def _update_pump(self):
        if not self.pump_config:
            return
        if self._pump_override_throttle is not None:
            throttle = self._pump_override_throttle
        elif not self.pump_enabled:
            throttle = -1.0
        else:
            if self.pump_variable:
                throttle = self.pump_config.idle + (self.pump_config.multiplier * self.pump_variable_sum / 10)
            else:
                throttle = self.pump_config.idle + (self.pump_config.multiplier / 10)
            throttle += self.manual_pump_load

        throttle = max(-1.0, min(1.0, throttle))
        pulse_range = self.pump_config.pulse_max - self.pump_config.pulse_min
        pulse = self.pump_config.pulse_min + pulse_range * ((throttle + 1) / 2)
        duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
        self.pca.channels[self.pump_config.output_channel].duty_cycle = duty_cycle

    def reset(self, reset_pump: bool = True):
        for name, config in self.channel_configs.items():
            duty_cycle = int((config.center / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[config.output_channel].duty_cycle = duty_cycle
        self._init_ramp_state()
        if reset_pump and self.pump_config:
            duty_cycle = int((self.pump_config.pulse_min / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[self.pump_config.output_channel].duty_cycle = duty_cycle
        self.is_safe_state = False
        self.input_count = 0
        self._pump_override_throttle = None

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

    def set_pump(self, enabled: bool):
        self.pump_enabled = enabled

    def toggle_pump_variable(self, variable: bool):
        self.pump_variable = variable

    def update_pump_load(self, adjustment: float):
        self.manual_pump_load = max(-1.0, min(0.3, self.manual_pump_load + adjustment / 10))

    def reset_pump_load(self):
        self.manual_pump_load = 0.0
        self._update_pump()

    def disable_channels(self, disabled: bool):
        self.toggle_channels = not disabled

    def clear_pump_override(self):
        self._pump_override_throttle = None

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
        self.reset(reset_pump=True)
        try:
            was_monitoring = self.running
            if was_monitoring:
                self._stop_monitoring()
            self._load_config(config_file)
            self.reset(reset_pump=True)
            if was_monitoring:
                self._start_monitoring()
            return True
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _init_ramp_state(self):
        now = time.time()
        self._channel_ramp_state: Dict[int, tuple[float, float]] = {}
        for cfg in self.channel_configs.values():
            self._channel_ramp_state[cfg.output_channel] = (float(cfg.center), now)
        # Track last observed dt for adaptive clamp
        self._last_ramp_dt: float = 0.0

    def _simple_cleanup(self):
        try:
            self._stop_monitoring()
            self.reset(reset_pump=True)
        except:
            pass
