"""
Balanced PWM Controller - Clean structure with battle-tested reliability
"""

import atexit
import threading
import time
import yaml
# Removed deque import for better real-time performance
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
    # TODO: why some of the values are pre-filled?
    input_channel: Optional[int]
    output_channel: int
    pulse_min: int
    pulse_max: int
    direction: int
    center: Optional[float] = None
    deadzone: float = 10.0
    gamma_positive: float = 1.0
    gamma_negative: float = 1.0
    affects_pump: bool = False
    toggleable: bool = False

    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.pulse_range = self.pulse_max - self.pulse_min
        if self.center is None:
            self.center = self.pulse_min + (self.pulse_range / 2)
        self.deadzone_threshold = self.deadzone / 100.0 * 2


@dataclass
class PumpConfig:
    """Configuration specific to pump control."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    idle: float
    multiplier: float
    input_channel: Optional[int] = None


# ============================================================================
# Constants
# ============================================================================

class PWMConstants:
    """Hardware and timing constants."""
    PWM_FREQUENCY = 50  # Hz for standard servos
    MAX_CHANNELS = 16  # PCA9685 limitation
    PWM_PERIOD_US = 20000  # microseconds
    DUTY_CYCLE_MAX = 65535

    # Validation limits
    GAMMA_MIN = 0.1
    GAMMA_MAX = 3.0
    PULSE_MIN = 0
    PULSE_MAX = 4095
    PUMP_IDLE_MIN = -1.0
    PUMP_IDLE_MAX = 0.6
    PUMP_MULTIPLIER_MAX = 1.0

    # Safety parameters
    DEFAULT_TIME_WINDOW = 5  # seconds
    SAFE_STATE_THRESHOLD = 0.25  # 25% of threshold rate


# ============================================================================
# Simple PWM Controller
# ============================================================================

class PWMController:
    """Simple, reliable PWM controller with clean structure."""

    def __init__(self, config_file: str, pump_variable: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0,
                 default_unset_to_zero: bool = True):

        # Simple state variables
        self.pump_variable = pump_variable
        self.toggle_channels = toggle_channels
        self.pump_enabled = True
        self.manual_pump_load = 0.0
        self.pump_variable_sum = 0.0
        self._pump_override_throttle: Optional[float] = None  # Direct throttle override via name-based API

        # Rate monitoring
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking
        self.time_window = PWMConstants.DEFAULT_TIME_WINDOW

        # Simple threading
        self.running = False
        self.input_event = threading.Event()
        self.monitor_thread = None
        self.input_count = 0
        self.last_input_time = time.time()

        # Load and validate configuration
        self._load_config(config_file)

        # Initialize hardware
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = PWMConstants.PWM_FREQUENCY

        # Current values for each channel
        self.values = [0.0] * PWMConstants.MAX_CHANNELS

        # Rate monitoring setup - simple counter-based approach
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
        """Load and parse configuration file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")

        with open(config_path, 'r') as file:
            raw_config = yaml.safe_load(file)

        self.channel_configs, self.pump_config = self._parse_config(raw_config)
        self._validate_config()

    def _parse_config(self, raw_config: Dict) -> tuple[Dict[str, ChannelConfig], Optional[PumpConfig]]:
        """Parse raw configuration into structured configs."""
        channel_configs = {}
        pump_config = None

        for name, config in raw_config['CHANNEL_CONFIGS'].items():
            if name == 'pump':
                pump_config = PumpConfig(
                    output_channel=config['output_channel'],
                    pulse_min=config['pulse_min'],
                    pulse_max=config['pulse_max'],
                    idle=config['idle'],
                    multiplier=config['multiplier'],
                    input_channel=self._normalize_none(config.get('input_channel'))
                )
            else:
                channel_configs[name] = ChannelConfig(
                    input_channel=self._normalize_none(config.get('input_channel')),
                    output_channel=config['output_channel'],
                    pulse_min=config['pulse_min'],
                    pulse_max=config['pulse_max'],
                    direction=config['direction'],
                    center=self._normalize_none(config.get('center')),
                    deadzone=config.get('deadzone', 10.0),
                    gamma_positive=config.get('gamma_positive', 1.0),
                    gamma_negative=config.get('gamma_negative', 1.0),
                    affects_pump=config.get('affects_pump', False),
                    toggleable=config.get('toggleable', False)
                )

        return channel_configs, pump_config

    @staticmethod
    def _normalize_none(value: Any) -> Optional[Any]:
        """Normalize various representations of None."""
        none_values = [None, "None", "none", "null", "NONE", "Null", "NULL", "", "n/a", "N/A"]
        return None if value in none_values else value

    def _validate_config(self):
        """Simple configuration validation."""
        errors = []
        used_inputs = {}
        used_outputs = {}

        # Validate channels
        for name, config in self.channel_configs.items():
            # Check direction
            if config.direction not in [-1, 1]:
                errors.append(f"Channel '{name}': direction must be -1 or 1")

            # Check input channel
            if config.input_channel is not None:
                if config.input_channel in used_inputs:
                    errors.append(f"Channel '{name}': input {config.input_channel} already used")
                elif not 0 <= config.input_channel < PWMConstants.MAX_CHANNELS:
                    errors.append(f"Channel '{name}': input must be 0-{PWMConstants.MAX_CHANNELS - 1}")
                else:
                    used_inputs[config.input_channel] = name

            # Check output channel
            if config.output_channel in used_outputs:
                errors.append(f"Channel '{name}': output {config.output_channel} already used")
            elif not 0 <= config.output_channel < PWMConstants.MAX_CHANNELS:
                errors.append(f"Channel '{name}': output must be 0-{PWMConstants.MAX_CHANNELS - 1}")
            else:
                used_outputs[config.output_channel] = name

            # Check pulse range
            if not PWMConstants.PULSE_MIN <= config.pulse_min <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_min out of range")
            if not PWMConstants.PULSE_MIN <= config.pulse_max <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_max out of range")
            if config.pulse_min >= config.pulse_max:
                errors.append(f"Channel '{name}': pulse_min must be less than pulse_max")

            # Check gamma values
            if not PWMConstants.GAMMA_MIN <= config.gamma_positive <= PWMConstants.GAMMA_MAX:
                errors.append(f"Channel '{name}': gamma_positive out of range")
            if not PWMConstants.GAMMA_MIN <= config.gamma_negative <= PWMConstants.GAMMA_MAX:
                errors.append(f"Channel '{name}': gamma_negative out of range")

        # Validate pump
        if self.pump_config:
            if self.pump_config.output_channel in used_outputs:
                errors.append(f"Pump: output {self.pump_config.output_channel} already used")
            if not PWMConstants.PUMP_IDLE_MIN <= self.pump_config.idle <= PWMConstants.PUMP_IDLE_MAX:
                errors.append(f"Pump: idle out of range")
            if not 0 < self.pump_config.multiplier <= PWMConstants.PUMP_MULTIPLIER_MAX:
                errors.append(f"Pump: multiplier out of range")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    # Index-based input counting removed with name-based API.

    def _start_monitoring(self):
        """Start simple monitoring thread."""
        if self.skip_rate_checking or (self.monitor_thread and self.monitor_thread.is_alive()):
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        # print(f"Input rate monitoring started (threshold: {self.input_rate_threshold}Hz)")

    def _stop_monitoring(self):
        """Stop monitoring thread - simple and reliable."""
        if not self.running:
            return

        self.running = False
        self.input_event.set()  # Wake up thread

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        self.monitor_thread = None
        print("Input rate monitoring stopped")

    def _monitor_loop(self):
        """Simple monitoring loop - based on your reliable original."""
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
                    print("Input rate too low. Entering safe state...")
                    self.reset(reset_pump=False)
                    self.is_safe_state = False
                    self.input_count = 0

    # Removed index-based update_values; use update_named exclusively.

    def update_named(self, commands: Dict[str, float], *, unset_to_zero: Optional[bool] = None,
                     one_shot_pump_override: bool = True) -> None:
        """Update channel values by name.

        Args:
            commands: Mapping from channel name to command value in [-1, 1]. Unknown names are ignored.
            unset_to_zero: When True, channels not present in the mapping are set to 0.

        Also supports an optional 'pump' key to directly override pump throttle in [-1, 1].
        """
        # Rate monitoring signal
        if not self.skip_rate_checking:
            self.input_event.set()
            if not self.is_safe_state:
                return

        # Handle pump override if provided
        if 'pump' in commands and self.pump_config:
            try:
                pump_val = float(commands['pump'])
                self._pump_override_throttle = max(-1.0, min(1.0, pump_val))
            except Exception:
                pass

        # Resolve zeroing behavior (per-call override or controller default)
        do_zero = self._default_unset_to_zero if unset_to_zero is None else unset_to_zero

        # If requested, zero all channels first
        if do_zero:
            for cfg in self.channel_configs.values():
                self.values[cfg.output_channel] = 0.0

        # Apply provided named commands
        self.pump_variable_sum = 0.0
        for name, val in commands.items():
            cfg = self.channel_configs.get(name)
            if cfg is None:
                continue  # Ignore unknown names
            try:
                value = float(val)
            except Exception:
                continue

            # Clamp and apply deadzone
            value = max(-1.0, min(1.0, value))
            if abs(value) < cfg.deadzone_threshold:
                value = 0.0

            # Store value
            self.values[cfg.output_channel] = value

        # Compute pump variable sum based on current values for channels that affect pump
        for cfg in self.channel_configs.values():
            if cfg.affects_pump:
                self.pump_variable_sum += abs(self.values[cfg.output_channel])

        # Push to hardware
        self._update_channels()
        self._update_pump()

        # By default, manual pump override applies for this update only
        if one_shot_pump_override:
            self._pump_override_throttle = None

    def _update_channels(self):
        """Update channel outputs."""
        for name, config in self.channel_configs.items():
            if not self.toggle_channels and config.toggleable:
                continue

            value = self.values[config.output_channel]

            # Apply gamma correction
            if value >= 0:
                adjusted = value ** config.gamma_positive
            else:
                adjusted = -((-value) ** config.gamma_negative)

            # Calculate pulse width
            pulse = config.center + (adjusted * config.pulse_range / 2 * config.direction)
            pulse = max(config.pulse_min, min(config.pulse_max, pulse))

            # Set PWM
            duty_cycle = int((pulse / PWMConstants.PWM_PERIOD_US) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[config.output_channel].duty_cycle = duty_cycle

    def _update_pump(self):
        """Update pump output."""
        if not self.pump_config:
            return

        # Direct override (highest priority)
        if self._pump_override_throttle is not None:
            throttle = self._pump_override_throttle
        elif not self.pump_enabled:
            throttle = -1.0
        elif self.pump_config.input_channel is None:
            # Automatic pump control
            if self.pump_variable:
                throttle = self.pump_config.idle + (self.pump_config.multiplier * self.pump_variable_sum / 10)
            else:
                throttle = self.pump_config.idle + (self.pump_config.multiplier / 10)
            throttle += self.manual_pump_load
        else:
            # Manual pump control
            if self.pump_config.input_channel < len(self.values):
                throttle = self.values[self.pump_config.input_channel]
            else:
                throttle = self.pump_config.idle

        # Clamp and convert
        throttle = max(-1.0, min(1.0, throttle))
        pulse_range = self.pump_config.pulse_max - self.pump_config.pulse_min
        pulse = self.pump_config.pulse_min + pulse_range * ((throttle + 1) / 2)

        duty_cycle = int((pulse / PWMConstants.PWM_PERIOD_US) * PWMConstants.DUTY_CYCLE_MAX)
        self.pca.channels[self.pump_config.output_channel].duty_cycle = duty_cycle

    def reset(self, reset_pump: bool = True):
        """Reset all channels to center/idle positions."""
        for name, config in self.channel_configs.items():
            duty_cycle = int((config.center / PWMConstants.PWM_PERIOD_US) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[config.output_channel].duty_cycle = duty_cycle

        if reset_pump and self.pump_config:
            duty_cycle = int((self.pump_config.pulse_min / PWMConstants.PWM_PERIOD_US) * PWMConstants.DUTY_CYCLE_MAX)
            self.pca.channels[self.pump_config.output_channel].duty_cycle = duty_cycle

        self.is_safe_state = False
        self.input_count = 0
        self._pump_override_throttle = None

    def get_average_input_rate(self) -> float:
        """Calculate average input rate using simple counter approach."""
        current_time = time.time()
        elapsed = current_time - self.rate_window_start
        
        if elapsed <= 0:
            return 0.0
        
        rate = self.input_counter / elapsed
        
        # Reset counter every window to prevent overflow
        if elapsed >= self.time_window:
            self.input_counter = 0
            self.rate_window_start = current_time
        
        return rate

    def set_pump(self, enabled: bool):
        """Enable/disable pump."""
        self.pump_enabled = enabled
        print(f"Pump enabled: {self.pump_enabled}")

    def toggle_pump_variable(self, variable: bool):
        """Toggle variable pump speed."""
        self.pump_variable = variable
        print(f"Pump variable: {self.pump_variable}")

    def update_pump_load(self, adjustment: float):
        """Manually adjust pump load."""
        self.manual_pump_load = max(-1.0, min(0.3, self.manual_pump_load + adjustment / 10))

    def reset_pump_load(self):
        """Reset manual pump load."""
        self.manual_pump_load = 0.0
        self._update_pump()

    def disable_channels(self, disabled: bool):
        """Enable/disable toggleable channels."""
        self.toggle_channels = not disabled
        print(f"Toggleable channels enabled: {self.toggle_channels}")

    def clear_pump_override(self):
        """Clear direct pump throttle override set via name-based API."""
        self._pump_override_throttle = None

    def get_input_mapping(self) -> Dict[str, Dict[str, int]]:
        """Get input/output channel mappings."""
        result = {}
        for name, config in self.channel_configs.items():
            if config.input_channel is not None:
                result[name] = {
                    'input_num': config.input_channel,
                    'output_channel': config.output_channel
                }
        return result

    def get_channel_names(self, include_pump: bool = True) -> List[str]:
        """Return configured channel names. Optionally include 'pump'."""
        names = list(self.channel_configs.keys())
        if include_pump and self.pump_config:
            names.append('pump')
        return names

    def build_zero_commands(self, include_toggleable: bool = True, include_pump: bool = False) -> Dict[str, float]:
        """Convenience helper to create a zeroed command dict.

        Args:
            include_toggleable: Include channels marked toggleable in the returned dict.
            include_pump: Include 'pump' key set to 0.0.
        """
        commands: Dict[str, float] = {}
        for name, cfg in self.channel_configs.items():
            if include_toggleable or not cfg.toggleable:
                commands[name] = 0.0
        if include_pump and self.pump_config:
            commands['pump'] = 0.0
        return commands

    def reload_config(self, config_file: str) -> bool:
        """Reload configuration from file."""
        self.reset(reset_pump=True)
        print(f"Reloading configuration from {config_file}")

        try:
            # Stop monitoring temporarily
            was_monitoring = self.running
            if was_monitoring:
                self._stop_monitoring()

            # Load new configuration
            self._load_config(config_file)
            self.reset(reset_pump=True)

            # Restart monitoring if it was running
            if was_monitoring:
                self._start_monitoring()

            print("Configuration reloaded successfully")
            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def _simple_cleanup(self):
        """Simple, reliable cleanup."""
        try:
            print("PWM Controller cleanup...")
            self._stop_monitoring()
            self.reset(reset_pump=True)
            print("PWM Controller cleanup complete")
        except:
            pass  # Ignore cleanup errors
