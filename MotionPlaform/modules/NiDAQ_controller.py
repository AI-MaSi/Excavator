"""
NiDAQ joystick reader.
Reads analog (joystick) and digital (button) inputs from a National Instruments DAQ device.
Optionally returns values as floats or 8-bit signed integers.
"""

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.errors import DaqError

# ----- Configuration -----
# Hardware voltage range: defines both DAQ input range and normalization
MIN_VOLTAGE = 0.5      # Minimum joystick voltage (maps to -1.0)
MAX_VOLTAGE = 4.5      # Maximum joystick voltage (maps to +1.0)

AI_CHANNELS = [
    "Dev2/ai0", "Dev2/ai1", "Dev2/ai2", "Dev2/ai3",
    "Dev2/ai4", "Dev2/ai5", "Dev2/ai6", "Dev2/ai7"
]

DI_CHANNELS = [
    "Dev2/port0/line0", "Dev2/port0/line1", "Dev2/port0/line2", "Dev2/port0/line3",
    "Dev2/port0/line4", "Dev2/port0/line5", "Dev2/port0/line6", "Dev2/port0/line7",
    "Dev2/port1/line0", "Dev2/port1/line1", "Dev2/port1/line2", "Dev2/port1/line3"
]


class NiDAQJoysticks:
    def __init__(self, output_format="float"):
        """
        :param output_format: 'float' for normalized [-1, 1] AI and {0.0, 1.0} DI,
                              'int8' for [-128, 127] AI and {0, 127} DI.
        """
        if output_format not in ("float", "int8"):
            raise ValueError("output_format must be 'float' or 'int8'")
        self.output_format = output_format
        self.task_ai = nidaqmx.Task()
        self.task_di = nidaqmx.Task()
        self._init_channels()

    def _init_channels(self):
        """Initialize DAQ tasks for AI and DI channels with hardware-level voltage range."""
        try:
            for ch in DI_CHANNELS:
                self.task_di.di_channels.add_di_chan(ch)
            for ch in AI_CHANNELS:
                self.task_ai.ai_channels.add_ai_voltage_chan(
                    ch,
                    min_val=MIN_VOLTAGE,
                    max_val=MAX_VOLTAGE,
                    terminal_config=TerminalConfiguration.RSE
                )
            print(f"NiDAQ initialized: {len(AI_CHANNELS)} AI channels, {len(DI_CHANNELS)} DI channels.")
            print(f"Hardware voltage range: {MIN_VOLTAGE}V to {MAX_VOLTAGE}V")
        except DaqError as e:
            self.close()
            raise RuntimeError(f"Failed to initialize NiDAQ: {e}")

    def _normalize_ai(self, ai_values):
        """Convert raw voltages to requested format (maps MIN_VOLTAGE→-1.0, MAX_VOLTAGE→+1.0)."""
        voltage_range = MAX_VOLTAGE - MIN_VOLTAGE
        normalized = [(v - MIN_VOLTAGE) / voltage_range * 2 - 1 for v in ai_values]
        if self.output_format == "int8":
            return [max(-128, min(127, int(round(v * 127)))) for v in normalized]
        return normalized

    def _normalize_di(self, di_values):
        """Convert raw digital readings to requested format."""
        if self.output_format == "int8":
            return [127 if bool(v) else 0 for v in di_values]
        return [float(v) for v in di_values]

    def read(self):
        """
        Read AI and DI channel values.
        :return: (ai_list, di_list)
        """
        try:
            ai_raw = self.task_ai.read()
            di_raw = self.task_di.read()
        except DaqError as e:
            raise RuntimeError(f"Failed to read from NiDAQ: {e}")

        ai_processed = self._normalize_ai(ai_raw)
        di_processed = self._normalize_di(di_raw)

        return ai_processed, di_processed

    def close(self):
        """Stop and close DAQ tasks."""
        for task in (self.task_ai, self.task_di):
            try:
                task.stop()
                task.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


if __name__ == "__main__":
    """
    Test mode: Print active channels in real-time.
    Useful for identifying which physical input corresponds to which channel.
    """
    import time

    THRESHOLD = 0.05  # 5% threshold for analog inputs

    print("NiDAQ Channel Monitor")
    print("=" * 50)
    print("Move joysticks or press buttons to see active channels.")
    print("Press Ctrl+C to exit.\n")

    controller = NiDAQJoysticks(output_format="float")

    try:
        while True:
            ai_values, di_values = controller.read()

            active_channels = []

            # Check analog inputs
            for i, (channel, value) in enumerate(zip(AI_CHANNELS, ai_values)):
                if abs(value) > THRESHOLD:
                    active_channels.append(f"[AI:{i}] {channel}: {value:+.3f}")

            # Check digital inputs
            for i, (channel, value) in enumerate(zip(DI_CHANNELS, di_values)):
                if value > 0.5:  # Button pressed
                    active_channels.append(f"[DI:{i}] {channel}: PRESSED")

            # Print active channels on one line
            if active_channels:
                print("\r" + " | ".join(active_channels) + " " * 20, end="", flush=True)
            else:
                print("\r" + "No active channels" + " " * 50, end="", flush=True)

            time.sleep(0.05)  # 20Hz update rate

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        controller.close()
