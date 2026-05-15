import serial
import time
import logging
import os
import serial.tools.list_ports
import struct
import math
import threading

try:
    from .rt_utils import apply_rt_to_thread, SCHED_FIFO
except ImportError:
    from rt_utils import apply_rt_to_thread, SCHED_FIFO


class USBSerialReader:
    """Read IMU quaternion data from Pico over USB serial."""

    MAX_SENSORS = 4
    FLOATS_PER_SENSOR = 7  # w, x, y, z, gx, gy, gz

    # Binary protocol frame versions
    FRAME_VERSION_DATA = 1
    FRAME_VERSION_CTRL = 2
    FRAME_VERSION_DESC = 3
    FRAME_VERSION_CAL_REPORT = 5

    # Control frame message types (version=2)
    MSG_TYPE_ERR_I2C = 0x05
    MSG_TYPE_ERR_IMU = 0x06
    MSG_TYPE_CAL_WAIT = 0x07
    MSG_TYPE_ERR_CAL = 0x08

    STARTUP_CALIBRATION_DURATION_S = 30.0

    CAL_REPORT_ACCEPTED = 0x01
    CAL_FAIL_INSUFFICIENT_SAMPLES = 0x0001
    CAL_FAIL_GYRO_STD = 0x0002
    CAL_FAIL_ACCEL_STD = 0x0004
    CAL_FAIL_ACCEL_NORM = 0x0008

    def __init__(self, baud_rate=115200, timeout=1.0,
                 log_level: str = "INFO", port: str | None = None, debug: bool = False,
                 verify_checksum: bool = True, heartbeat_timeout: float = 3.0,
                 rt_priority: int = 0,
                 rt_lock_memory: bool = False, rt_cpu_core: int | None = None):
        self.logger = logging.getLogger(f"{__name__}.USBSerialReader")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.baud_rate = baud_rate
        self.timeout = timeout
        self.write_timeout = 1.0
        self.ser = None
        self.port = port
        self._requested_port = port  # remember original for reconnect
        self.debug = debug
        self.verify_checksum = verify_checksum
        self.heartbeat_timeout = heartbeat_timeout
        self._blocking_reads = os.name != "nt"
        self._read_timeout = min(max(timeout, 0.001), 0.02) if self._blocking_reads else 0
        self.num_sensors = 0  # discovered at runtime from first frame
        self.last_timestamp_us = None
        self._imu_descriptors = []  # list of {"bus": int, "addr": int}
        self._descriptor_signature = None
        self._target_sps = 0       # target sample rate reported by firmware
        self._last_data_time = time.time()
        self._connect_time = time.time()
        self._bin_buf = bytearray()
        self._checksum_failures = 0
        self._header_failures = 0
        self._reconnect_count = 0
        self._timestamp_sps = 0.0
        self._host_sps = 0.0
        self._rt_priority = int(rt_priority)
        self._rt_lock_memory = bool(rt_lock_memory)
        self._rt_cpu_core = rt_cpu_core
        self._startup_phase = "disconnected"
        self._calibration_wait_start_time = None
        self._last_calibration_log_time = 0.0
        self._calibration_report = None
        self._last_error_msg_type = None
        self._lock = threading.Lock()
        self._reader_thread = None
        self._reader_stop = threading.Event()
        self._latest_frame = None
        self._latest_frame_seq = 0
        self._last_returned_seq = 0

        # SPS (samples per second) tracking
        self._reset_sps_tracking()

        self.connect()

    # ------------------------------------------------------------------
    # Status API
    # ------------------------------------------------------------------

    def _copy_calibration_report(self):
        report = self._calibration_report
        if report is None:
            return None
        return {
            **report,
            "thresholds": dict(report.get("thresholds", {})),
            "sensors": [dict(sensor) for sensor in report.get("sensors", [])],
        }

    def status(self):
        """Return current connection status as a dict.

        Keys:
            connected (bool): Whether the serial port is open.
            port (str|None): Current serial port name.
            num_sensors (int): Number of IMUs detected in last frame.
            imu_descriptors (list[dict]): Per-IMU info from firmware.
                Each dict: {"bus": int, "addr": int, "label": str}.
                Empty until descriptor frame is received.
            target_sps (int): Target sample rate reported by firmware.
            sps (float): Actual received frames per second.
            last_timestamp_us (int|None): Last sensor timestamp in microseconds.
            checksum_failures (int): Cumulative checksum mismatch count.
            header_failures (int): Cumulative invalid header count.
            reconnect_count (int): Number of automatic reconnections.
            uptime_s (float): Seconds since initial connection.
        """
        connected = self.ser is not None and self.ser.is_open
        with self._lock:
            return {
            "connected": connected,
            "port": self.port,
            "num_sensors": self.num_sensors,
            "imu_descriptors": list(self._imu_descriptors),
            "target_sps": self._target_sps,
            "sps": self._host_sps,
            "host_sps": self._host_sps,
            "timestamp_sps": self._timestamp_sps,
            "last_timestamp_us": self.last_timestamp_us,
            "startup_phase": self._startup_phase,
            "calibration_wait_s": (
                time.time() - self._calibration_wait_start_time
                if self._calibration_wait_start_time is not None else 0.0
            ),
            "calibration_report": self._copy_calibration_report(),
            "checksum_failures": self._checksum_failures,
            "header_failures": self._header_failures,
            "reconnect_count": self._reconnect_count,
            "uptime_s": time.time() - self._connect_time,
            }

    @property
    def connected(self):
        """True if the serial port is open."""
        return self.ser is not None and self.ser.is_open

    @property
    def sps(self):
        """Current host-observed frames-per-second rate."""
        return self._host_sps

    @property
    def host_sps(self):
        """Current host-observed frames-per-second rate."""
        return self._host_sps

    @property
    def timestamp_sps(self):
        """Current frames-per-second rate derived from firmware timestamps."""
        return self._timestamp_sps

    @property
    def target_sps(self):
        """Target sample rate reported by firmware (0 if not yet received)."""
        return self._target_sps

    @property
    def imu_descriptors(self):
        """List of IMU descriptors from firmware.

        Each entry is a dict with:
            bus (int): I2C bus number (0 or 1)
            addr (int): I2C address (e.g. 0x6A, 0x6B)
            label (str): Human-readable label (e.g. "I2C0:0x6A")
        """
        return list(self._imu_descriptors)

    def set_log_level(self, level: str):
        """Update logger level at runtime."""
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def find_pico_port(self):
        """Try to find a likely XIAO RP2040 / Pico port automatically."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.logger.error("No serial ports found.")
            return None

        keywords = ("XIAO", "RP2040", "Pico", "USB Serial", "CDC", "ACM")
        vidpid_markers = ("VID:PID=2E8A:", "VID:PID=2886:")

        def score_port(port_info):
            device = (port_info.device or "").upper()
            desc = (port_info.description or "").upper()
            hwid = (port_info.hwid or "").upper()
            score = 0
            if "TTYACM" in device:
                score += 100
            if "COM" in device:
                score += 60
            if "TTYAMA" in device or "TTYS" in device:
                score -= 100
            if any(k in desc for k in keywords):
                score += 40
            if any(m in hwid for m in vidpid_markers):
                score += 80
            if "USB" in desc or "CDC" in desc:
                score += 20
            return score

        ranked = sorted(ports, key=score_port, reverse=True)

        for port in ranked:
            desc = port.description or ""
            hwid = port.hwid or ""
            if any(k in desc for k in keywords) or any(m in hwid for m in vidpid_markers):
                self.logger.info(f"Found device on port: {port.device}")
                return port.device

        fallback = ranked[0].device
        self.logger.warning(f"No descriptor match; defaulting to {fallback}")
        return fallback

    def connect(self):
        """Connect to the serial port."""
        if not self.port:
            self.port = self.find_pico_port()
        if not self.port:
            raise serial.SerialException("No serial port found")

        # Use non-blocking reads and a bounded write timeout so a wedged CDC
        # endpoint cannot hang the whole process on Windows.
        self.ser = serial.Serial(
            self.port,
            self.baud_rate,
            timeout=self._read_timeout,
            write_timeout=self.write_timeout,
            inter_byte_timeout=0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        if hasattr(self.ser, "set_buffer_size"):
            try:
                if os.name == "nt":
                    self.ser.set_buffer_size(rx_size=512)
            except Exception:
                pass
        self.ser.reset_input_buffer()
        self._bin_buf.clear()
        self._descriptor_signature = None
        self._imu_descriptors = []
        self.num_sensors = 0
        self._target_sps = 0
        self._latest_frame = None
        self._latest_frame_seq = 0
        self._last_returned_seq = 0
        self._startup_phase = "connected"
        self._calibration_wait_start_time = None
        self._last_calibration_log_time = 0.0
        self._calibration_report = None
        self._last_error_msg_type = None
        self._last_data_time = time.time()
        self._connect_time = time.time()
        self._reset_sps_tracking()
        self.logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
        time.sleep(0.1)

    def reconnect(self):
        """Close and re-open connection after a reset or timeout."""
        self._reconnect_count += 1
        self.logger.info(f"Reconnecting (attempt #{self._reconnect_count})...")

        # Close existing connection
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

        # Re-detect port (may have changed after Pico reset)
        self.port = self._requested_port
        time.sleep(0.5)

        try:
            self.connect()
            self.logger.info("Reconnected — waiting for stream")
            return True
        except serial.SerialException as e:
            self.logger.warning(f"Reconnect failed: {e}")
            return False

    def send_command(self, command: str):
        """Legacy no-op: stream-only firmware does not accept runtime commands."""
        self.logger.warning("Ignoring runtime command for stream-only firmware: %s", command)
        return False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _handle_control_msg(self, msg_type):
        """Handle Pico control frames."""
        now = time.time()
        if msg_type == self.MSG_TYPE_CAL_WAIT:
            if self._calibration_wait_start_time is None:
                self._calibration_wait_start_time = now
                self.logger.info(
                    "Pico startup calibration in progress; keep IMUs stationary "
                    "(~%.0fs after power-on)",
                    self.STARTUP_CALIBRATION_DURATION_S,
                )
            elif (now - self._last_calibration_log_time) >= 5.0:
                elapsed = now - self._calibration_wait_start_time
                total = self.STARTUP_CALIBRATION_DURATION_S
                remaining = max(0.0, total - elapsed)
                self.logger.info(
                    "Pico startup calibration still running (%.0fs elapsed, ~%.0fs remaining); keep IMUs stationary",
                    elapsed,
                    remaining,
                )
            self._last_calibration_log_time = now
            self._startup_phase = "calibrating"
            self._last_data_time = now
            return False
        if msg_type == self.MSG_TYPE_ERR_I2C:
            self._startup_phase = "error_i2c"
            if self._last_error_msg_type != msg_type:
                self.logger.error("Pico reported I2C initialization failure")
                self._last_error_msg_type = msg_type
            return False
        if msg_type == self.MSG_TYPE_ERR_IMU:
            self._startup_phase = "error_imu"
            if self._last_error_msg_type != msg_type:
                self.logger.error("Pico reported IMU initialization failure")
                self._last_error_msg_type = msg_type
            return False
        if msg_type == self.MSG_TYPE_ERR_CAL:
            self._startup_phase = "error_calibration"
            if self._last_error_msg_type != msg_type:
                self.logger.error("Pico startup calibration failed; keep robot/IMUs still and power-cycle or reset Pico")
                self._last_error_msg_type = msg_type
            return False
        return False

    def _fill_binary_buffer(self, min_bytes=1):
        """Pull bytes from serial, preferring low-latency reads on POSIX."""
        if not self.ser or not self.ser.is_open:
            return 0

        read_count = 0
        try:
            if self._blocking_reads and len(self._bin_buf) < min_bytes:
                needed = max(1, min_bytes - len(self._bin_buf))
                chunk = self.ser.read(needed)
                if chunk:
                    self._bin_buf.extend(chunk)
                    read_count += len(chunk)

            waiting = self.ser.in_waiting
            if waiting > 0:
                chunk = self.ser.read(waiting)
                if chunk:
                    self._bin_buf.extend(chunk)
                    read_count += len(chunk)
        except (serial.SerialException, OSError):
            return 0

        return read_count

    # ------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------

    def _check_heartbeat(self):
        """Check if data stream is alive. Returns True if OK, False if timed out."""
        if self.heartbeat_timeout <= 0:
            return True

        elapsed = time.time() - self._last_data_time
        if elapsed <= self.heartbeat_timeout:
            return True

        if self._blocking_reads and self.ser and self.ser.is_open:
            try:
                rescue = self.ser.read(1)
            except (serial.SerialException, OSError):
                rescue = b""
            if rescue:
                self._bin_buf.extend(rescue)
                self._last_data_time = time.time()
                return True

        self.logger.warning(f"No data for {elapsed:.1f}s — connection may be lost")
        return False

    def _update_sps(self):
        """Update host-observed and timestamp-derived SPS counters."""
        now = time.perf_counter()
        if self._prime_sps:
            self._host_frame_count = 0
            self._host_sps = 0.0
            self._host_sps_time = now
            self._ts_frame_count = 0
            self._timestamp_sps = 0.0
            self._ts_window_start_us = self.last_timestamp_us
            self._ts_prev_us = self.last_timestamp_us
            self._prime_sps = False
            return

        self._host_frame_count += 1
        elapsed = now - self._host_sps_time
        if elapsed >= 1.0:
            self._host_sps = self._host_frame_count / elapsed
            self._host_frame_count = 0
            self._host_sps_time = now

        ts_us = self.last_timestamp_us
        if ts_us is None:
            return

        if self._ts_prev_us is None or self._ts_window_start_us is None or ts_us < self._ts_prev_us:
            self._ts_frame_count = 0
            self._timestamp_sps = 0.0
            self._ts_window_start_us = ts_us
            self._ts_prev_us = ts_us
            return

        self._ts_frame_count += 1
        elapsed_ts_us = ts_us - self._ts_window_start_us
        if elapsed_ts_us >= 1_000_000:
            self._timestamp_sps = self._ts_frame_count * 1_000_000.0 / elapsed_ts_us
            self._ts_frame_count = 0
            self._ts_window_start_us = ts_us

        self._ts_prev_us = ts_us

    def _reset_sps_tracking(self):
        """Restart SPS measurement after connect or resync."""
        self._host_frame_count = 0
        self._host_sps = 0.0
        self._host_sps_time = time.perf_counter()
        self._ts_frame_count = 0
        self._timestamp_sps = 0.0
        self._ts_window_start_us = None
        self._ts_prev_us = None
        self._prime_sps = True

    def _parse_descriptor_payload(self, start, desc_count):
        """Decode descriptor payload from the current binary buffer."""
        target_sps = struct.unpack_from("<H", self._bin_buf, start + 4)[0]
        descriptors = []
        for i in range(desc_count):
            bus = self._bin_buf[start + 6 + i * 2]
            addr = self._bin_buf[start + 6 + i * 2 + 1]
            descriptors.append({
                "bus": bus,
                "addr": addr,
                "label": f"I2C{bus}:0x{addr:02X}",
            })
        return target_sps, descriptors

    def _apply_descriptor(self, desc_count, target_sps, descriptors):
        """Store descriptor data and log only when it changes."""
        signature = (target_sps, tuple((d["bus"], d["addr"]) for d in descriptors))
        changed = signature != self._descriptor_signature

        with self._lock:
            self._target_sps = target_sps
            self._imu_descriptors = descriptors
            self.num_sensors = desc_count

        if changed:
            self._descriptor_signature = signature
            labels = ", ".join(d["label"] for d in descriptors)
            self.logger.info(
                "IMU descriptor: %d sensor(s) @ %d Hz - %s",
                desc_count,
                target_sps,
                labels,
            )

    def _calibration_failure_names(self, flags):
        names = []
        if flags & self.CAL_FAIL_INSUFFICIENT_SAMPLES:
            names.append("insufficient_samples")
        if flags & self.CAL_FAIL_GYRO_STD:
            names.append("gyro_std")
        if flags & self.CAL_FAIL_ACCEL_STD:
            names.append("accel_std")
        if flags & self.CAL_FAIL_ACCEL_NORM:
            names.append("accel_norm")
        return names

    def _calibration_report_frame_len(self, sensor_count):
        return 28 + sensor_count * 72

    def _parse_calibration_report_payload(self, start, sensor_count):
        offset = start + 4
        duration_ms = struct.unpack_from("<I", self._bin_buf, offset)[0]
        offset += 4
        flags = self._bin_buf[offset]
        offset += 2  # flags + reserved
        gyro_std_limit, accel_std_limit, accel_norm_min, accel_norm_max = struct.unpack_from(
            "<ffff", self._bin_buf, offset
        )
        offset += 16

        sensors = []
        for index in range(sensor_count):
            sample_count, read_error_count, failure_flags = struct.unpack_from("<IIH", self._bin_buf, offset)
            offset += 12  # sample_count + read_error_count + failure_flags + reserved
            values = struct.unpack_from("<fffffffffffffff", self._bin_buf, offset)
            offset += 60
            sensors.append({
                "index": index,
                "sample_count": sample_count,
                "read_error_count": read_error_count,
                "failure_flags": failure_flags,
                "failures": self._calibration_failure_names(failure_flags),
                "gyro_bias_dps": values[0:3],
                "gyro_std_dps": values[3:6],
                "accel_mean_g": values[6:9],
                "accel_std_g": values[9:12],
                "accel_norm_mean_g": values[12],
                "accel_norm_min_g": values[13],
                "accel_norm_max_g": values[14],
            })

        return {
            "accepted": bool(flags & self.CAL_REPORT_ACCEPTED),
            "duration_s": duration_ms / 1000.0,
            "flags": flags,
            "thresholds": {
                "gyro_std_dps": gyro_std_limit,
                "accel_std_g": accel_std_limit,
                "accel_norm_min_g": accel_norm_min,
                "accel_norm_max_g": accel_norm_max,
            },
            "sensors": sensors,
        }

    def _apply_calibration_report(self, report):
        with self._lock:
            already_logged = self._calibration_report is not None
            self._calibration_report = report

        if already_logged:
            return

        state = "accepted" if report["accepted"] else "rejected"
        self.logger.info("IMU startup calibration %s after %.1fs", state, report["duration_s"])
        for sensor in report["sensors"]:
            failure_text = ",".join(sensor["failures"]) if sensor["failures"] else "ok"
            self.logger.info(
                "  cal[%d] samples=%d read_errors=%d gyro_bias=%s dps gyro_std=%s dps accel_std=%s g accel_norm=%.4f/%.4f/%.4f result=%s",
                sensor["index"],
                sensor["sample_count"],
                sensor["read_error_count"],
                tuple(round(v, 4) for v in sensor["gyro_bias_dps"]),
                tuple(round(v, 4) for v in sensor["gyro_std_dps"]),
                tuple(round(v, 5) for v in sensor["accel_std_g"]),
                sensor["accel_norm_min_g"],
                sensor["accel_norm_mean_g"],
                sensor["accel_norm_max_g"],
                failure_text,
            )

    def _read_binary_frame(self, refill=True):
        """Parse binary IMU frame from serial buffer."""
        if not self.ser or not self.ser.is_open:
            return None

        # Read available data into buffer
        if refill:
            self._fill_binary_buffer(min_bytes=8)

        # Frame: 0xAA 0x55 | ver(1) | count(1) | ts_us(4) | payload | checksum(2)
        while True:
            if len(self._bin_buf) < 4:
                return None

            # Find sync bytes
            sync = self._bin_buf.find(b"\xAA\x55")
            if sync < 0:
                # Keep a trailing 0xAA in case sync splits across reads
                if self._bin_buf.endswith(b"\xAA"):
                    self._bin_buf[:] = b"\xAA"
                else:
                    self._bin_buf.clear()
                return None
            if sync > 0:
                del self._bin_buf[:sync]
                if refill:
                    self._fill_binary_buffer(min_bytes=8)
                if len(self._bin_buf) < 8:
                    return None

            # Parse header
            if len(self._bin_buf) < 8:
                if refill:
                    self._fill_binary_buffer(min_bytes=8)
            if len(self._bin_buf) < 8:
                return None

            version = self._bin_buf[2]

            # Handle control frames (version=2)
            if version == self.FRAME_VERSION_CTRL:
                if refill:
                    self._fill_binary_buffer(min_bytes=6)
                if len(self._bin_buf) < 6:
                    return None
                msg_type = self._bin_buf[3]
                expected_cs = struct.unpack_from("<H", self._bin_buf, 4)[0]
                calc_cs = (version + msg_type) & 0xFFFF
                if calc_cs == expected_cs:
                    self._handle_control_msg(msg_type)
                del self._bin_buf[:6]
                continue

            if version == self.FRAME_VERSION_CAL_REPORT:
                if refill:
                    self._fill_binary_buffer(min_bytes=8)
                if len(self._bin_buf) < 8:
                    return None
                sensor_count = self._bin_buf[3]
                if sensor_count < 1 or sensor_count > self.MAX_SENSORS:
                    del self._bin_buf[:1]
                    continue
                frame_len = self._calibration_report_frame_len(sensor_count)
                if refill:
                    self._fill_binary_buffer(min_bytes=frame_len)
                if len(self._bin_buf) < frame_len:
                    return None
                checksum_offset = frame_len - 2
                expected = struct.unpack_from("<H", self._bin_buf, checksum_offset)[0]
                calc = sum(self._bin_buf[2:checksum_offset]) & 0xFFFF
                if calc == expected:
                    report = self._parse_calibration_report_payload(0, sensor_count)
                    self._apply_calibration_report(report)
                else:
                    self._checksum_failures += 1
                del self._bin_buf[:frame_len]
                continue

            # Handle IMU descriptor frame (version=3)
            if version == self.FRAME_VERSION_DESC:
                if refill:
                    self._fill_binary_buffer(min_bytes=8)
                if len(self._bin_buf) < 8:
                    return None
                desc_count = self._bin_buf[3]
                if desc_count < 1 or desc_count > self.MAX_SENSORS:
                    del self._bin_buf[:1]
                    continue
                # Frame: sync(2) + ver(1) + count(1) + sps(2) + N*2 + checksum(2)
                desc_frame_len = 4 + 2 + desc_count * 2 + 2
                if refill:
                    self._fill_binary_buffer(min_bytes=desc_frame_len)
                if len(self._bin_buf) < desc_frame_len:
                    return None
                # Verify checksum
                cs_off = desc_frame_len - 2
                expected_cs = struct.unpack_from("<H", self._bin_buf, cs_off)[0]
                calc_cs = sum(self._bin_buf[2:cs_off]) & 0xFFFF
                if calc_cs == expected_cs:
                    target_sps, descriptors = self._parse_descriptor_payload(0, desc_count)
                    self._apply_descriptor(desc_count, target_sps, descriptors)
                del self._bin_buf[:desc_frame_len]
                continue

            sensor_count = self._bin_buf[3]

            if version != self.FRAME_VERSION_DATA or sensor_count < 1 or sensor_count > self.MAX_SENSORS:
                # Drop one byte and rescan for sync
                self._header_failures += 1
                del self._bin_buf[0:1]
                continue

            # Calculate frame length
            payload_len = 4 + sensor_count * self.FLOATS_PER_SENSOR * 4
            frame_len = 2 + 1 + 1 + payload_len + 2

            if refill:
                self._fill_binary_buffer(min_bytes=frame_len)
            if len(self._bin_buf) < frame_len:
                return None

            # Verify checksum
            checksum_offset = 2 + 1 + 1 + payload_len
            expected = struct.unpack_from("<H", self._bin_buf, checksum_offset)[0]
            calc = sum(self._bin_buf[2:checksum_offset]) & 0xFFFF

            if calc != expected:
                self._checksum_failures += 1
                if self.debug and (self._checksum_failures % 50 == 0):
                    self.logger.warning(
                        "Checksum mismatch (count=%d): calc=0x%04X expected=0x%04X",
                        self._checksum_failures,
                        calc,
                        expected,
                    )
                if not self.verify_checksum:
                    break
                # Drop one byte and rescan for sync
                del self._bin_buf[0:1]
                continue

            # Parse timestamp
            ts = struct.unpack_from("<I", self._bin_buf, 4)[0]
            ts = int(ts)

            # Parse sensor data
            imu_data = []
            offset = 8
            for _ in range(sensor_count):
                vals = struct.unpack_from("<fffffff", self._bin_buf, offset)
                imu_data.append(list(vals))
                offset += 28

            del self._bin_buf[:frame_len]
            self.last_timestamp_us = ts
            self._last_data_time = time.time()
            self._startup_phase = "streaming"
            self._update_sps()
            sensor_count_changed = False
            with self._lock:
                if self.num_sensors != sensor_count:
                    self.num_sensors = sensor_count
                    sensor_count_changed = True
            if sensor_count_changed:
                self.logger.info(f"Receiving data from {sensor_count} IMU(s)")
            return imu_data

    def _reader_loop(self):
        if self._rt_priority > 0 or self._rt_lock_memory or self._rt_cpu_core is not None:
            cpu_affinity = None if self._rt_cpu_core is None else {int(self._rt_cpu_core)}
            success = apply_rt_to_thread(
                priority=self._rt_priority,
                policy=SCHED_FIFO,
                lock_memory=self._rt_lock_memory,
                cpu_affinity=cpu_affinity,
                quiet=False,
            )
            if success:
                details = []
                if self._rt_priority > 0:
                    details.append(f"SCHED_FIFO-{self._rt_priority}")
                if self._rt_lock_memory:
                    details.append("mlockall")
                if self._rt_cpu_core is not None:
                    details.append(f"core {self._rt_cpu_core}")
                self.logger.info("USB reader thread: applied %s", ", ".join(details) if details else "RT settings")
            else:
                self.logger.warning("USB reader thread: Failed to apply requested RT settings")

        while not self._reader_stop.is_set():
            try:
                frame = self.read_imus(auto_reconnect=True)
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_frame_seq += 1
                elif self._blocking_reads:
                    time.sleep(0.0005)
            except Exception as exc:
                self.logger.warning("Background IMU reader stopped: %s", exc)
                time.sleep(0.05)

    def start_background_reader(self):
        """Start a latest-only background drain thread."""
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return True
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def stop_background_reader(self):
        """Stop the background reader thread if running."""
        self._reader_stop.set()
        thread = self._reader_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._reader_thread = None

    def get_latest_imus(self, only_new=True):
        """Return the newest frame captured by the background reader."""
        with self._lock:
            if self._latest_frame is None:
                return None
            if only_new and self._latest_frame_seq == self._last_returned_seq:
                return None
            self._last_returned_seq = self._latest_frame_seq
            frame = self._latest_frame  # grab reference; copy happens outside lock
        return [list(pkt) for pkt in frame]

    def read_imus(self, auto_reconnect=True):
        """
        Read IMU data (returns latest frame, discarding any backlog).

        Args:
            auto_reconnect: If True, automatically reconnect on timeout/error.

        Returns:
            List of N arrays, each containing [w, x, y, z, gx, gy, gz]
            (quaternion + gyro in dps), or None if no data available.
        """
        if self.debug and self.ser and self.ser.in_waiting > 0:
            peek = self.ser.read(self.ser.in_waiting)
            if peek:
                self.logger.info("RX(data): %r", peek)
            # Put bytes back into binary buffer for parsing
            self._bin_buf.extend(peek)

        # Refill once, then drain only the backlog already buffered. Otherwise a
        # continuous stream can keep this call busy forever on POSIX.
        latest_data = self._read_binary_frame(refill=True)
        while True:
            data = self._read_binary_frame(refill=False)
            if data is None:
                break
            latest_data = data

        # Heartbeat check: reconnect if no data for too long
        if latest_data is None and auto_reconnect and not self._check_heartbeat():
            self.reconnect()

        return latest_data

    def quaternion_to_pitch(self, w, x, y, z):
        """Extract pitch angle from quaternion using gravity vector projection (degrees).

        Computes pitch as the angle of the gravity vector projected through the
        quaternion rotation, matching the firmware's gravity-based extraction.
        """
        # Rotate world gravity [0,0,-1] by conjugate of quaternion to get
        # gravity in sensor frame, then compute pitch = atan2(-gx, gz)
        gx = 2.0 * (x * z - w * y)
        gz = 1.0 - 2.0 * (x * x + y * y)
        pitch = math.atan2(-gx, gz) * 180.0 / math.pi
        return pitch

    def iter_stream(self, sleep_s=0.001):
        """Generator that yields parsed IMU data as it arrives."""
        try:
            while True:
                data = self.read_imus()
                if data is not None:
                    yield data
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.close()

    def close(self):
        """Close serial connection."""
        try:
            self.stop_background_reader()
        except Exception:
            pass
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Connection closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read 1-4 IMU quaternions over USB serial")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--debug", action="store_true", help="Print raw serial data")
    parser.add_argument("--no-checksum", action="store_true",
                        help="Skip checksum verification (debug only)")
    parser.add_argument("--heartbeat", type=float, default=3.0,
                        help="Reconnect if no data for this many seconds (0=disable)")
    parser.add_argument("--poll-sleep-ms", type=float,
                        default=0.0 if os.name != "nt" else 1.0,
                        help="Host loop sleep between polls in ms (default: 0 on Linux, 1 on Windows)")
    args = parser.parse_args()

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        debug=args.debug,
        verify_checksum=not args.no_checksum,
        heartbeat_timeout=args.heartbeat,
    )

    # Wait briefly for descriptor frame to arrive with first data
    print(f"Connecting... (Ctrl+C to stop)")
    desc_shown = False
    t0 = time.time()
    try:
        poll_sleep_s = max(0.0, args.poll_sleep_ms / 1000.0)
        while True:
            data = reader.read_imus()
            if data is not None:
                # Show descriptor info once when first received
                if not desc_shown and reader.imu_descriptors:
                    desc_shown = True
                    descs = reader.imu_descriptors
                    print(f"\n--- IMU Descriptor ---")
                    print(f"  Sensors:    {reader.num_sensors}")
                    print(f"  Target SPS: {reader.target_sps} Hz")
                    for i, d in enumerate(descs):
                        print(f"  [{i}] {d['label']}  (bus={d['bus']}, addr=0x{d['addr']:02X})")
                    print(f"----------------------\n")
                if time.time() - t0 >= 1.0:
                    st = reader.status()
                    # Build per-IMU pitch readout with labels
                    descs = reader.imu_descriptors
                    pitches = []
                    for i, imu in enumerate(data):
                        w, x, y, z = imu[0:4]
                        pitch = reader.quaternion_to_pitch(w, x, y, z)
                        label = descs[i]["label"] if i < len(descs) else f"IMU{i}"
                        pitches.append(f"{label}:{pitch:+6.1f}")
                    host_sps_str = f"{st['host_sps']:.0f}"
                    ts_sps_str = f"{st['timestamp_sps']:.0f}"
                    if st['target_sps']:
                        host_sps_str += f"/{st['target_sps']}"
                        ts_sps_str += f"/{st['target_sps']}"
                    print(
                        f"Host={host_sps_str:>7s} Hz  FW={ts_sps_str:>7s} Hz  "
                        f"{' | '.join(pitches)}"
                    )
                    t0 = time.time()
            if poll_sleep_s > 0:
                time.sleep(poll_sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
