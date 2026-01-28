"""
Lightweight loop performance tracker using Welford's online algorithm.

Tracks loop timing, processing time, and headroom with O(1) space complexity.
Designed for real-time control loops with minimal overhead when disabled.

TODO: Consider null-object pattern for true zero overhead when disabled?
      Currently, method calls still happen (~20-50ns each) but exit immediately
      after boolean check. Could instead set tracker=None and guard with
      `if self._tracker:` in callers. Savings ~20ns/call - probably negligible
      at typical control rates but worth measuring if loop timing is tight.

Usage:
    tracker = LoopPerfTracker(enabled=True)

    while running:
        tracker.tick_start()
        # ... do work ...
        tracker.tick_end(target_period_s=0.01)  # 100 Hz target

    stats = tracker.get_stats()
"""

import threading
import time
from typing import Dict, Any, Optional

__all__ = ['LoopPerfTracker', 'IntervalTracker', 'ControlLoopPerfTracker']


class LoopPerfTracker:
    """Lightweight loop performance tracker using Welford's online algorithm.

    Tracks:
    - Loop interval (time between consecutive tick_start calls)
    - Processing time (duration from tick_start to tick_end)
    - Headroom (target_period - processing_time)
    - Jitter (std deviation of loop interval)
    - Timing violations (processing time > target period)

    Uses __slots__ for minimal memory footprint. Thread-safe with a single lock.
    Cost when disabled: single boolean check per tick (~1ns).
    """

    __slots__ = (
        '_enabled', '_lock',
        # Loop interval stats (Welford)
        '_loop_n', '_loop_mean', '_loop_m2', '_loop_min', '_loop_max',
        # Processing time stats (Welford)
        '_proc_n', '_proc_mean', '_proc_m2', '_proc_min', '_proc_max',
        # Headroom stats (Welford)
        '_head_n', '_head_mean', '_head_m2', '_head_min', '_head_max',
        # Rate measurement (windowed)
        '_rate_count', '_rate_window_start', '_hz',
        # Violation tracking
        '_violation_count', '_total_count',
        # Timing state
        '_last_tick_start', '_current_tick_start',
    )

    def __init__(self, enabled: bool = False):
        """Initialize the performance tracker.

        Args:
            enabled: Whether tracking is active. When False, tick_start/tick_end
                     are essentially no-ops (single boolean check).
        """
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._reset_internal()

    def _reset_internal(self) -> None:
        """Reset all internal state (called under lock or at init)."""
        # Loop interval
        self._loop_n = 0
        self._loop_mean = 0.0
        self._loop_m2 = 0.0
        self._loop_min = float('inf')
        self._loop_max = 0.0
        # Processing time
        self._proc_n = 0
        self._proc_mean = 0.0
        self._proc_m2 = 0.0
        self._proc_min = float('inf')
        self._proc_max = 0.0
        # Headroom
        self._head_n = 0
        self._head_mean = 0.0
        self._head_m2 = 0.0
        self._head_min = float('inf')
        self._head_max = float('-inf')  # Can be negative
        # Rate
        self._rate_count = 0
        self._rate_window_start = time.perf_counter()
        self._hz = 0.0
        # Violations
        self._violation_count = 0
        self._total_count = 0
        # Timing
        self._last_tick_start: Optional[float] = None
        self._current_tick_start: Optional[float] = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable tracking at runtime."""
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._reset_internal()

    def tick_start(self) -> None:
        """Call at the start of each loop iteration.

        Records the loop interval (time since last tick_start).
        """
        if not self._enabled:
            return

        now = time.perf_counter()

        # Update rate counter
        self._rate_count += 1
        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self._hz = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_window_start = now

        # Track loop interval
        if self._last_tick_start is not None:
            interval = now - self._last_tick_start
            if interval > 0:
                with self._lock:
                    self._loop_n += 1
                    delta = interval - self._loop_mean
                    self._loop_mean += delta / self._loop_n
                    self._loop_m2 += delta * (interval - self._loop_mean)
                    if interval < self._loop_min:
                        self._loop_min = interval
                    if interval > self._loop_max:
                        self._loop_max = interval

        self._last_tick_start = now
        self._current_tick_start = now

    def tick_end(self, target_period_s: float = 0.0) -> None:
        """Call at the end of each loop iteration (after processing, before sleep).

        Args:
            target_period_s: Target loop period in seconds. Used to compute
                             headroom and detect timing violations.
        """
        if not self._enabled:
            return

        if self._current_tick_start is None:
            return

        now = time.perf_counter()
        proc_time = now - self._current_tick_start
        headroom = target_period_s - proc_time if target_period_s > 0 else 0.0

        with self._lock:
            self._total_count += 1

            # Processing time stats
            self._proc_n += 1
            delta = proc_time - self._proc_mean
            self._proc_mean += delta / self._proc_n
            self._proc_m2 += delta * (proc_time - self._proc_mean)
            if proc_time < self._proc_min:
                self._proc_min = proc_time
            if proc_time > self._proc_max:
                self._proc_max = proc_time

            # Headroom stats
            if target_period_s > 0:
                self._head_n += 1
                hdelta = headroom - self._head_mean
                self._head_mean += hdelta / self._head_n
                self._head_m2 += hdelta * (headroom - self._head_mean)
                if headroom < self._head_min:
                    self._head_min = headroom
                if headroom > self._head_max:
                    self._head_max = headroom

                # Violation tracking
                if proc_time > target_period_s:
                    self._violation_count += 1

        self._current_tick_start = None

    def record_interval(self, interval_s: float) -> None:
        """Manually record a loop interval (alternative to tick_start for external timing).

        Useful when the caller already has timing data from another source.
        """
        if not self._enabled or interval_s <= 0:
            return

        with self._lock:
            self._loop_n += 1
            delta = interval_s - self._loop_mean
            self._loop_mean += delta / self._loop_n
            self._loop_m2 += delta * (interval_s - self._loop_mean)
            if interval_s < self._loop_min:
                self._loop_min = interval_s
            if interval_s > self._loop_max:
                self._loop_max = interval_s

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics.

        Returns:
            Dictionary with timing stats in milliseconds:
            - hz: Measured loop frequency
            - loop_avg_ms, loop_std_ms, loop_min_ms, loop_max_ms
            - proc_avg_ms, proc_std_ms, proc_min_ms, proc_max_ms
            - headroom_avg_ms, headroom_min_ms, headroom_max_ms
            - violation_pct: Percentage of iterations exceeding target period
            - samples: Number of samples collected
        """
        if not self._enabled:
            return {}

        with self._lock:
            def compute_std(n: int, m2: float) -> float:
                if n > 1:
                    return (m2 / (n - 1)) ** 0.5
                return 0.0

            def safe_min(val: float) -> float:
                return 0.0 if val == float('inf') else val

            def safe_max_neg(val: float) -> float:
                return 0.0 if val == float('-inf') else val

            loop_std = compute_std(self._loop_n, self._loop_m2)
            proc_std = compute_std(self._proc_n, self._proc_m2)
            head_std = compute_std(self._head_n, self._head_m2)

            violation_pct = 0.0
            if self._total_count > 0:
                violation_pct = (self._violation_count / self._total_count) * 100.0

            return {
                'hz': float(self._hz),
                # Loop interval
                'loop_avg_ms': float(self._loop_mean * 1000.0),
                'loop_std_ms': float(loop_std * 1000.0),
                'loop_min_ms': float(safe_min(self._loop_min) * 1000.0),
                'loop_max_ms': float(self._loop_max * 1000.0),
                # Processing time
                'proc_avg_ms': float(self._proc_mean * 1000.0),
                'proc_std_ms': float(proc_std * 1000.0),
                'proc_min_ms': float(safe_min(self._proc_min) * 1000.0),
                'proc_max_ms': float(self._proc_max * 1000.0),
                # Headroom
                'headroom_avg_ms': float(self._head_mean * 1000.0),
                'headroom_std_ms': float(head_std * 1000.0),
                'headroom_min_ms': float(safe_max_neg(self._head_min) * 1000.0),
                'headroom_max_ms': float(safe_max_neg(self._head_max) * 1000.0),
                # Violations
                'violation_pct': float(violation_pct),
                'violation_count': int(self._violation_count),
                # Sample counts
                'samples': int(self._loop_n),
            }


class IntervalTracker:
    """Lightweight interval tracker for sample-based measurements (IMU, ADC, etc.).

    Simpler than LoopPerfTracker - just tracks intervals between samples.
    Uses Welford's algorithm for O(1) space.
    """

    __slots__ = (
        '_enabled', '_lock',
        '_n', '_mean', '_m2', '_min', '_max',
        '_rate_count', '_rate_window_start', '_hz',
        '_last_ts',
    )

    def __init__(self, enabled: bool = False):
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._reset_internal()

    def _reset_internal(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float('inf')
        self._max = 0.0
        self._rate_count = 0
        self._rate_window_start = time.perf_counter()
        self._hz = 0.0
        self._last_ts: Optional[float] = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        with self._lock:
            self._reset_internal()

    def record_sample(self, timestamp: Optional[float] = None) -> None:
        """Record a sample arrival. Uses current time if timestamp not provided."""
        if not self._enabled:
            return

        now = timestamp if timestamp is not None else time.perf_counter()

        # Rate tracking
        self._rate_count += 1
        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self._hz = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_window_start = now

        # Interval tracking
        if self._last_ts is not None:
            interval = now - self._last_ts
            if interval > 0:
                with self._lock:
                    self._n += 1
                    delta = interval - self._mean
                    self._mean += delta / self._n
                    self._m2 += delta * (interval - self._mean)
                    if interval < self._min:
                        self._min = interval
                    if interval > self._max:
                        self._max = interval

        self._last_ts = now

    def record_interval(self, interval_s: float) -> None:
        """Manually record an interval (alternative to record_sample for external timing).

        Useful when the caller already has timing data from another source.
        """
        if not self._enabled or interval_s <= 0:
            return

        now = time.perf_counter()

        # Rate tracking
        self._rate_count += 1
        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self._hz = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_window_start = now

        # Interval tracking with provided value
        with self._lock:
            self._n += 1
            delta = interval_s - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (interval_s - self._mean)
            if interval_s < self._min:
                self._min = interval_s
            if interval_s > self._max:
                self._max = interval_s

    def get_stats(self) -> Dict[str, Any]:
        """Get interval statistics in milliseconds."""
        if not self._enabled:
            return {}

        with self._lock:
            std = 0.0
            if self._n > 1:
                std = (self._m2 / (self._n - 1)) ** 0.5

            return {
                'hz': float(self._hz),
                'avg_interval_ms': float(self._mean * 1000.0),
                'std_interval_ms': float(std * 1000.0),
                'min_interval_ms': float(self._min * 1000.0) if self._min != float('inf') else 0.0,
                'max_interval_ms': float(self._max * 1000.0),
                'samples': int(self._n),
            }


class ControlLoopPerfTracker:
    """Performance tracker for control loops with stage timing and percentile support.

    Uses a ring buffer for recent samples to enable percentile calculations (p95, p99).
    Tracks named stages (e.g., sensor, ik, pwm) in addition to overall loop timing.

    Usage:
        tracker = ControlLoopPerfTracker(enabled=True, target_hz=100.0)

        while running:
            tracker.loop_start()

            tracker.stage_start('sensor')
            read_sensors()
            tracker.stage_end('sensor')

            tracker.stage_start('compute')
            compute()
            tracker.stage_end('compute')

            tracker.loop_end()

        stats = tracker.get_stats()
    """

    def __init__(self, enabled: bool = False, target_hz: float = 100.0, buffer_size: int = 1000):
        """Initialize the control loop tracker.

        Args:
            enabled: Whether tracking is active
            target_hz: Target loop frequency for headroom/violation calculations
            buffer_size: Size of ring buffer for percentile calculations
        """
        self._enabled = bool(enabled)
        self._target_period = 1.0 / max(1.0, target_hz)
        self._buffer_size = max(100, buffer_size)
        self._lock = threading.Lock()

        # Ring buffers for percentile support
        self._loop_times: list = []
        self._compute_times: list = []

        # Stage timing (dict of stage_name -> list of durations)
        self._stage_times: Dict[str, list] = {}
        self._stage_starts: Dict[str, float] = {}

        # Counters
        self._violation_count = 0
        self._loop_count = 0

        # Rate tracking
        self._rate_count = 0
        self._rate_window_start = time.perf_counter()
        self._hz = 0.0

        # Timing state
        self._last_loop_start: Optional[float] = None
        self._current_loop_start: Optional[float] = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable tracking at runtime."""
        self._enabled = bool(enabled)

    def set_target_hz(self, hz: float) -> None:
        """Update target frequency."""
        self._target_period = 1.0 / max(1.0, hz)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._loop_times.clear()
            self._compute_times.clear()
            self._stage_times.clear()
            self._stage_starts.clear()
            self._violation_count = 0
            self._loop_count = 0
            self._rate_count = 0
            self._rate_window_start = time.perf_counter()
            self._hz = 0.0
            self._last_loop_start = None
            self._current_loop_start = None

    def loop_start(self) -> None:
        """Call at the start of each control loop iteration."""
        if not self._enabled:
            return

        now = time.perf_counter()

        # Rate tracking
        self._rate_count += 1
        elapsed = now - self._rate_window_start
        if elapsed >= 0.5:
            self._hz = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_window_start = now

        # Loop interval tracking
        if self._last_loop_start is not None:
            interval = now - self._last_loop_start
            if interval > 0:
                with self._lock:
                    self._loop_times.append(interval)
                    if len(self._loop_times) > self._buffer_size:
                        self._loop_times.pop(0)

        self._last_loop_start = now
        self._current_loop_start = now

    def loop_end(self) -> None:
        """Call at the end of each control loop iteration (after all stages, before sleep)."""
        if not self._enabled or self._current_loop_start is None:
            return

        now = time.perf_counter()
        compute_time = now - self._current_loop_start

        with self._lock:
            self._compute_times.append(compute_time)
            if len(self._compute_times) > self._buffer_size:
                self._compute_times.pop(0)

            self._loop_count += 1
            if compute_time > self._target_period:
                self._violation_count += 1

        self._current_loop_start = None

    def stage_start(self, stage_name: str) -> None:
        """Mark the start of a named stage."""
        if not self._enabled:
            return
        self._stage_starts[stage_name] = time.perf_counter()

    def stage_end(self, stage_name: str) -> None:
        """Mark the end of a named stage and record duration."""
        if not self._enabled:
            return

        start = self._stage_starts.get(stage_name)
        if start is None:
            return

        duration = time.perf_counter() - start
        with self._lock:
            if stage_name not in self._stage_times:
                self._stage_times[stage_name] = []
            self._stage_times[stage_name].append(duration)
            if len(self._stage_times[stage_name]) > self._buffer_size:
                self._stage_times[stage_name].pop(0)

        del self._stage_starts[stage_name]

    def record_stage(self, stage_name: str, duration_s: float) -> None:
        """Directly record a stage duration (alternative to stage_start/stage_end)."""
        if not self._enabled:
            return

        with self._lock:
            if stage_name not in self._stage_times:
                self._stage_times[stage_name] = []
            self._stage_times[stage_name].append(duration_s)
            if len(self._stage_times[stage_name]) > self._buffer_size:
                self._stage_times[stage_name].pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.

        Returns:
            Dictionary with:
            - hz: Measured loop frequency
            - loop_avg_ms, loop_std_ms, loop_min_ms, loop_max_ms
            - loop_p95_ms, loop_p99_ms: Percentile latencies
            - compute_avg_ms, compute_std_ms, compute_min_ms, compute_max_ms
            - headroom_avg_ms: Average time remaining before target period
            - cpu_usage_pct: Compute time as percentage of target period
            - violation_pct: Percentage of loops exceeding target period
            - stages: Dict of stage_name -> {avg_ms, min_ms, max_ms}
            - samples: Number of samples in buffer
        """
        if not self._enabled:
            return {}

        with self._lock:
            if not self._loop_times:
                return {
                    'hz': 0.0,
                    'loop_avg_ms': 0.0, 'loop_std_ms': 0.0,
                    'loop_min_ms': 0.0, 'loop_max_ms': 0.0,
                    'loop_p95_ms': 0.0, 'loop_p99_ms': 0.0,
                    'compute_avg_ms': 0.0, 'compute_std_ms': 0.0,
                    'compute_min_ms': 0.0, 'compute_max_ms': 0.0,
                    'headroom_avg_ms': 0.0, 'cpu_usage_pct': 0.0,
                    'violation_pct': 0.0, 'violation_count': 0,
                    'stages': {},
                    'samples': 0,
                }

            # Convert to ms for calculations
            loop_ms = [t * 1000.0 for t in self._loop_times]
            compute_ms = [t * 1000.0 for t in self._compute_times] if self._compute_times else [0.0]

            # Basic stats
            loop_avg = sum(loop_ms) / len(loop_ms)
            loop_std = (sum((t - loop_avg) ** 2 for t in loop_ms) / max(1, len(loop_ms) - 1)) ** 0.5
            compute_avg = sum(compute_ms) / len(compute_ms)
            compute_std = (sum((t - compute_avg) ** 2 for t in compute_ms) / max(1, len(compute_ms) - 1)) ** 0.5

            # Percentiles (sorted copy)
            sorted_loop = sorted(loop_ms)
            p95_idx = int(len(sorted_loop) * 0.95)
            p99_idx = int(len(sorted_loop) * 0.99)
            loop_p95 = sorted_loop[min(p95_idx, len(sorted_loop) - 1)]
            loop_p99 = sorted_loop[min(p99_idx, len(sorted_loop) - 1)]

            # Headroom and CPU usage
            target_ms = self._target_period * 1000.0
            headroom_avg = target_ms - compute_avg
            cpu_usage_pct = (compute_avg / target_ms) * 100.0 if target_ms > 0 else 0.0

            # Violation percentage
            violation_pct = (self._violation_count / max(1, self._loop_count)) * 100.0

            # Stage stats
            stages = {}
            for name, times in self._stage_times.items():
                if times:
                    times_ms = [t * 1000.0 for t in times]
                    stages[name] = {
                        'avg_ms': sum(times_ms) / len(times_ms),
                        'min_ms': min(times_ms),
                        'max_ms': max(times_ms),
                    }

            return {
                'hz': float(self._hz),
                'loop_avg_ms': float(loop_avg),
                'loop_std_ms': float(loop_std),
                'loop_min_ms': float(min(loop_ms)),
                'loop_max_ms': float(max(loop_ms)),
                'loop_p95_ms': float(loop_p95),
                'loop_p99_ms': float(loop_p99),
                'compute_avg_ms': float(compute_avg),
                'compute_std_ms': float(compute_std),
                'compute_min_ms': float(min(compute_ms)),
                'compute_max_ms': float(max(compute_ms)),
                'headroom_avg_ms': float(headroom_avg),
                'cpu_usage_pct': float(cpu_usage_pct),
                'violation_pct': float(violation_pct),
                'violation_count': int(self._violation_count),
                'stages': stages,
                'samples': len(self._loop_times),
            }
