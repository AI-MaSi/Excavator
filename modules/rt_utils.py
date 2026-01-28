"""
Real-time utilities for Linux RT kernels.

Provides optional RT enhancements (priority scheduling, memory locking, CPU affinity)
that can be enabled/disabled at runtime. Designed for stability over performance.

Usage:
    from modules.rt_utils import RTConfig, apply_rt_to_thread

    # Apply RT settings to current thread (safe - no-op if RT unavailable)
    apply_rt_to_thread(priority=50, lock_memory=True)

    # Or use RTConfig for more control
    rt = RTConfig(enabled=True, priority=50)
    rt.apply_to_current_thread()

    # Check what's available
    print(RTConfig.get_system_info())
"""

import os
import sys
import threading
import ctypes
from typing import Optional, Dict, Any, Set

__all__ = ['RTConfig', 'apply_rt_to_thread', 'is_rt_capable', 'get_system_info']


# Priority ranges (conservative defaults)
PRIORITY_MIN = 1
PRIORITY_MAX = 89  # Leave 90-99 for kernel threads
PRIORITY_DEFAULT = 50

# Scheduler policies
SCHED_OTHER = 0  # Normal (non-RT)
SCHED_FIFO = 1   # RT: First-in-first-out
SCHED_RR = 2     # RT: Round-robin


def is_rt_capable() -> bool:
    """Check if the system supports RT scheduling for this process."""
    try:
        # Try to query current scheduler - if this works, we have basic sched support
        os.sched_getscheduler(0)
        # Check if we have permission (need CAP_SYS_NICE or root)
        # We don't actually set anything, just check if we could
        return os.geteuid() == 0 or _has_cap_sys_nice()
    except (AttributeError, OSError):
        return False


def _has_cap_sys_nice() -> bool:
    """Check if process has CAP_SYS_NICE capability."""
    try:
        # Try reading capabilities from /proc
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('CapEff:'):
                    cap_hex = int(line.split()[1], 16)
                    CAP_SYS_NICE = 23
                    return bool(cap_hex & (1 << CAP_SYS_NICE))
    except Exception:
        pass
    return False


def get_system_info() -> Dict[str, Any]:
    """Get system RT capability information."""
    info = {
        'rt_capable': is_rt_capable(),
        'euid': os.geteuid(),
        'is_root': os.geteuid() == 0,
        'has_cap_sys_nice': _has_cap_sys_nice(),
        'python_version': sys.version_info[:3],
        'has_sched_module': hasattr(os, 'sched_setscheduler'),
    }

    # Check for RT kernel
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            info['kernel_version'] = version.strip()
            info['is_rt_kernel'] = 'PREEMPT_RT' in version or 'RT' in version
    except Exception:
        info['kernel_version'] = 'unknown'
        info['is_rt_kernel'] = False

    # Get available CPUs
    try:
        info['available_cpus'] = list(os.sched_getaffinity(0))
        info['cpu_count'] = os.cpu_count()
    except Exception:
        info['available_cpus'] = []
        info['cpu_count'] = os.cpu_count()

    return info


def apply_rt_to_thread(
    priority: int = PRIORITY_DEFAULT,
    policy: int = SCHED_FIFO,
    lock_memory: bool = False,
    cpu_affinity: Optional[Set[int]] = None,
    quiet: bool = False
) -> bool:
    """Apply RT settings to the current thread.

    Safe to call even if RT is unavailable - will silently succeed with no-op.

    Args:
        priority: RT priority (1-89, higher = more priority)
        policy: SCHED_FIFO (default) or SCHED_RR
        lock_memory: Lock all memory to prevent page faults
        cpu_affinity: Set of CPUs to pin to (None = don't change)
        quiet: Suppress warning messages

    Returns:
        True if RT settings were applied, False if unavailable/failed
    """
    success = True

    # Clamp priority to safe range
    priority = max(PRIORITY_MIN, min(PRIORITY_MAX, priority))

    # Set scheduler priority
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, policy, param)
    except (AttributeError, PermissionError, OSError) as e:
        if not quiet:
            print(f"[RT] Could not set scheduler priority: {e}")
        success = False

    # Lock memory
    if lock_memory:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            MCL_CURRENT = 1
            MCL_FUTURE = 2
            result = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
            if result != 0:
                errno = ctypes.get_errno()
                if not quiet:
                    print(f"[RT] mlockall failed with errno {errno}")
                success = False
        except Exception as e:
            if not quiet:
                print(f"[RT] Could not lock memory: {e}")
            success = False

    # Set CPU affinity
    if cpu_affinity is not None:
        try:
            os.sched_setaffinity(0, cpu_affinity)
        except (AttributeError, PermissionError, OSError) as e:
            if not quiet:
                print(f"[RT] Could not set CPU affinity: {e}")
            success = False

    return success


def reset_to_normal(quiet: bool = False) -> bool:
    """Reset current thread to normal (non-RT) scheduling.

    Returns:
        True if reset succeeded, False otherwise
    """
    try:
        param = os.sched_param(0)
        os.sched_setscheduler(0, SCHED_OTHER, param)
        return True
    except (AttributeError, PermissionError, OSError) as e:
        if not quiet:
            print(f"[RT] Could not reset scheduler: {e}")
        return False


class RTConfig:
    """Configuration class for RT settings.

    Allows enabling/disabling RT features and applying them to threads.
    Thread-safe for concurrent apply calls.

    Example:
        rt = RTConfig(enabled=True, priority=60, lock_memory=True)

        # In main control loop thread:
        rt.apply_to_current_thread()

        # In background thread with lower priority:
        rt.apply_to_current_thread(priority_offset=-20)
    """

    def __init__(
        self,
        enabled: bool = False,
        priority: int = PRIORITY_DEFAULT,
        policy: int = SCHED_FIFO,
        lock_memory: bool = False,
        cpu_affinity: Optional[Set[int]] = None,
        quiet: bool = False
    ):
        """Initialize RT configuration.

        Args:
            enabled: Whether RT features are active (False = all ops are no-ops)
            priority: Base RT priority (1-89)
            policy: SCHED_FIFO or SCHED_RR
            lock_memory: Lock memory to prevent page faults
            cpu_affinity: CPUs to pin to (None = don't pin)
            quiet: Suppress warning messages
        """
        self.enabled = enabled
        self.priority = max(PRIORITY_MIN, min(PRIORITY_MAX, priority))
        self.policy = policy
        self.lock_memory = lock_memory
        self.cpu_affinity = cpu_affinity
        self.quiet = quiet
        self._lock = threading.Lock()
        self._applied_threads: Set[int] = set()

    def apply_to_current_thread(self, priority_offset: int = 0) -> bool:
        """Apply RT settings to the current thread.

        Args:
            priority_offset: Adjust priority relative to base (e.g., -10 for lower priority)

        Returns:
            True if applied successfully, False if disabled or failed
        """
        if not self.enabled:
            return False

        thread_id = threading.get_ident()
        adjusted_priority = max(PRIORITY_MIN, min(PRIORITY_MAX, self.priority + priority_offset))

        success = apply_rt_to_thread(
            priority=adjusted_priority,
            policy=self.policy,
            lock_memory=self.lock_memory,
            cpu_affinity=self.cpu_affinity,
            quiet=self.quiet
        )

        if success:
            with self._lock:
                self._applied_threads.add(thread_id)

        return success

    def reset_current_thread(self) -> bool:
        """Reset current thread to normal scheduling."""
        thread_id = threading.get_ident()
        success = reset_to_normal(quiet=self.quiet)

        if success:
            with self._lock:
                self._applied_threads.discard(thread_id)

        return success

    def get_applied_thread_count(self) -> int:
        """Get number of threads with RT settings applied."""
        with self._lock:
            return len(self._applied_threads)

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system RT capability information."""
        return get_system_info()

    def __repr__(self) -> str:
        policy_name = {SCHED_FIFO: 'FIFO', SCHED_RR: 'RR', SCHED_OTHER: 'OTHER'}.get(self.policy, '?')
        return (f"RTConfig(enabled={self.enabled}, priority={self.priority}, "
                f"policy={policy_name}, lock_memory={self.lock_memory})")
