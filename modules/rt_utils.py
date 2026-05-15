# pyright: reportAttributeAccessIssue=false

"""Real-time utilities for Linux RT kernels.

Provides optional RT enhancements such as scheduler priority, memory locking,
and CPU affinity. Helpers are conservative by default and safe to call when RT
support is unavailable.
"""

import os
import sys
import threading
import ctypes
from typing import Optional, Dict, Any, Set

__all__ = [
    'RTConfig',
    'apply_rt_to_thread',
    'is_rt_capable',
    'get_system_info',
    'get_thread_info',
    'scheduler_name',
]


# Priority ranges (conservative defaults)
PRIORITY_MIN = 1
PRIORITY_MAX = 89  # Leave 90-99 for kernel threads
PRIORITY_DEFAULT = 50

# Scheduler policies
SCHED_OTHER = 0  # Normal (non-RT)
SCHED_FIFO = 1   # RT: First-in-first-out
SCHED_RR = 2     # RT: Round-robin

_sched_getscheduler = getattr(os, 'sched_getscheduler', None)
_sched_setscheduler = getattr(os, 'sched_setscheduler', None)
_sched_getparam = getattr(os, 'sched_getparam', None)
_sched_param = getattr(os, 'sched_param', None)
_sched_getaffinity = getattr(os, 'sched_getaffinity', None)
_sched_setaffinity = getattr(os, 'sched_setaffinity', None)
_geteuid_func = getattr(os, 'geteuid', None)


def _geteuid() -> int:
    """Best-effort effective uid."""
    if _geteuid_func is None:
        return -1
    try:
        return int(_geteuid_func())
    except Exception:
        return -1


def scheduler_name(policy: int) -> str:
    """Return human-readable scheduler name."""
    return {
        SCHED_OTHER: 'SCHED_OTHER',
        SCHED_FIFO: 'SCHED_FIFO',
        SCHED_RR: 'SCHED_RR',
    }.get(int(policy), f'UNKNOWN({policy})')


def _resolve_target_id(target_id: Optional[int]) -> int:
    """Translate None to the calling thread/process target id expected by os.*."""
    return 0 if target_id is None else int(target_id)


def is_rt_capable() -> bool:
    """Check if the system supports RT scheduling for this process."""
    try:
        # Try to query current scheduler - if this works, we have basic sched support
        if _sched_getscheduler is None:
            return False
        _sched_getscheduler(0)
        # Check if we have permission (need CAP_SYS_NICE or root)
        # We don't actually set anything, just check if we could
        return _geteuid() == 0 or _has_cap_sys_nice()
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
        'euid': _geteuid(),
        'is_root': _geteuid() == 0,
        'has_cap_sys_nice': _has_cap_sys_nice(),
        'python_version': sys.version_info[:3],
        'has_sched_module': _sched_setscheduler is not None,
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
        if _sched_getaffinity is None:
            raise RuntimeError("sched_getaffinity unavailable")
        info['available_cpus'] = list(_sched_getaffinity(0))
        info['cpu_count'] = os.cpu_count()
    except Exception:
        info['available_cpus'] = []
        info['cpu_count'] = os.cpu_count()

    try:
        if _sched_getscheduler is None or _sched_getparam is None:
            raise RuntimeError("scheduler queries unavailable")
        info['current_scheduler'] = scheduler_name(_sched_getscheduler(0))
        info['current_priority'] = int(_sched_getparam(0).sched_priority)
    except Exception:
        info['current_scheduler'] = 'unknown'
        info['current_priority'] = None

    return info


def get_thread_info(target_id: Optional[int] = None) -> Dict[str, Any]:
    """Return scheduler, priority, affinity, and identity for a thread/task.

    Args:
        target_id: Linux task id / native thread id. None means the calling thread.
    """
    tid = _resolve_target_id(target_id)
    info: Dict[str, Any] = {
        'target_id': tid if tid != 0 else threading.get_native_id(),
    }
    if target_id is None:
        info['python_thread_id'] = threading.get_ident()
        info['python_native_id'] = threading.get_native_id()

    try:
        if _sched_getscheduler is None:
            raise RuntimeError("sched_getscheduler unavailable")
        policy = _sched_getscheduler(tid)
        info['scheduler'] = scheduler_name(policy)
        info['scheduler_id'] = int(policy)
    except Exception:
        info['scheduler'] = 'unknown'
        info['scheduler_id'] = None

    try:
        if _sched_getparam is None:
            raise RuntimeError("sched_getparam unavailable")
        info['priority'] = int(_sched_getparam(tid).sched_priority)
    except Exception:
        info['priority'] = None

    try:
        if _sched_getaffinity is None:
            raise RuntimeError("sched_getaffinity unavailable")
        info['cpu_affinity'] = sorted(_sched_getaffinity(tid))
    except Exception:
        info['cpu_affinity'] = None

    return info


def apply_rt_to_thread(
    priority: Optional[int] = PRIORITY_DEFAULT,
    policy: int = SCHED_FIFO,
    lock_memory: bool = True,
    cpu_affinity: Optional[Set[int]] = None,
    quiet: bool = False,
    target_id: Optional[int] = None,
) -> bool:
    """Apply RT settings to the current thread.

    Safe to call even if RT is unavailable - will silently succeed with no-op.

    Args:
        priority: RT priority (1-89, higher = more priority). If None or <= 0,
                  scheduler policy is left unchanged.
        policy: SCHED_FIFO (default) or SCHED_RR
        lock_memory: Lock all memory to prevent page faults
        cpu_affinity: Set of CPUs to pin to (None = don't change)
        quiet: Suppress warning messages
        target_id: Linux task id / native thread id to configure. None means
                   the calling thread.

    Returns:
        True if RT settings were applied, False if unavailable/failed
    """
    success = True
    sched_target = _resolve_target_id(target_id)

    # Set scheduler priority
    if priority is not None and priority > 0:
        try:
            priority = max(PRIORITY_MIN, min(PRIORITY_MAX, int(priority)))
            if _sched_param is None or _sched_setscheduler is None:
                raise AttributeError("sched_setscheduler unavailable")
            param = _sched_param(priority)
            _sched_setscheduler(sched_target, policy, param)
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
            if _sched_setaffinity is None:
                raise AttributeError("sched_setaffinity unavailable")
            _sched_setaffinity(sched_target, cpu_affinity)
        except (AttributeError, PermissionError, OSError) as e:
            if not quiet:
                print(f"[RT] Could not set CPU affinity: {e}")
            success = False

    return success


def reset_to_normal(quiet: bool = False, target_id: Optional[int] = None) -> bool:
    """Reset current thread to normal (non-RT) scheduling.

    Returns:
        True if reset succeeded, False otherwise
    """
    try:
        if _sched_param is None or _sched_setscheduler is None:
            raise AttributeError("sched_setscheduler unavailable")
        param = _sched_param(0)
        _sched_setscheduler(_resolve_target_id(target_id), SCHED_OTHER, param)
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
