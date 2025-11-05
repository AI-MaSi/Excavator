import time
import math


class PIDController_old:
    """
    Simple PID controller.
    """

    def __init__(self, kp=1.0, ki=0.1, kd=0.05, min_output=-1.0, max_output=1.0):
        """
        Initialize the PID controller with gain values.

        Args:
            kp: Proportional gain (default: 1.0)
            ki: Integral gain (default: 0.1)
            kd: Derivative gain (default: 0.05)
            min_output: Minimum output value (default: -1.0)
            max_output: Maximum output value (default: 1.0)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output

        # Internal state
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.last_time = time.monotonic()

    def reset(self):
        """Reset the controller's internal state."""
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.last_time = time.monotonic()

    def compute(self, setpoint, current_value):
        """
        # Inner Loop: Joint-space PID control
        Compute PID control output based on setpoint and current value.

        Args:
            setpoint: Target value
            current_value: Current measured value

        Returns:
            control_output: Clamped control output in range [min_output, max_output]
        """
        # Calculate error
        error = setpoint - current_value

        # Get current time and compute time delta
        current_time = time.monotonic()
        dt = current_time - self.last_time
        dt = max(dt, 0.001)  # protect against zero/negative dt

        # Integral term
        self.integral_sum += error * dt

        # Derivative term
        error_derivative = (error - self.last_error) / dt

        # PID output
        output = (self.kp * error) + (self.ki * self.integral_sum) + (self.kd * error_derivative)

        # Clamp output
        clamped_output = max(self.min_output, min(self.max_output, output))

        # Anti-windup: prevent runaway integral when saturated
        if clamped_output != output:
            self.integral_sum -= error * dt

        # Update state
        self.last_error = error
        self.last_time = current_time

        return clamped_output


class PIDController:
    """
    PID with:
      • Derivative on MEASUREMENT (reduces setpoint kick) + 1st-order filter
      • Integral limits (prevents deep windup)
      • Sign-aware clamping anti-windup (matches the diagram logic)

    Tuning quick notes:
      - kp: raises responsiveness; too high → overshoot/oscillation.
      - ki: removes steady-state error; increase until bias disappears, but watch for windup.
      - kd: damps/filters fast changes; set mainly for disturbance rejection.
      - deriv_filter_tau (s): larger → smoother derivative, slower; smaller → faster, noisier.
      - Imin/Imax: cap the stored integral; start with ≈ [min_output/ki, max_output/ki].
    """

    def __init__(
        self,
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_output=-1.0,
        max_output=1.0,
        deriv_filter_tau=0.05,   # ↑ if measurement is noisy; ↓ if you need faster D
        Imin=None,               # default computed from outputs & ki
        Imax=None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.deriv_filter_tau = max(0.0, float(deriv_filter_tau))

        # Integral limits: default to what actuator can deliver via the I term alone
        if Imin is None or Imax is None:
            if ki != 0.0:
                Imin = min_output / ki
                Imax = max_output / ki
            else:
                Imin, Imax = -math.inf, math.inf
        self.Imin = min(Imin, Imax)
        self.Imax = max(Imin, Imax)

        # Internal state
        self.integral_sum = 0.0
        self.last_error = 0.0          # kept for reference; not used by D-on-measurement
        self.last_value = None         # last measured value (for D term)
        self.d_filt = 0.0              # filtered derivative of measurement
        self.last_time = time.monotonic()

    def reset(self, keep_integral=False):
        """
        Reset internal state.
        Set keep_integral=True to preserve integral_sum (bumpless handover).
        """
        if not keep_integral:
            self.integral_sum = 0.0
        self.last_error = 0.0
        self.last_value = None
        self.d_filt = 0.0
        self.last_time = time.monotonic()

    def set_integral_limits(self, Imin, Imax):
        """Update integral limits at runtime (e.g., after retuning Ki)."""
        self.Imin = min(Imin, Imax)
        self.Imax = max(Imin, Imax)
        # Immediately enforce the new bounds
        self.integral_sum = max(self.Imin, min(self.Imax, self.integral_sum))

    def compute(self, setpoint, current_value):
        """
        Compute clamped control output in [min_output, max_output].
        """
        # Timebase
        now = time.monotonic()
        dt = max(now - self.last_time, 1e-3)  # protect against tiny/negative dt

        # Error
        error = setpoint - current_value

        # ----- Derivative on MEASUREMENT (reduces setpoint kick) -----
        if self.last_value is None:
            raw_d = 0.0
        else:
            raw_d = -(current_value - self.last_value) / dt  # minus sign: d(-PV)/dt = d(error)/dt (when setpoint const)

        if self.deriv_filter_tau == 0.0:
            self.d_filt = raw_d
        else:
            # 1st-order low-pass: y += alpha*(x - y), alpha = dt/(tau+dt)
            alpha = dt / (self.deriv_filter_tau + dt)
            self.d_filt += alpha * (raw_d - self.d_filt)

        d_term = self.kd * self.d_filt

        # ----- Integral path with limits -----
        integral_step = error * dt
        tentative_integral = self.integral_sum + integral_step
        # Bound the integral state to keep it “actuator-sized”
        tentative_integral = max(self.Imin, min(self.Imax, tentative_integral))

        # Unsaturated controller output (uses tentative integral)
        unsat = (self.kp * error) + (self.ki * tentative_integral) + d_term

        # Actuator clamp
        out = max(self.min_output, min(self.max_output, unsat))
        saturated = (out != unsat)

        # ----- Sign-aware anti-windup (clamping per diagram) -----
        # Only hold the integrator when saturated AND error would push further into the limit.
        same_sign = (unsat > 0 and error > 0) or (unsat < 0 and error < 0)
        if not (saturated and same_sign):
            # Update integral when: not saturated, OR saturated but error helps unwind
            self.integral_sum = tentative_integral
        # else: hold integral_sum (turn integrator "off")

        # Update state for next call
        self.last_error = error
        self.last_value = current_value
        self.last_time = now

        return out