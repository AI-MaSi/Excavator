import time


class PIDController:
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
