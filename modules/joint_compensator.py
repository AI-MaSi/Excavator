"""Per-joint feedforward compensator (stub — not wired in).

Place to centralize predictable, pose/state-dependent PWM contributions
that should NOT live inside the PID. The PID stays a small error-correction
loop; this module models everything we already know about the actuator
before the loop runs.

Intended call site (when wired):
    pid_outputs = [...]                       # from joint-space PID loop
    ff = compensator.compute(
        joint_angles=current_joint_angles,    # rad, [slew, boom, arm, bucket]
        joint_velocities=last_joint_vel_radps,  # rad/s, optional
        target_joint_angles=target_joint_angles,  # rad, for velocity FF
        joint_quats=current_fk_quats,         # for world-pitch of each link
    )
    combined = [p + f for p, f in zip(pid_outputs, ff)]
    combined = np.clip(combined, -1.0, 1.0)
    # See "Anti-windup with FF" note below: feed the saturation back into PID.

Why outside the PID:
- Compensation is a function of pose / desired motion, not error.
- If baked into error, the integrator slowly relearns each compensation and
  fights the FF — windup, lag, retuning hassles.
- Decoupling lets each compensation be identified offline against static
  data and unit-tested without the live control loop.

----------------------------------------------------------------------
Compensations to consider (each can be added independently)
----------------------------------------------------------------------

1. Gravity feedforward (boom / arm / bucket; slew is gravity-neutral)
   - Model: pwm_i = a_i * cos(world_pitch_i) + b_i  (one term per joint)
   - Inputs: absolute link orientations (from joint_quats[i] → Y twist).
   - Identification: park the arm in a sweep of static poses, log the steady
     PWM the PID converges to (zero error → all of it is "hold against
     gravity"), then fit a_i and b_i.
   - Notes: world_pitch_i is the absolute pitch of link i, which is what
     gravity actually torques against — not the relative joint angle.

2. Asymmetric valve compensation
   - Hydraulic valves often have different flow on extend vs retract.
   - Already partially handled in servo_config_200.yaml via
     deadband_us_pos / deadband_us_neg + dither. Anything beyond a
     constant deadband (e.g., gain asymmetry) can be modeled here as a
     per-channel piecewise gain on (pid_out + grav_ff) before the PWM map.

3. Stiction / break-away bias
   - When |joint_velocity| ~ 0 and |error| > epsilon, push a small bias in
     the sign of error to break static friction.
   - Different from dither: dither is unconditional vibration; this is a
     conditional bump tied to the error state.

4. Velocity feedforward
   - For tracking moving setpoints. PWM_ff_vel = c_i * desired_joint_rate.
   - Useful in velocity-mode IK; less critical with S-curve smoothing.

5. Coupling / cross-axis compensation
   - On some excavators, lift-boom motion induces unwanted arm motion via
     hydraulic line crosstalk. Identifiable as off-diagonal coefficients.
   - Add only if measurement shows it; otherwise YAGNI.

6. Pump-pressure scaling
   - PWM authority scales with hydraulic pressure. If pressure is measured
     (ADCPi), PWM commands can be inversely scaled by pressure_norm so the
     control loop sees a constant effective gain.

----------------------------------------------------------------------
Anti-windup interaction with PID
----------------------------------------------------------------------

The existing PID's anti-windup checks saturation against its OWN output
range. Once we sum FF onto the PID output, the actual saturation point
shifts:

    combined = pid_out + ff
    actual   = clip(combined, -1, 1)
    excess   = combined - actual    # nonzero when saturated

If the PID's own integrator doesn't see this excess, it can wind up while
the actuator is already pegged by FF. Two options when wiring in:

  a. Tell the PID about the external saturation each cycle (cleanest).
     PIDController would need a small `back_calculate(excess)` hook that
     drains a fraction of `excess` from `integral_sum`.

  b. Squeeze PID authority to [-1 + |ff_max|, 1 - |ff_max|]. Coarse but
     trivially correct as long as |ff| <= ff_max.

Either is fine for hydraulics; (b) is the lower-touch starting point.

The current simple PID (modules/pid.py) is fine — no changes are required
purely for the wrap or for FF support. Only revisit it if you adopt option
(a) above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class JointCompensatorConfig:
    """Per-joint enable flags and gains for each compensation term.

    Each list is length 4: [slew, boom, arm, bucket]. Gains default to zero
    so an un-tuned compensator is a transparent no-op.
    """

    # Gravity FF: pwm = grav_a[i] * cos(world_pitch[i]) + grav_b[i].
    # Slew is gravity-neutral; leave grav_a[0] / grav_b[0] at 0.
    grav_a: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    grav_b: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Stiction kick: small bias in sign(error) when |velocity| < threshold
    # and |error| > min_error. Applied per joint.
    stiction_bias: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    stiction_velocity_thresh_radps: float = 0.01
    stiction_error_thresh_rad: float = 0.005

    # Velocity FF: pwm = vel_gain[i] * desired_joint_rate[i].
    vel_gain: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Hard cap on the FF magnitude so the PID always retains some authority.
    ff_clip: float = 0.3


class JointCompensator:
    """Stub. Returns zeros until populated.

    Intended public surface:
        comp = JointCompensator(cfg, robot_config)
        ff = comp.compute(
            joint_angles=...,        # rad (n_joints,)
            joint_quats=...,         # absolute link quats (n_joints, 4)
            joint_velocities=...,    # rad/s (n_joints,) or None
            target_joint_angles=..., # rad (n_joints,) or None
            angle_error=...,         # rad (n_joints,) or None
        )
        # ff is a length-n_joints array of additive PWM values, pre-clip.

    Each `_compute_*` method below is a hook to fill in. They all currently
    return zeros, so wiring this in produces no change in behaviour until
    individual terms are enabled.
    """

    def __init__(self, cfg: JointCompensatorConfig, robot_config) -> None:
        self.cfg = cfg
        self.robot_config = robot_config
        self.n = int(getattr(robot_config, "num_joints", 4))

    # ------------------------------------------------------------------
    # Individual term hooks — each returns a length-n_joints np.float32
    # contribution. Keep them pure functions of the inputs so unit tests can
    # exercise each in isolation against a pose dictionary.
    # ------------------------------------------------------------------

    def _gravity_ff(self, joint_quats: np.ndarray) -> np.ndarray:
        """pwm_i = grav_a[i] * cos(world_pitch_i) + grav_b[i]. TODO."""
        return np.zeros(self.n, dtype=np.float32)

    def _stiction_kick(
        self,
        joint_velocities: Optional[np.ndarray],
        angle_error: Optional[np.ndarray],
    ) -> np.ndarray:
        """Sign-of-error bias when stationary and error is non-trivial. TODO."""
        return np.zeros(self.n, dtype=np.float32)

    def _velocity_ff(
        self,
        target_joint_angles: Optional[np.ndarray],
        joint_angles: Optional[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        """pwm_i = vel_gain[i] * desired_joint_rate[i]. TODO."""
        return np.zeros(self.n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Aggregator
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        joint_angles: Optional[np.ndarray] = None,
        joint_quats: Optional[np.ndarray] = None,
        joint_velocities: Optional[np.ndarray] = None,
        target_joint_angles: Optional[np.ndarray] = None,
        angle_error: Optional[np.ndarray] = None,
        dt: float = 0.005,
    ) -> np.ndarray:
        """Return per-joint additive PWM contributions, clipped to ``ff_clip``.

        All inputs are optional so individual term hooks can be filled in
        and wired up incrementally. Currently returns zeros.
        """
        ff = np.zeros(self.n, dtype=np.float32)
        if joint_quats is not None:
            ff = ff + self._gravity_ff(np.asarray(joint_quats, dtype=np.float32))
        ff = ff + self._stiction_kick(joint_velocities, angle_error)
        ff = ff + self._velocity_ff(target_joint_angles, joint_angles, dt)

        cap = float(self.cfg.ff_clip)
        if cap > 0.0:
            ff = np.clip(ff, -cap, cap)
        return ff.astype(np.float32)
