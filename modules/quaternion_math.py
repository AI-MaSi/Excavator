"""
Common quaternion math operations optimized with numba.

All quaternions use [w, x, y, z] format, where w is the scalar part
and [x, y, z] is the vector part.
"""

import numpy as np
import numba


# General utility functions for numerical operations
@numba.njit(fastmath=False)
def normalize_vector(x, eps=1e-9):
    """Normalize a vector to unit length."""
    x = np.asarray(x, dtype=np.float32)
    eps = np.float32(eps)
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


@numba.njit(fastmath=False)
def saturate(x, lower, upper):
    """Clamp values between lower and upper bounds."""
    x = np.asarray(x, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    return np.maximum(np.minimum(x, upper), lower)


@numba.njit(fastmath=False)
def wrap_to_pi(angles):
    """Wrap angles to [-π, π] range."""
    angles = np.asarray(angles, dtype=np.float32)
    pi = np.float32(3.141592653589793)
    two_pi = np.float32(6.283185307179586)
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + pi) % two_pi
    # map to [-pi, pi) (note: +pi maps to -pi by design)
    result = wrapped_angle - pi
    return result




# Quaternion operations
@numba.njit(fastmath=False)
def quat_normalize(q, eps=1e-9):
    """
    Normalize a quaternion to unit magnitude.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Normalized quaternion of unit length
    """
    q = np.asarray(q, dtype=np.float32)
    eps = np.float32(eps)
    norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if norm < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


@numba.njit(fastmath=False)
def quat_multiply(q1, q2):
    """
    Multiply two quaternions.

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Result of quaternion multiplication q1*q2
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    # Quaternion multiplication using optimized formula
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = np.float32(0.5) * (xx + (z1 - x1) * (x2 - y2))

    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.array([w, x, y, z], dtype=np.float32)


@numba.njit(fastmath=False)
def quat_inverse(q):
    """
    Compute quaternion inverse.

    For non-unit quaternions, inverse is conjugate(q) / ||q||^2.
    """
    q = np.asarray(q, dtype=np.float32)
    norm_sq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    if norm_sq < np.float32(1e-12):
        # Degenerate: return identity to avoid divide-by-zero
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_conjugate(q) / norm_sq



@numba.njit(fastmath=False)
def quat_conjugate(q):
    """
    Computes the conjugate of a quaternion.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    q = np.asarray(q, dtype=np.float32)
    result = np.empty(4, dtype=np.float32)
    result[0] = q[0]   # w
    result[1] = -q[1]  # -x
    result[2] = -q[2]  # -y
    result[3] = -q[3]  # -z
    return result


@numba.njit(fastmath=False)
def quat_rotate_vector(q, v):
    """
    Rotate a 3D vector using a quaternion.

    Args:
        q: Quaternion [w, x, y, z]
        v: 3D vector [x, y, z]

    Returns:
        Rotated 3D vector
    """
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    # Extract quaternion components
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Optimized rotation using quaternion formula
    # v' = v + 2 * qw * (q_vec × v) + 2 * q_vec × (q_vec × v)

    # First cross product: q_vec × v
    cross1 = np.array([
        qy * v[2] - qz * v[1],
        qz * v[0] - qx * v[2],
        qx * v[1] - qy * v[0]
    ], dtype=np.float32)

    # Second cross product: q_vec × (q_vec × v)
    cross2 = np.array([
        qy * cross1[2] - qz * cross1[1],
        qz * cross1[0] - qx * cross1[2],
        qx * cross1[1] - qy * cross1[0]
    ], dtype=np.float32)

    return v + np.float32(2.0) * qw * cross1 + np.float32(2.0) * cross2


@numba.njit(fastmath=False)
def quat_from_axis_angle(axis, angle):
    """Convert axis-angle to quaternion."""
    axis = np.asarray(axis, dtype=np.float32)
    angle = np.float32(angle)

    # Normalize axis
    axis_norm = normalize_vector(axis)
    half_angle = angle * np.float32(0.5)
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    return np.array([
        cos_half,
        axis_norm[0] * sin_half,
        axis_norm[1] * sin_half,
        axis_norm[2] * sin_half
    ], dtype=np.float32)


@numba.njit(fastmath=False)
def axis_angle_from_quat(q, eps=1e-6):
    """Convert quaternion to axis-angle representation."""
    q = np.asarray(q, dtype=np.float32)
    eps = np.float32(eps)

    q = quat_unique(q)  # Ensure positive w

    # Extract imaginary part
    quat_im = np.array([q[1], q[2], q[3]], dtype=np.float32)
    mag = np.linalg.norm(quat_im)

    if mag < eps:
        # Small angle, return zero rotation - FORCE FLOAT32
        return np.zeros(3, dtype=np.float32)

    # Compute angle
    half_angle = np.arctan2(mag, q[0])
    angle = np.float32(2.0) * half_angle

    # Compute axis
    axis = quat_im / mag

    return (axis * angle).astype(np.float32)


@numba.njit(fastmath=False)
def axis_angle_from_quat_signed(q, eps=1e-6):
    """
    Convert quaternion to axis-angle representation with signed angle.

    Unlike axis_angle_from_quat (which enforces w >= 0 and returns angles in [0, pi]),
    this keeps the original quaternion sign and returns angles in [-pi, pi].
    """
    q = np.asarray(q, dtype=np.float32)
    eps = np.float32(eps)

    # Extract imaginary part
    quat_im = np.array([q[1], q[2], q[3]], dtype=np.float32)
    mag = np.linalg.norm(quat_im)

    if mag < eps:
        return np.zeros(3, dtype=np.float32)

    # Angle in [-pi, pi] (sign follows quaternion sign)
    half_angle = np.arctan2(mag, q[0])
    angle = np.float32(2.0) * half_angle

    # Axis from vector part; scale by signed angle
    axis = quat_im / mag
    return (axis * angle).astype(np.float32)

@numba.njit(fastmath=False)
def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Convert Euler angles (XYZ convention) to quaternion.

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        Quaternion [w, x, y, z] as float32 array
    """
    # Ensure float32
    roll = np.float32(roll)
    pitch = np.float32(pitch)
    yaw = np.float32(yaw)

    # Half angles
    cy = np.cos(yaw * np.float32(0.5))
    sy = np.sin(yaw * np.float32(0.5))
    cr = np.cos(roll * np.float32(0.5))
    sr = np.sin(roll * np.float32(0.5))
    cp = np.cos(pitch * np.float32(0.5))
    sp = np.sin(pitch * np.float32(0.5))

    # Compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qw, qx, qy, qz], dtype=np.float32)


@numba.njit(fastmath=False)
def euler_xyz_from_quat(quat):
    """
    Convert quaternion to Euler angles (XYZ convention).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Tuple of (roll, pitch, yaw) in radians as float32. [0, 2π]!
    """
    # Ensure float32
    quat = np.asarray(quat, dtype=np.float32)
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sin_roll = np.float32(2.0) * (q_w * q_x + q_y * q_z)
    cos_roll = np.float32(1.0) - np.float32(2.0) * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    # Pitch (y-axis rotation)
    sin_pitch = np.float32(2.0) * (q_w * q_y - q_z * q_x)
    # Handle gimbal lock
    if abs(sin_pitch) >= np.float32(1.0):
        pitch = np.copysign(np.float32(np.pi) / np.float32(2.0), sin_pitch)
    else:
        pitch = np.arcsin(sin_pitch)

    # Yaw (z-axis rotation)
    sin_yaw = np.float32(2.0) * (q_w * q_z + q_x * q_y)
    cos_yaw = np.float32(1.0) - np.float32(2.0) * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    return roll, pitch, yaw

@numba.njit(fastmath=False)
def quat_unique(q):
    """Ensure quaternion has positive real part."""
    q = np.asarray(q, dtype=np.float32)
    if q[0] < np.float32(0.0):
        return -q
    return q


@numba.njit(fastmath=False)
def quat_box_minus(q1, q2):
    """Quaternion box-minus operator (quaternion difference)."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    # Compute q1 * conj(q2)
    quat_diff = quat_multiply(q1, quat_conjugate(q2))

    # Convert to axis-angle representation
    return axis_angle_from_quat(quat_diff)


@numba.njit(fastmath=False)
def quat_error_magnitude(q1, q2):
    """Compute angular error between two quaternions."""
    diff_axis_angle = quat_box_minus(q1, q2)
    return np.linalg.norm(diff_axis_angle)


@numba.njit(fastmath=False)
def compute_quat_error(q1, q2):
    """
    Compute the orientation error between two quaternions.
    Returns the error as an axis-angle representation.

    Args:
        q1: Current orientation quaternion [w, x, y, z]
        q2: Target orientation quaternion [w, x, y, z]

    Returns:
        axis_angle_error: 3D vector representing orientation error
    """
    # Ensure float32 inputs and normalize
    q1 = quat_normalize(np.asarray(q1, dtype=np.float32))
    q2 = quat_normalize(np.asarray(q2, dtype=np.float32))

    # Compute q2 * q1^-1 (error quaternion)
    q1_conj = quat_conjugate(q1)
    quat_error = quat_multiply(q2, q1_conj)

    # Convert to axis-angle representation
    return axis_angle_from_quat(quat_error)


@numba.njit(fastmath=False)
def quat_exponential(v, scalar=0.0):
    """
    Compute quaternion exponential exp(q).
    For quaternion q = [scalar, v]

    Args:
        v: Vector part [x, y, z]
        scalar: Scalar part (default: 0.0)

    Returns:
        Exponential quaternion [w, x, y, z]
    """
    # Ensure float32 inputs
    v = np.asarray(v, dtype=np.float32)
    scalar = np.float32(scalar)

    v_norm = np.sqrt(np.sum(v * v))

    # Handle small vector norm case
    if v_norm < np.float32(1e-10):
        return np.array([np.exp(scalar), 0.0, 0.0, 0.0], dtype=np.float32)

    # Calculate the exponential
    exp_scalar = np.exp(scalar)
    factor = exp_scalar * np.sin(v_norm) / v_norm

    return np.array([
        exp_scalar * np.cos(v_norm),
        factor * v[0],
        factor * v[1],
        factor * v[2]
    ], dtype=np.float32)


@numba.njit(fastmath=False)
def integrate_quat_exp(q, angular_velocity, dt):
    """
    Integrate quaternion with exponential method.

    Args:
        q: Current quaternion [w, x, y, z]
        angular_velocity: Angular velocity [wx, wy, wz] in rad/s
        dt: Time step in seconds

    Returns:
        Updated quaternion
    """
    # Ensure float32 inputs
    q = np.asarray(q, dtype=np.float32)
    angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
    dt = np.float32(dt)

    half_angle = np.float32(0.5) * dt
    omega_vector = angular_velocity * half_angle
    q_exp = quat_exponential(omega_vector)
    q_new = quat_multiply(q, q_exp)
    return quat_normalize(q_new)


# Convenience helpers for Y-axis rotations (degrees)
def quat_from_y_deg(y_deg: float) -> np.ndarray:
    """
    Create a quaternion [w, x, y, z] representing a rotation of y_deg degrees
    around the Y axis. Returns float32 array.
    """
    half = np.deg2rad(y_deg) * 0.5
    w = np.cos(half)
    y = np.sin(half)
    return np.array([w, 0.0, y, 0.0], dtype=np.float32)

def y_deg_from_quat(q: np.ndarray) -> float:
    """
    Extract Y-axis rotation in degrees from a quaternion [w, x, y, z].
    Assumes rotation is primarily about Y.
    """
    q = np.asarray(q, dtype=np.float32)
    q = quat_normalize(q)
    # For pure Y-axis rotation, sin(theta/2) = y, cos(theta/2) = w
    angle = 2.0 * np.arctan2(q[2], q[0])
    return float(np.rad2deg(angle))


@numba.njit(fastmath=False)
def compute_pose_error(pos1, quat1, pos2, quat2):
    """
    Compute the position and orientation error between current and target poses.

    Args:
        pos1: Current position [x, y, z]
        quat1: Current orientation quaternion [w, x, y, z]
        pos2: Target position [x, y, z]
        quat2: Target orientation quaternion [w, x, y, z]

    Returns:
        pos_error: Position error vector [dx, dy, dz]
        rot_error: Rotation error as axis-angle vector [rx, ry, rz]
    """
    # Ensure float32
    pos1 = np.asarray(pos1, dtype=np.float32)
    quat1 = np.asarray(quat1, dtype=np.float32)
    pos2 = np.asarray(pos2, dtype=np.float32)
    quat2 = np.asarray(quat2, dtype=np.float32)

    # Position error is straightforward
    pos_error = pos2 - pos1

    # Orientation error in axis-angle form
    rot_error = compute_quat_error(quat1, quat2)

    return pos_error, rot_error


@numba.njit(fastmath=False)
def apply_delta_pose(source_pos, source_quat, delta_pose, eps=1e-6):
    """
    Apply delta pose (6DOF: position + axis-angle) to source pose.

    Args:
        source_pos: Current position [x, y, z]
        source_quat: Current quaternion [w, x, y, z]
        delta_pose: Delta [dx, dy, dz, rx, ry, rz]
        eps: Small angle threshold

    Returns:
        target_pos, target_quat
    """
    # Ensure float32
    source_pos = np.asarray(source_pos, dtype=np.float32)
    source_quat = np.asarray(source_quat, dtype=np.float32)
    delta_pose = np.asarray(delta_pose, dtype=np.float32)
    eps = np.float32(eps)

    # Apply position delta
    target_pos = source_pos + delta_pose[0:3]

    # Apply rotation delta
    rot_delta = delta_pose[3:6]
    angle = np.linalg.norm(rot_delta)

    if angle > eps:
        axis = rot_delta / angle
        delta_quat = quat_from_axis_angle(axis, angle)
        target_quat = quat_multiply(delta_quat, source_quat)
    else:
        target_quat = source_quat.copy()

    return target_pos, quat_normalize(target_quat)


@numba.njit(fastmath=False)
def check_jacobian_singularity(jacobian, threshold=1e-3):
    """
    Check if Jacobian is near singular configuration.
    Returns (is_singular, condition_number, min_singular_value)
    """
    jacobian = np.asarray(jacobian, dtype=np.float32)
    threshold = np.float32(threshold)

    U, S, Vh = np.linalg.svd(jacobian, full_matrices=False)

    min_sv = np.min(S)
    max_sv = np.max(S)

    condition_number = max_sv / (min_sv + np.float32(1e-12))  # Avoid division by zero
    is_singular = min_sv < threshold

    return is_singular, condition_number, min_sv


@numba.njit(fastmath=False)
def apply_joint_limits(joint_angles, joint_limits_lower, joint_limits_upper):
    """Apply joint limits with clamping."""
    return saturate(joint_angles, joint_limits_lower, joint_limits_upper)


@numba.njit(fastmath=False)
def apply_joint_velocity_limits(delta_joints, velocity_limits, dt=0.01):
    """Apply joint velocity limits."""
    delta_joints = np.asarray(delta_joints, dtype=np.float32)
    velocity_limits = np.asarray(velocity_limits, dtype=np.float32)
    dt = np.float32(dt)

    max_delta = velocity_limits * dt
    return saturate(delta_joints, -max_delta, max_delta)


@numba.njit(fastmath=False)
def check_workspace_limits(target_pos, max_reach, min_reach=0.0):
    """Check if target position is within workspace."""
    target_pos = np.asarray(target_pos, dtype=np.float32)
    max_reach = np.float32(max_reach)
    min_reach = np.float32(min_reach)

    distance = np.linalg.norm(target_pos)
    return (distance >= min_reach) and (distance <= max_reach)


@numba.njit(fastmath=False)
def compute_manipulability(jacobian):
    """Compute manipulability measure (sqrt of determinant of J*J^T)."""
    jacobian = np.asarray(jacobian, dtype=np.float32)
    jjt = np.dot(jacobian, jacobian.T)
    det_jjt = np.linalg.det(jjt)
    return np.sqrt(np.abs(det_jjt))
