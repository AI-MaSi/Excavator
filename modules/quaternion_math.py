"""
Common quaternion math operations optimized with numba.

All quaternions use [w, x, y, z] format, where w is the scalar part
and [x, y, z] is the vector part.
"""

import numpy as np
import numba


@numba.njit(fastmath=False, nogil=True)
def normalize_vector(x, eps=1e-9):
    """Normalize a vector to unit length."""
    x = np.asarray(x, dtype=np.float32)
    eps = np.float32(eps)
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm




# Quaternion operations
@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
def euler_xyz_from_quat(quat):
    """
    Convert quaternion to Euler angles (XYZ convention).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Tuple of (roll, pitch, yaw) in radians as float32. [-π, π].
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

@numba.njit(fastmath=False, nogil=True)
def quat_unique(q):
    """Ensure quaternion has positive real part."""
    q = np.asarray(q, dtype=np.float32)
    if q[0] < np.float32(0.0):
        return -q
    return q


@numba.njit(fastmath=False, nogil=True)
def quat_enforce_hemisphere(q, q_prev):
    """Keep quaternion in the same hemisphere as a previous quaternion.

    Quaternions q and -q represent the same rotation, but flipping between
    them causes interpolation and visualization artefacts. This negates q
    when it points away from q_prev (negative dot product).

    Args:
        q: Current quaternion [w, x, y, z]
        q_prev: Previous quaternion [w, x, y, z]

    Returns:
        q or -q, whichever is closer to q_prev.
    """
    q = np.asarray(q, dtype=np.float32)
    q_prev = np.asarray(q_prev, dtype=np.float32)
    dot = q[0]*q_prev[0] + q[1]*q_prev[1] + q[2]*q_prev[2] + q[3]*q_prev[3]
    if dot < np.float32(0.0):
        return -q
    return q


@numba.njit(fastmath=False, nogil=True)
def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion [w, x, y, z]  (returned when t=0)
        q2: End quaternion [w, x, y, z]    (returned when t=1)
        t:  Interpolation parameter in [0, 1]

    Returns:
        Interpolated unit quaternion [w, x, y, z]
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    t = np.float32(t)

    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

    # Stay in same hemisphere
    if dot < np.float32(0.0):
        q2 = -q2
        dot = -dot

    # If very close, use linear interp to avoid division by zero
    if dot > np.float32(0.9995):
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w1 = np.sin((np.float32(1.0) - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return quat_normalize(w1 * q1 + w2 * q2)


@numba.njit(fastmath=False, nogil=True)
def quat_remove_yaw(q):
    """Remove yaw (Z-axis rotation) component from a quaternion.

    Computes the yaw angle, builds its inverse, and pre-multiplies
    to cancel it out, leaving only roll and pitch.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Quaternion with yaw zeroed out.
    """
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q[0], q[1], q[2], q[3]
    yaw = np.arctan2(np.float32(2.0) * (w * z + x * y),
                     np.float32(1.0) - np.float32(2.0) * (y * y + z * z))
    half = -yaw / np.float32(2.0)
    yaw_inv = np.array([np.cos(half), np.float32(0.0), np.float32(0.0),
                         np.sin(half)], dtype=np.float32)
    return quat_multiply(yaw_inv, q)


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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


@numba.njit(fastmath=False, nogil=True)
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
