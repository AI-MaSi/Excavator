"""
IMU Calibration Loading and Application

Loads yaw calibration from imu_calibration.json and applies corrections.
"""

import json
import os
import time
import numpy as np
from modules.quaternion_math import quat_from_axis_angle, quat_multiply, quat_normalize

class IMUCalibration:
    """Manages IMU yaw calibration offsets."""

    def __init__(self, calibration_file='imu_calibration.json'):
        self.calibration_file = calibration_file
        self.yaw_offsets_rad = None
        self.is_loaded = False
        self.calibration_date = None

        # Try to load calibration
        self.load_calibration()

    def load_calibration(self):
        """Load calibration from JSON file."""
        if not os.path.exists(self.calibration_file):
            print("\n" + "="*60)
            print("⚠  WARNING: NO IMU CALIBRATION FILE FOUND!")
            print("="*60)
            print(f"  Missing file: {self.calibration_file}")
            print("  IMU yaw offsets will NOT be corrected!")
            print()
            print("  To calibrate:")
            print("    1. Mark ground reference points (see calibrate_ground_plane.py)")
            print("    2. Run: python calibrate_ground_plane.py")
            print("    3. Follow prompts to touch EE to ground marks")
            print()
            print("  System will continue WITHOUT calibration in 3 seconds...")
            print("="*60 + "\n")
            time.sleep(3)
            self.is_loaded = False
            return False

        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)

            self.yaw_offsets_rad = np.array([
                data['yaw_offsets_rad']['boom'],
                data['yaw_offsets_rad']['arm'],
                data['yaw_offsets_rad']['bucket']
            ], dtype=np.float32)

            self.calibration_date = data.get('timestamp', 'Unknown date')

            print("\n" + "="*60)
            print("✓ IMU CALIBRATION LOADED")
            print("="*60)
            print(f"  File: {self.calibration_file}")
            print(f"  Date: {self.calibration_date}")
            print()
            print("  Yaw Offsets:")
            print(f"    Boom:   {np.degrees(self.yaw_offsets_rad[0]):7.2f}°")
            print(f"    Arm:    {np.degrees(self.yaw_offsets_rad[1]):7.2f}°")
            print(f"    Bucket: {np.degrees(self.yaw_offsets_rad[2]):7.2f}°")
            print("="*60 + "\n")

            self.is_loaded = True
            return True

        except Exception as e:
            print("\n" + "="*60)
            print("✗ ERROR LOADING IMU CALIBRATION")
            print("="*60)
            print(f"  File: {self.calibration_file}")
            print(f"  Error: {e}")
            print("  System will continue WITHOUT calibration...")
            print("="*60 + "\n")
            time.sleep(2)
            self.is_loaded = False
            return False

    def apply_yaw_corrections(self, imu_quats):
        """Apply yaw corrections to raw IMU quaternions.

        Args:
            imu_quats: List of 3 quaternions [boom, arm, bucket] from IMUs

        Returns:
            Corrected quaternions with yaw offsets applied
        """
        if not self.is_loaded:
            print("⚠ No calibration loaded, returning uncorrected quaternions")
            return imu_quats

        corrected = []
        for i, raw_quat in enumerate(imu_quats):
            # Create yaw correction quaternion (Z-axis rotation)
            yaw_correction = quat_from_axis_angle(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                self.yaw_offsets_rad[i]
            )

            # Apply correction: corrected = yaw_correction * raw
            corrected_quat = quat_multiply(yaw_correction, raw_quat)
            corrected.append(quat_normalize(corrected_quat))

        return corrected


# Global calibration instance (singleton pattern)
_calibration = None

def get_calibration():
    """Get global calibration instance (lazy initialization)."""
    global _calibration
    if _calibration is None:
        _calibration = IMUCalibration()
    return _calibration

def apply_calibration(imu_quats):
    """Convenience function to apply calibration."""
    calib = get_calibration()
    return calib.apply_yaw_corrections(imu_quats)
