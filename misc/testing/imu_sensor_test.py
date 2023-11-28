
from time import sleep

import board
import adafruit_tca9548a
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
from adafruit_bno08x.i2c import BNO08X_I2C  # Make sure you have this module
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)

class IMUSensorManager:
    def __init__(self, tca_address=0x71, channels=range(8), bno08x_address=0x4a):
        self.i2c = board.I2C()
        self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=tca_address)
        self.sensors = {}
        #self.initialize_ism330(channels)
        self.initialize_bno08(bno08x_address)

    def initialize_ism330(self, channels):
        for channel in channels:
            try:
                self.sensors[channel] = ISM330DHCX(self.tca[channel])
                print(f"ISM330DHCX on channel {channel} initialized.")
            except Exception as e:
                print(f"An error occurred while initializing the sensor on channel {channel}: {e}")

    def initialize_bno08(self, address):
        try:
            self.bno08x = BNO08X_I2C(self.i2c, address=address)
            self.bno08x.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno08x.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno08x.enable_feature(BNO_REPORT_MAGNETOMETER)
            self.bno08x.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            print(f"BNO08x sensor initialized.")
        except Exception as e:
            print(f"An error occurred while initializing the BNO08x sensor: {e}")

    def read_ism330(self, channel):
        try:
            sensor = self.sensors[channel]

            # Read and print accelerometer and gyroscope data
            accel_x, accel_y, accel_z = sensor.acceleration
            gyro_x, gyro_y, gyro_z = sensor.gyro

            print(f"ISM330DHCX {channel} - Acceleration (m/s^2): ({accel_x}, {accel_y}, {accel_z})")
            print(f"ISM330DHCX {channel} - Gyroscope (dps): ({gyro_x}, {gyro_y}, {gyro_z})")

        except Exception as e:
            print(f"An error occurred while reading from channel {channel}: {e}")

    def read_bno08(self):
        try:
            print("Acceleration:")
            accel_x, accel_y, accel_z = self.bno08x.acceleration  # pylint:disable=no-member
            print("X: %0.6f  Y: %0.6f Z: %0.6f  m/s^2" % (accel_x, accel_y, accel_z))
            print("")

            print("Gyro:")
            gyro_x, gyro_y, gyro_z = self.bno08x.gyro  # pylint:disable=no-member
            print("X: %0.6f  Y: %0.6f Z: %0.6f rads/s" % (gyro_x, gyro_y, gyro_z))
            print("")

            print("Magnetometer:")
            mag_x, mag_y, mag_z = self.bno08x.magnetic  # pylint:disable=no-member
            print("X: %0.6f  Y: %0.6f Z: %0.6f uT" % (mag_x, mag_y, mag_z))
            print("")

            print("Rotation Vector Quaternion:")
            quat_i, quat_j, quat_k, quat_real = self.bno08x.quaternion  # pylint:disable=no-member
            print(
                "I: %0.6f  J: %0.6f K: %0.6f  Real: %0.6f" % (quat_i, quat_j, quat_k, quat_real)
            )
            print("")

        except Exception as e:
            print(f"An error occurred while reading from BNO08x: {e}")


# Example usage:
imu_manager = IMUSensorManager(channels=[0, 1, 2])
sleep(2)


while True:
    for channel in [0, 1, 2]:
        imu_manager.read_ism330(channel)
    print("-------------------------------------------------")
    imu_manager.read_bno08()
    sleep(1)
