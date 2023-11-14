
from time import sleep
import random
import struct

try:
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
    IMU_DRIVERS_AVAILABLE = True
except ImportError:
    IMU_DRIVERS_AVAILABLE = False


class IMUSensorManager:

    def __init__(self, multiplexer_channels=range(8), tca_address=0x71, bno08x_address=0x4a, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.multiplexer_channels = multiplexer_channels

        if not self.simulation_mode and IMU_DRIVERS_AVAILABLE:
            self.i2c = board.I2C()
            self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=tca_address)
            self.sensors = {}
            self.initialize_ism330(multiplexer_channels)
            self.initialize_bno08(bno08x_address)

        elif self.simulation_mode:
            print("Simulation mode activated! Simulated sensor values will be used.")

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

    def pack_data(self, data):
        # little endian, doubles. Because Mevea.
        format_str = '<' + 'd' * len(data)
        return struct.pack(format_str, *data)


    # not tested
    def read_all_and_pack(self):
        all_ism330_data = bytearray()
        for channel in self.multiplexer_channels:
            ism330_data = self.read_ism330(channel, pack=True)
            all_ism330_data.extend(ism330_data)

        bno08_data = self.read_bno08(pack=True)
        return all_ism330_data + bno08_data

    def read_ism330(self, channel, pack=False):

        if self.simulation_mode:
            accel_x, accel_y, accel_z = [random.uniform(0, 10) for _ in range(3)]
            gyro_x, gyro_y, gyro_z = [random.uniform(0, 10) for _ in range(3)]
            data = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

        elif not self.simulation_mode and IMU_DRIVERS_AVAILABLE:
            try:
                sensor = self.sensors[channel]
                accel_x, accel_y, accel_z = sensor.acceleration
                gyro_x, gyro_y, gyro_z = sensor.gyro
                data = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

            except Exception as e:
                print(f"An error occurred while reading from channel {channel}: {e}")
                data = None

        else:
            data = None

        return self.pack_data(data) if pack else data

    def read_bno08(self, pack=False):

        if self.simulation_mode:
            accel_x, accel_y, accel_z = [random.uniform(0, 10) for _ in range(3)]
            gyro_x, gyro_y, gyro_z = [random.uniform(0, 10) for _ in range(3)]
            mag_x, mag_y, mag_z = [random.uniform(0, 10) for _ in range(3)]
            quat_i, quat_j, quat_k, quat_real = [random.uniform(0, 10) for _ in range(4)]

            data = (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
                    mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real)

        elif not self.simulation_mode and IMU_DRIVERS_AVAILABLE:
            try:
                accel_x, accel_y, accel_z = self.bno08x.acceleration  # pylint:disable=no-member
                gyro_x, gyro_y, gyro_z = self.bno08x.gyro  # pylint:disable=no-member
                mag_x, mag_y, mag_z = self.bno08x.magnetic  # pylint:disable=no-member
                quat_i, quat_j, quat_k, quat_real = self.bno08x.quaternion  # pylint:disable=no-member

                data = (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
                        mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real)

            except Exception as e:
                print(f"An error occurred while reading: {e}")
                data = None

        else:
            data = None

        return self.pack_data(data) if pack else data

'''
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
     '''
