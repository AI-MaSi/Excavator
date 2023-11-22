import random
# import struct

from config import multiplexer_channels, tca_address # bno08x_address

try:
    import board
    import adafruit_tca9548a
    from adafruit_lsm6ds.ism330dhcx import ISM330DHCX

    """
    BNO HAS PROBLEMS!!!!!!!!
    from adafruit_bno08x.i2c import BNO08X_I2C  # Make sure you have this module
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_MAGNETOMETER,
        BNO_REPORT_ROTATION_VECTOR,
    )
    """
    IMU_MODULES_AVAILABLE = True
except ImportError:
    IMU_MODULES_AVAILABLE = False


bno_features = [
                "BNO_REPORT_ACCELEROMETER",
                "BNO_REPORT_GYROSCOPE",
                "BNO_REPORT_MAGNETOMETER",
                "BNO_REPORT_ROTATION_VECTOR"
            ]

class IMUSensorManager:
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.multiplexer_channels = multiplexer_channels
        self.bno08x = None

        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                self.i2c = board.I2C()
                self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=tca_address)
                self.sensors = {}
                self.initialize_ism330(multiplexer_channels)
                # self.initialize_bno08(bno08x_address)
            else:
                raise IMUmodulesNotAvailableError("IMU-modules are not available but required for non-simulation mode.")
        elif self.simulation_mode:
            print("Simulation mode activated! Simulated sensor values will be used.")

    def initialize_ism330(self, channels):
        for channel in channels:
            try:
                self.sensors[channel] = ISM330DHCX(self.tca[channel])
                print(f"ISM330DHCX on channel {channel} initialized.")
            except Exception as e:
                raise ISM330InitializationError(f"Error initializing ISM330DHCX on channel {channel}: {e}")

    """
    def initialize_bno08(self, address):
        try:
            self.bno08x = BNO08X_I2C(self.i2c, address=address)

            # not tested
            for feature in bno_features:
                self.bno08x.enable_feature(feature)
            print(f"BNO08x sensor initialized.")
            # find right exception
        except Exception as e:
            raise BNO08xInitializationError(f"Error initializing BNO08x sensor: {e}")

    @staticmethod
    def pack_data(data):
        # little endian, doubles. Because Mevea.
        format_str = '<' + 'd' * len(data)
        return struct.pack(format_str, *data)
    """
    def read_all(self):
        # Combined data from all sensors
        # BNO has problems, skipping it
        combined_data = []

        # Read data from each ISM330 sensor
        for channel in multiplexer_channels:
            try:
                ism330_data = self.read_ism330(channel)
                combined_data.extend(ism330_data)
            except ISM330ReadError as e:
                print(f"Failed to read from ISM330 sensor at channel {channel}: {e}")


        """
        # Read data from the BNO08 sensor
        try:
            bno08_data = self.read_bno08()
            combined_data.extend(bno08_data)
        except BNO08xReadError as e:
            print(f"Failed to read from BNO08 sensor: {e}")
            # Optionally handle the error or re-raise it
        """

        return combined_data

    def read_ism330(self, channel):
        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                try:
                    sensor = self.sensors[channel]
                    accel_x, accel_y, accel_z = sensor.acceleration
                    gyro_x, gyro_y, gyro_z = sensor.gyro
                    data = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

                # adafruit sensor exception
                except Exception as e:
                    raise ISM330ReadError(f"Error reading from ISM330DHCX on channel {channel}: {e}")
            else:
                raise ISM330ReadError("ISM330DHCX drivers are not available or simulation mode is not enabled.")

        else:
            accel_x, accel_y, accel_z = [random.uniform(0, 10) for _ in range(3)]
            gyro_x, gyro_y, gyro_z = [random.uniform(0, 10) for _ in range(3)]
            data = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

        #return self.pack_data(data) if pack else data
        # return list(data)
        return data

    def read_bno08(self):

        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                try:
                    accel_x, accel_y, accel_z = self.bno08x.acceleration  # pylint:disable=no-member
                    gyro_x, gyro_y, gyro_z = self.bno08x.gyro  # pylint:disable=no-member
                    mag_x, mag_y, mag_z = self.bno08x.magnetic  # pylint:disable=no-member
                    quat_i, quat_j, quat_k, quat_real = self.bno08x.quaternion  # pylint:disable=no-member

                    data = (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
                            mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real)
                #find specific exception
                except Exception as e:
                    raise BNO08xReadError(f"Error reading from BNO08x sensor: {e}")
            else:
                raise BNO08xReadError("BNO08x drivers are not available or simulation mode is not enabled.")
        else:
            accel_x, accel_y, accel_z = [random.uniform(0, 10) for _ in range(3)]
            gyro_x, gyro_y, gyro_z = [random.uniform(0, 10) for _ in range(3)]
            mag_x, mag_y, mag_z = [random.uniform(0, 10) for _ in range(3)]
            quat_i, quat_j, quat_k, quat_real = [random.uniform(0, 10) for _ in range(4)]

            data = (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
                    mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real)

        # return self.pack_data(data) if pack else data
        # return list(data)
        return data


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

class IMUmodulesNotAvailableError(Exception):
    pass

class ISM330InitializationError(Exception):
    pass

class ISM330ReadError(Exception):
    pass

class BNO08xInitializationError(Exception):
    pass

class BNO08xReadError(Exception):
    pass
