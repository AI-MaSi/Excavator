import board
import adafruit_tca9548a
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
import time


class IMUReader:
    def __init__(self):
        self.i2c = board.I2C()
        self.mux = adafruit_tca9548a.TCA9548A(self.i2c, address=0x71)
        self.imus = {}
        self.init_imus()

    def init_imus(self):
        """Initialize all available IMUs on multiplexer"""
        imu_count = 0
        for channel in range(8):  # TCA9548A has 8 channels
            try:
                channel_bus = self.mux[channel]
                imu = ISM330DHCX(channel_bus)
                # Test read to verify it works
                imu.acceleration
                self.imus[imu_count] = {'channel': channel, 'imu': imu}
                print(f"Initialized IMU_{imu_count} on channel {channel}")
                imu_count += 1
            except Exception as e:
                continue

    def read(self,decimals=3):
        """
        Read all values from available IMUs
        Returns dict with IMU_[n] as keys, each containing gyro and accel data
        """
        values = {}

        for imu_num, imu_data in self.imus.items():
            try:
                imu = imu_data['imu']
                accel = imu.acceleration
                gyro = imu.gyro

                values[f'IMU_{imu_num}'] = {
                    'accel_x': round(accel[0], decimals),
                    'accel_y': round(accel[1], decimals),
                    'accel_z': round(accel[2], decimals),
                    'gyro_x': round(gyro[0], decimals),
                    'gyro_y': round(gyro[1], decimals),
                    'gyro_z': round(gyro[2], decimals),
                }
            except Exception as e:
                print(f"Error reading IMU_{imu_num}: {e}")
                values[f'IMU_{imu_num}'] = None

        return values
