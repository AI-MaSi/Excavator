import yaml
import adafruit_tca9548a
from adafruit_lsm6ds import Rate
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
import board

class ISM330DHCX:
    def __init__(self, config_file, decimals=2):

        # Load configs from .yaml file
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.sensor_configs = configs['IMU_CONFIG']

        self.decimals = decimals
        self.i2c = board.I2C()
        self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=self.sensor_configs['multiplexer_address'])
        self.sensors = {}
        self.__initialize_ism330()

    def __initialize_ism330(self):
        multiplexer_channels = self.sensor_configs['multiplexer_channels']
        for channel in multiplexer_channels:
            try:
                self.sensors[channel] = ISM330DHCX(self.tca[channel])

                # Set the data rates for accelerometer and gyro from config
                accel_data_rate = eval(self.config['IMU_CONFIG']['accelerometer_data_rate'])
                gyro_data_rate = eval(self.config['IMU_CONFIG']['gyro_data_rate'])

                self.sensors[channel].accelerometer_data_rate = accel_data_rate
                self.sensors[channel].gyro_data_rate = gyro_data_rate

                print(f"ISM330DHCX on channel {channel} initialized.")
            except Exception as e:
                error_msg = f"Error initializing ISM330DHCX on channel {channel}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)

    def read_all(self):
        combined_data = []

        # Read data from each ISM330 sensor
        for channel in self.sensors:
            try:
                ism330_data = self.read_ism330(channel)
                combined_data.extend(ism330_data)
            except Exception as e:
                error_msg = f"Failed to read from ISM330 sensor at channel {channel}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)

        return combined_data

    def read_ism330(self, channel):
        try:
            sensor = self.sensors[channel]
            accel_x, accel_y, accel_z = [round(val, self.decimals) for val in sensor.acceleration]
            gyro_x, gyro_y, gyro_z = [round(val, self.decimals) for val in sensor.gyro]
            data = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

        except Exception as e:
            error_msg = f"Error reading from ISM330DHCX on channel {channel}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)

        return data