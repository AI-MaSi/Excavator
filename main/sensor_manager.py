import random
import time
import threading
from config import multiplexer_channels, tca_address, bno08x_address

try:
    import board
    import adafruit_tca9548a
    from adafruit_lsm6ds import Rate
    from adafruit_lsm6ds.ism330dhcx import ISM330DHCX

    import RPi.GPIO as GPIO
    from ADCPi import ADCPi
    """
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_MAGNETOMETER,
        BNO_REPORT_ROTATION_VECTOR,
            )
    """
    IMU_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import modules: {e}")
    IMU_MODULES_AVAILABLE = False

"""
bno_features = [
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
            ]
"""
class IMUSensorManager:
    def __init__(self, simulation_mode=False, decimals=2):
        self.simulation_mode = simulation_mode
        self.multiplexer_channels = multiplexer_channels
        self.decimals = decimals
        self.bno08x = None

        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                self.i2c = board.I2C()
                self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=tca_address)
                self.sensors = {}
                self.initialize_ism330(multiplexer_channels)

                # refresh rate too slow!
                # self.initialize_bno08(bno08x_address)
                # time.sleep(1)
            else:
                raise IMUmodulesNotAvailableError("IMU-modules are not available but required for non-simulation mode.")
        elif self.simulation_mode:
            print("Simulation mode activated! Simulated sensor values will be used.")

    def initialize_ism330(self, channels):
        for channel in channels:
            try:
                self.sensors[channel] = ISM330DHCX(self.tca[channel])

                # Set the data rates for accelerometer and gyro. 26Hz
                self.sensors[channel].accelerometer_data_rate = Rate.RATE_26_HZ
                self.sensors[channel].gyro_data_rate = Rate.RATE_26_HZ

                print(f"ISM330DHCX on channel {channel} initialized.")
            except Exception as e:
                raise ISM330InitializationError(f"Error initializing ISM330DHCX on channel {channel}: {e}")

    """
    def initialize_bno08(self, address):
        try:
            self.bno08x = BNO08X_I2C(self.i2c, address=address)
            self.bno08x.initialize()

            for feature in bno_features:
                self.bno08x.enable_feature(feature)

            print(f"BNO08x sensor initialized.")
            # find right exception
        except Exception as e:
            raise BNO08xInitializationError(f"Error initializing BNO08x sensor: {e}")
    """
    def read_all(self):
        combined_data = []

        # Read data from each ISM330 sensor
        for channel in multiplexer_channels:
            try:
                ism330_data = self.read_ism330(channel)
                combined_data.extend(ism330_data)
            except ISM330ReadError as e:
                print(f"Failed to read from ISM330 sensor at channel {channel}: {e}")

        # if self.bno08x is not None:
        #bno08_data = self.read_bno08()
        #combined_data.extend(bno08_data)
        return combined_data

    def read_ism330(self, channel):
        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                try:
                    sensor = self.sensors[channel]
                    accel_x, accel_y, accel_z = [round(val, self.decimals) for val in sensor.acceleration]
                    gyro_x, gyro_y, gyro_z = [round(val, self.decimals) for val in sensor.gyro]
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
                    mag_x, mag_y, mag_z = [round(val, self.decimals) for val in self.bno08x.magnetic]
                    quat_i, quat_j, quat_k, quat_real = [round(val, self.decimals) for val in self.bno08x.quaternion]
                    data = mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real
                except Exception as e:
                    print(e)
                    # find the right exceptions!
                    mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real = [0.0 for _ in range(7)]
                    data = mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real

                    self.bno08x.soft_reset()
                    print("soft reset")
                    self.bno08x.initialize()
                    print("re init")

            else:
                raise BNO08xReadError("BNO08x drivers are not available or simulation mode is not enabled.")
        else:

            mag_x, mag_y, mag_z = [random.uniform(0, 10) for _ in range(3)]
            quat_i, quat_j, quat_k, quat_real = [random.uniform(0, 10) for _ in range(4)]

            data = mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real

        return data

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

class RPMSensor:
    def __init__(self, rpm_sensor_pin=4, magnets=14, decimals=2):
        self.hall_sensor_pin = rpm_sensor_pin
        self.magnets = magnets
        self.pulse_count = 0
        self.rpm = 0
        self.decimals = decimals
        self.setup_gpio()
        self.start_measurement()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.hall_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.hall_sensor_pin, GPIO.FALLING, callback=self.sensor_callback)

    def sensor_callback(self, channel):
        self.pulse_count += 1

    def calculate_rpm(self):
        last_checked_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - last_checked_time

            if elapsed_time >= 1:  # Update every second
                self.rpm = (self.pulse_count / self.magnets) * 60 / elapsed_time
                self.pulse_count = 0
                last_checked_time = current_time

            time.sleep(0.1)

    def start_measurement(self):
        self.thread = threading.Thread(target=self.calculate_rpm)
        self.thread.daemon = True
        self.thread.start()

    def get_rpm(self):
        return round(self.rpm, self.decimals)

    def cleanup(self):
        GPIO.cleanup()

class PressureSensor:
    def __init__(self, i2c_addr=(0x6E, 0x6F), channels=6, decimals=2):
        self.channel_numbers = channels
        self.decimals = decimals
        self.data = [0] * self.channel_numbers
        self.initialized = False

        try:
            self.adc = ADCPi(i2c_addr[0], i2c_addr[1], 12)
            self.adc.set_conversion_mode(1)
            self.initialized = True
        except OSError as e:
            print(f"Failed to set I2C-address! Error: {e}")

    def read_pressure(self):
        if self.initialized:
            for i in range(1, self.channel_numbers + 1):
                voltage = self.adc.read_voltage(i)
                if voltage > 0.40:
                    # round values and convert the voltages to psi (rough)
                    self.data[i - 1] = round(((1000 * (voltage - 0.5) / (4.5 - 0.5))), self.decimals)
                else:
                    self.data[i - 1] = 0
            return self.data
        else:
            print("ADCPi not initialized!")
            return None

class CenterPositionSensor:
    def __init__(self, sensor_pin=17):
        # set the right GPIO pin
        pass

    def check_center_position(self):
        # check position
        pass