import random
from time import time, sleep
import threading
import yaml




try:
    import board
    import adafruit_tca9548a
    from adafruit_lsm6ds import Rate
    from adafruit_lsm6ds.ism330dhcx import ISM330DHCX

    import RPi.GPIO as GPIO
    from ADCPi import ADCPi

    IMU_MODULES_AVAILABLE = True
    GPIO.cleanup()
except ImportError as e:
    print(f"Failed to import modules: {e}")
    IMU_MODULES_AVAILABLE = False


class IMUSensorManager:
    # bno08x removed 5.2.2024
    def __init__(self, simulation_mode=False, decimals=2, tca_address=0x71):
        self.simulation_mode = simulation_mode
        self.multiplexer_channels = [0, 1, 2, 3]
        self.decimals = decimals
        self.bno08x = None

        if not self.simulation_mode:
            if IMU_MODULES_AVAILABLE:
                self.i2c = board.I2C()
                self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=tca_address)
                self.sensors = {}
                self.initialize_ism330(self.multiplexer_channels)

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

    def read_all(self):
        combined_data = []

        # Read data from each ISM330 sensor
        for channel in self.multiplexer_channels:
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


class IMUmodulesNotAvailableError(Exception):
    pass


class ISM330InitializationError(Exception):
    pass


class ISM330ReadError(Exception):
    pass


class RPMSensor:
    def __init__(self, sensor_pin=4, magnets=14, decimals=2):
        self.sensor_pin = sensor_pin
        self.magnets = magnets
        self.pulse_count = 0
        self.rpm = 0
        self.decimals = decimals
        self.setup_gpio()
        self.start_measurement()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.sensor_pin, GPIO.FALLING, callback=self.sensor_callback)

    def sensor_callback(self): #channel removed
        self.pulse_count += 1

    def calculate_rpm(self):
        last_checked_time = time()
        while True:
            current_time = time()
            elapsed_time = current_time - last_checked_time

            if elapsed_time >= 1:  # Update every second
                self.rpm = (self.pulse_count / self.magnets) * 60 / elapsed_time
                self.pulse_count = 0
                last_checked_time = current_time

            sleep(0.1)

    def start_measurement(self):
        self.thread = threading.Thread(target=self.calculate_rpm)
        self.thread.daemon = True
        self.thread.start()

    def get_rpm(self):
        return round(self.rpm, self.decimals)


    def cleanup(self):
        GPIO.cleanup(self.sensor_pin)


class PressureSensor:
    def __init__(self, decimals=3):

        config_file = 'sensor_config.yaml'

        # Load configs from .yaml file
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.sensor_configs = configs['PRESSURE_SENSORS']

        # ADCPi settings
        i2c_addresses = configs['ADC_CONFIG']['i2c_addresses']
        pga_gain = configs['ADC_CONFIG']['pga_gain']
        bit_rate = configs['ADC_CONFIG']['bit_rate']
        conversion_mode = configs['ADC_CONFIG']['conversion_mode']

        board_needs_initialization = {board_name: False for board_name in i2c_addresses.keys()}

        for sensor_config in self.sensor_configs.values():
            board_name = sensor_config['input'][0]
            if board_name in board_needs_initialization:

                board_needs_initialization[board_name] = True

        print(f"Boards needed to init: {board_needs_initialization}")

        # Initialize a dictionary to hold ADC objects
        self.adcs = {}

        self.decimals = decimals
        self.initialized = False

        for board_name, need_init in board_needs_initialization.items():
            if need_init:
                addr1, addr2 = i2c_addresses[board_name]

                try:
                    # Create an instance of ADCPi with the addresses
                    adc_instance = ADCPi(addr1, addr2, bit_rate)
                    adc_instance.set_conversion_mode(conversion_mode)
                    adc_instance.set_pga(pga_gain)

                    # Store the initialized instance in the dictionary
                    self.adcs[board_name] = adc_instance

                    print(f"Initialized {board_name} with addresses {hex(addr1)}, {hex(addr2)}")
                    print(f"PGA Gain: {pga_gain}, Bit Rate: {bit_rate} bits, Conversion Mode: {conversion_mode}")
                except OSError as e:
                    print(f"Failed to initialize ADCPi for {board_name}! Error: {e}")
                    return

        self.initialized = True

    def read_pressure(self, return_names=False):
        if self.initialized:
            sensor_data = []
            # Iterate through each sensor configured
            for sensor_name, sensor_config in self.sensor_configs.items():
                board_name = sensor_config['input'][0]  # Get the board name from sensor input config
                channel = sensor_config['input'][1]  # Get the channel from sensor input config
                adc_instance = self.adcs[board_name]  # Retrieve the appropriate ADCPi instance by board name

                # Read voltage from the specified channel on the appropriate ADCPi instance
                voltage = adc_instance.read_voltage(channel)

                # Process the voltage reading
                if voltage > 0.40:  # filter out under 0.4V
                    # Round values and convert the voltages to psi (rough)
                    # Apply calibration value
                    calibrated_value = round(
                        (1000 * (voltage - 0.5) / (4.5 - 0.5)) * sensor_config['calibration_value'], self.decimals)
                    value = calibrated_value
                else:
                    value = 0

                if return_names:
                    # Include sensor name and value for debugging
                    sensor_data.append((sensor_name, sensor_config['name'], value))
                else:
                    # Include only the value if not in debug mode
                    sensor_data.append(value)

            # Directly return the sensor data list
            return sensor_data
        else:
            print("ADCPi not initialized!")
            return None


class CenterPositionSensor:
    def __init__(self, sensor_pin=17):
        self.sensor_pin = sensor_pin
        self.setup_gpio()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def check_center_position(self):
        if GPIO.input(self.sensor_pin):
            return True  # Sensor is in center position
        else:
            return False

    def cleanup(self):
        GPIO.cleanup(self.sensor_pin)