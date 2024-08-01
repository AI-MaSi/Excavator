from time import time, sleep
import threading
import yaml
import RPi.GPIO as GPIO

class RPMSensor:
    def __init__(self, config_file, sensor_name, decimals=2):
        # Load configs from .yaml file
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.sensor_configs = configs['RPM_SENSORS'][sensor_name]

        self.sensor_pin = self.sensor_configs['GPIO pin']
        self.magnets = self.sensor_configs['magnets']
        self.pulse_count = 0
        self.rpm = 0
        self.decimals = decimals
        self.__setup_gpio()
        self.__start_measurement()

    def __setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.sensor_pin, GPIO.FALLING, callback=self.__sensor_callback)

    def __sensor_callback(self, channel):
        self.pulse_count += 1

    def __calculate_rpm(self):
        last_checked_time = time()
        while True:
            current_time = time()
            elapsed_time = current_time - last_checked_time

            if elapsed_time >= 0.1:  # Update every 100ms
                self.rpm = (self.pulse_count / self.magnets) * 60 / elapsed_time
                self.pulse_count = 0
                last_checked_time = current_time

            sleep(0.001)

    def __start_measurement(self):
        self.thread = threading.Thread(target=self.__calculate_rpm)
        self.thread.daemon = True
        self.thread.start()

    def read_rpm(self):
        # return as a list so .extend() works
        return round(self.rpm, self.decimals)

    def cleanup(self):
        GPIO.cleanup(self.sensor_pin)

class CenterPositionSensor:
    def __init__(self, sensor_pin=17):
        self.sensor_pin = sensor_pin
        self.__setup_gpio()

    def __setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def check_center_position(self):
        if GPIO.input(self.sensor_pin):
            return True  # Sensor is in center position
        else:
            return False

    def cleanup(self):
        GPIO.cleanup(self.sensor_pin)