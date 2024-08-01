import yaml
from ADCPi import ADCPi

class SensorManager:
    def __init__(self, config_file, decimals_pressure=3, decimals_angle=2):
        self.decimals_pressure = decimals_pressure
        self.decimals_angle = decimals_angle

        # Load configs from .yaml file
        with open(config_file, 'r') as file:
            configs = yaml.safe_load(file)
            self.pressure_sensor_configs = configs.get('PRESSURE_SENSORS', {})
            self.angle_sensor_configs = configs.get('ANGLE_SENSORS', {})

            # ADCPi settings
            self.i2c_addresses = configs['ADC_CONFIG']['i2c_addresses']
            self.pga_gain = configs['ADC_CONFIG']['pga_gain']
            self.bit_rate = configs['ADC_CONFIG']['bit_rate']
            self.conversion_mode = configs['ADC_CONFIG']['conversion_mode']

        # Initialize a dictionary to hold ADC objects
        self.adcs = {}
        self.initialized = False
        self.min_voltage = {}

        # Initialize data storage
        self.pressure_data = {}
        self.angle_data = {}
        self.calibrated_angle_data = {}

        # Initialize ADC instances
        self.initialize_adc()

    def initialize_adc(self):
        board_needs_initialization = {board_name: False for board_name in self.i2c_addresses.keys()}

        for sensor_config in self.pressure_sensor_configs.values():
            board_name = sensor_config['input'][0]
            if board_name in board_needs_initialization:
                board_needs_initialization[board_name] = True

        for sensor_config in self.angle_sensor_configs.values():
            board_name = sensor_config['input'][0]
            if board_name in board_needs_initialization:
                board_needs_initialization[board_name] = True

        print(f"Boards needed to init: {board_needs_initialization}")

        for board_name, need_init in board_needs_initialization.items():
            if need_init:
                addr1, addr2 = self.i2c_addresses[board_name]

                try:
                    # Create an instance of ADCPi with the addresses
                    adc_instance = ADCPi(addr1, addr2, self.bit_rate)
                    adc_instance.set_conversion_mode(self.conversion_mode)
                    adc_instance.set_pga(self.pga_gain)

                    # Store the initialized instance in the dictionary
                    self.adcs[board_name] = adc_instance

                    print(f"Initialized {board_name} with addresses {hex(addr1)}, {hex(addr2)}")
                    print(f"PGA Gain: {self.pga_gain}, Bit Rate: {self.bit_rate} bits, Conversion Mode: {self.conversion_mode}")
                except OSError as e:
                    print(f"Failed to initialize ADCPi for {board_name}! Error: {e}")
                    return

        self.initialized = True

    def read_pressure(self, return_names=False, alpha=0.9):
        if self.initialized and self.pressure_sensor_configs:
            sensor_data = []
            for sensor_name, sensor_config in self.pressure_sensor_configs.items():
                board_name = sensor_config['input'][0]
                channel = sensor_config['input'][1]
                adc_instance = self.adcs[board_name]

                voltage = adc_instance.read_voltage(channel)

                if voltage is None or voltage < 0:
                    print(f"Invalid voltage reading for sensor {sensor_name}: {voltage}")
                    value = None
                elif voltage > 0.40:
                    calibrated_value = round(
                        (1000 * (voltage - 0.5) / (4.5 - 0.5)) * sensor_config['calibration_value'],
                        self.decimals_pressure)

                    if sensor_name not in self.pressure_data:
                        self.pressure_data[sensor_name] = []
                    self.pressure_data[sensor_name].append(calibrated_value)
                    filtered_value = self.low_pass_filter(self.pressure_data[sensor_name], alpha)[-1]
                    value = filtered_value
                else:
                    value = 0

                if return_names:
                    sensor_data.append((sensor_name, sensor_config.get('name', ''), value))
                else:
                    sensor_data.append(value)

            return sensor_data
        elif not self.pressure_sensor_configs:
            print("No pressure sensors configured.")
            return []
        else:
            print("ADCPi not initialized!")
            return None

    def read_angle(self, return_names=False, alpha=0.5):
        if not self.initialized:
            print("ADCPi not initialized!")
            return None
        if not self.angle_sensor_configs:
            print("No angle sensors configured.")
            return []

        sensor_data = []
        for sensor_name, sensor_config in self.angle_sensor_configs.items():
            board_name = sensor_config['input'][0]
            channel = sensor_config['input'][1]
            adc_instance = self.adcs[board_name]

            voltage = adc_instance.read_voltage(channel)

            if voltage is None or voltage < 0:
                print(f"Invalid voltage reading for sensor {sensor_name}: {voltage}")
                filtered_angle = None
                filtered_calibrated_angle = None
            else:
                steps_per_revolution = sensor_config.get('steps_per_revolution', 360)
                angle = round((voltage - 0.5) * steps_per_revolution / (4.5 - 0.5), self.decimals_angle)

                if self.min_voltage.get(sensor_name) is None or voltage < self.min_voltage[sensor_name]:
                    self.min_voltage[sensor_name] = voltage

                calibrated_angle = round(
                    angle - (self.min_voltage[sensor_name] - 0.5) * steps_per_revolution / (4.5 - 0.5),
                    self.decimals_angle)

                if sensor_name not in self.angle_data:
                    self.angle_data[sensor_name] = []
                    self.calibrated_angle_data[sensor_name] = []
                self.angle_data[sensor_name].append(angle)
                self.calibrated_angle_data[sensor_name].append(calibrated_angle)

                filtered_angle = self.low_pass_filter(self.angle_data[sensor_name], alpha)[-1]
                filtered_calibrated_angle = self.low_pass_filter(self.calibrated_angle_data[sensor_name], alpha)[-1]

            if return_names:
                sensor_data.append(
                    (sensor_name, sensor_config.get('name', ''), filtered_angle, filtered_calibrated_angle))
            else:
                sensor_data.append((filtered_angle, filtered_calibrated_angle))

        return sensor_data

    @staticmethod
    def low_pass_filter(data, alpha):
        """
        Applies a low-pass filter to the input data using an exponential moving average.

        Parameters:
        - data: List of numerical values representing the raw data.
        - alpha: Smoothing factor (0 < alpha < 1). Higher alpha means more smoothing.

        Returns:
        - List of filtered data.
        """
        if not data:
            return []

        filtered_data = [data[0]]

        for i in range(1, len(data)):
            filtered_value = alpha * data[i] + (1 - alpha) * filtered_data[-1]
            filtered_data.append(filtered_value)

        return filtered_data
