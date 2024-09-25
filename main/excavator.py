import asyncio
import logging
import yaml
import aiohttp
import time
import hashlib
import json
import struct

from control_modules import (
    ADC_sensors,
    PWM_controller,
    IMU_sensors,
    GPIO_sensors
)


class Excavator:
    def __init__(self, config_path: str = 'excavator_config.yaml'):
        self.logger = None
        self.config = None

        self.load_config(config_path)
        self.setup_logging()

        self.pwm = None
        self.adc = None
        self.imu = None
        self.rpm = None
        self.http_session = None

        self.last_sent_data = None
        self.last_received_timestamp = 0

        self._initialize_components()
        #self.print_input_mappings()

    def load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logging(self):
        logging.basicConfig(level=self.config['log_level'],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        self.logger.info("Initializing Excavator components...")

        # Initialize PWM controller
        self.pwm = PWM_controller.PWM_hat(
            config_file=self.config['pwm_config'],
            #pwm_inputs=self.config['pwm_inputs'],
            simulation_mode=self.config['simulation'],
            pump_variable=self.config['pump_variable'],
            tracks_disabled=self.config['tracks_disabled'],
            input_rate_threshold=self.config['input_rate_threshold'],
            deadzone=self.config['deadzone']
        )

        # Initialize ADC sensors
        self.adc = ADC_sensors.ADC_hat(
            config_file=self.config['adc_config'],
            decimals=self.config['decimals'],
            simulation_mode=self.config['simulation']
        )

        # Initialize IMU sensors
        self.imu = IMU_sensors.ISM330DHCX(
            config_file=self.config['imu_config'],
            decimals=self.config['decimals']
        )

        # Initialize GPIO sensors
        self.rpm = GPIO_sensors.RPMSensor(
            config_file=self.config['gpio_config'],
            sensor_name='RPM_pump',
            decimals=self.config['decimals']
        )

        self.logger.info("All components initialized successfully.")

    @staticmethod
    def calculate_checksum(values, timestamp):
        data = json.dumps(values) + str(timestamp)
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def handle_excavator_inputs(control_values):
        """
        Process and map the received control values here.
        """
        # (trackR, trackL, scoop, lift_boom, tilt_boom, center_rotate, aux1, aux2)
        processed_values = control_values[:8]  # Take first 8 values. Zero based index

        # Add any additional processing logic here if needed
        return processed_values

    async def start(self):
        self.logger.info("Starting Excavator system...")
        try:
            self.http_session = aiohttp.ClientSession()
            await asyncio.gather(
                self.send_loop(),
                self.receive_loop()
            )
        except Exception as e:
            self.logger.error(f"Error during Excavator operation: {e}")
        finally:
            await self.stop()

    async def stop(self):
        self.logger.info("Stopping Excavator system...")
        try:
            self.pwm.reset(reset_pump=True)
            self.rpm.cleanup()
            if self.http_session:
                await self.http_session.close()
        except Exception as e:
            self.logger.error(f"Error during Excavator shutdown: {e}")
        finally:
            self.logger.info("Excavator system stopped.")

    async def send_loop(self):
        self.logger.info("Starting sensor data transmission...")
        send_period = 1.0 / self.config['send_frequency']
        server_url = f"http://{self.config['addr']}:{self.config['port']}/send/excavator"

        while True:
            try:
                loop_start = asyncio.get_event_loop().time()

                pressures = self.adc.read_scaled()
                #self.logger.debug(f"ADC pressures: {pressures}")
                pump_rpm = self.rpm.read_rpm()
                #self.logger.debug(f"Pump RPM: {pump_rpm}")
                imu_data = self.imu.read_all(read_mode='raw')
                #self.logger.debug(f"IMU data: {imu_data}")

                sensor_data = {
                    'pressures': pressures,
                    'pump_rpm': pump_rpm,
                    'imu_data': imu_data
                }

                self.logger.debug(f"Collected sensor data: {sensor_data}")

                # Skip sending if the data hasn't changed
                if sensor_data == self.last_sent_data:
                    self.logger.debug("Skipping send - no new data")
                    await asyncio.sleep(send_period)
                    continue

                timestamp = time.time()
                checksum = Excavator.calculate_checksum(sensor_data, timestamp)

                payload = {
                    "values": sensor_data,
                    "timestamp": timestamp,
                    "checksum": checksum
                }

                json_payload = json.dumps(payload)
                message_size = len(json_payload)
                size_header = struct.pack('>I', message_size)

                self.logger.debug(f"Total payload sensor data: {payload}")

                async with self.http_session.post(server_url, data=size_header + json_payload.encode(),
                                                  headers={'Content-Type': 'application/octet-stream'}) as response:
                    if response.status == 200:
                        self.logger.debug(f"Sent sensor data: {sensor_data}")
                        self.last_sent_data = sensor_data
                    else:
                        self.logger.error(f"Failed to send sensor data. Status: {response.status}")

                elapsed_time = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0.0, send_period - elapsed_time)
                await asyncio.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"Error in sensor data transmission: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    async def receive_loop(self):
        self.logger.info("Starting drive loop...")
        receive_period = 1.0 / self.config['receive_frequency']
        server_url = f"http://{self.config['addr']}:{self.config['port']}/receive/excavator"

        while True:
            try:
                loop_start = asyncio.get_event_loop().time()

                async with self.http_session.get(server_url) as response:
                    if response.status == 200:
                        size_header = await response.content.read(4)
                        message_size = struct.unpack('>I', size_header)[0]
                        json_data = await response.content.read(message_size)
                        control_data = json.loads(json_data.decode())

                        # Verify timestamp and checksum
                        if control_data['timestamp'] <= self.last_received_timestamp:
                            self.logger.debug("Skipping receive - old data")
                        else:
                            calculated_checksum = Excavator.calculate_checksum(control_data['values'],
                                                                          control_data['timestamp'])
                            if calculated_checksum != control_data['checksum']:
                                self.logger.error("Checksum mismatch in received data")
                            else:
                                self.last_received_timestamp = control_data['timestamp']

                                #print(f"Received control data: {control_data['values']}")

                                # Process the control data
                                processed_values = Excavator.handle_excavator_inputs(control_data['values'])

                                #print(f"Processed values: {processed_values}")
                                # Update PWM controller with processed values
                                angles = self.pwm.update_values(processed_values, return_servo_angles=True)
                                self.logger.debug(f"Servo angles: {angles}")

                    elif response.status == 204:
                        self.logger.debug("No new control data available")
                    else:
                        self.logger.error(f"Failed to receive control data. Status: {response.status}")

                elapsed_time = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0.0, receive_period - elapsed_time)
                await asyncio.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"Error in drive loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    def print_input_mappings(self):
        self.pwm.print_input_mappings()


async def main():
    excavator = Excavator()
    excavator.print_input_mappings()

    try:
        await excavator.start()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())