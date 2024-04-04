# shut up excavator yes beep beep f you too
# this uses the same config file as the masi_driver
# set to run every time excavator boots up, after service "show_ip.service"
from adafruit_servokit import ServoKit
import sys
import os
import yaml

# Define path
excavator_main_dir = '/home/pi/GitHub/Excavator/main'
sys.path.append(excavator_main_dir)

# Define the configuration file
config_file_name = 'driver_config_Motionplatform.yaml'

# Construct the full path
config_file_path = os.path.join(excavator_main_dir, config_file_name)

# Load configurations from the .yaml file using the full path
with open(config_file_path, 'r') as file:
    configs = yaml.safe_load(file)
    channel_configs = configs['CHANNEL_CONFIGS']


kit = ServoKit(channels=16)
center_val_servo = 90

if 'pump' in channel_configs and channel_configs['pump']['type'] == 'pump':
	kit.continuous_servo[channel_configs['pump']['output_channel']].throttle = -1

# resetting these not needed but why not
# this does not care about offset
for channel_name, config in channel_configs.items():
	if config['type'] == 'angle':
		kit.servo[config['output_channel']].angle = center_val_servo
