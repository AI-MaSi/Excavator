# shut up excavator yes beep beep f you too
# this uses the same config file as the masi_driver
# set to run every time excavator boots up, after service "show_ip.service"
from adafruit_servokit import ServoKit
import sys

sys.path.append('/home/pi/GitHub/Excavator/main')

from config import CHANNEL_CONFIGS, center_val_servo
kit = ServoKit(channels=16)

if 'pump' in CHANNEL_CONFIGS and CHANNEL_CONFIGS['pump']['type'] == 'throttle':
	kit.continuous_servo[CHANNEL_CONFIGS['pump']['output_channel']].throttle = -1

# resetting these not needed but why not
# this does not care about offset
for channel_name, config in CHANNEL_CONFIGS.items():
	if config['type'] == 'angle':
		kit.servo[config['output_channel']].angle = center_val_servo
