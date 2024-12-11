# stop the beeping at startup
import sys
#import os

# Get the absolute path of the directory containing the module
#module_path = os.path.abspath(os.path.join('..', 'main', 'control_modules'))

module_path = 'home/kaivuri/Documents/masi/main/control_modules'

# Add it to the Python path
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import the module
from PCA9685_controller import PWM_controller

path = '../main/configuration_files/channel_configs.yaml'

pwm = PWM_controller(
    config_file=path,
    simulation_mode=False,
	input_rate_threshold=0,	# set rate to 0 (or None) to disable rate monitoring
)

# no need to explicitly call pwm.reset() as it is called in the constructor
# also no need to call pwm.stop_monitoring() as input threshold is set to 0