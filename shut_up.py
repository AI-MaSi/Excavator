# shuts up the startup beep
from modules.PCA9685_controller import PWMController as P
_ = P(config_file="configuration_files/servo_config_200.yaml")

