from control_modules import PCA9685_controller as test

# test the configuration file

# Initialize PWM controller, validation happens here
try:
    test = test.PWMcontroller(
        config_file='configuration_files/kone_config.yaml',
    )
except ValueError as e:
    print(f"Configuration file is invalid:\n{e}")
    exit(1)

print("Configuration file is valid!")
