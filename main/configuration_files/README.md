# Excavator Control System Configuration Files

This repository contains the configuration files for the excavator control system. These YAML files are used to set up and customize the behavior of various control modules.

## Key Configuration Areas

- **PWM Control**: Fine-tune servo and motor responses
- **Sensor Calibration**: Adjust sensor readings for accuracy


## Usage

1. Review and modify these YAML files to get wanted behaviour. You can add or remove settings as needed (e.g. add new sensors, or remove PWM channels).
2. Ensure these configuration files are placed in the `configuration_files/` directory of your project.
3. The control modules will read these configurations on startup to initialize the system correctly.

## Important Notes

- Be cautious when modifying these files, as incorrect settings can lead to unexpected behavior.

For detailed information on each setting, refer to the comments within the YAML files.
