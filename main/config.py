# here you can set some values for the masi_client.py and masi_driver.py script
# shut_up.py also uses this at the startup


# host = '10.214.5.110'
local_addr = '127.0.0.1'
connect_addr = '127.0.0.1'
port = 5111
identification_number = 0 # 0 excavator, 1 Mevea, 2 Motion Platform
inputs = 20
outputs = 0

# '<QI20DB'
# '<'  Little-endian
endian_specifier = '<'
# 'Q' 8 byte integer (UNIX-timestamp)
unix_format = 'Q'
# 'I'  Unsigned int (sequence number) 4 bytes
sequence_format = 'I'
# 'i' Signed int (handshake) 4 bytes
handshake_format = 'i'
# 'd'  doubles (data) 8 bytes
data_format = 'd'
# 'B'  Unsigned char (checksum) 1 byte
checksum_format = 'B'

file_path = "log/example_data.bin"
BUFFER_SIZE = 10

# ISM330 IMU's use these multiplexer channels
multiplexer_channels = [1, 2, 3]

# Hal sensor GPIO pins
gpio_rpm = 4
gpio_center = 17

"""
CHANNEL CONFIGS

"channel"
PWM channels. 16 channels available, 9 used.

"type"
Type of the servo. angle / throttle / none. Set to 'none' if you wish to skip it.

"offset"
Offset from the center_val_servo value

"direction"
Set -1 to flip the direction

"multiplier"
Input multiplier

"idle"
Pump idle (lowest) speed

"variable"
Set the pump to change speed based on the channels used. 1 / 0 boolean.

------------------------------------------------------------------------------

VARIABLES

"hydraulic_multiplier_channels"
Channels that affect pump multiplier. lift_boom, tilt_boom, scoop, tool1, tool2

"center_val_servo"
Rough universal center value for all "angle" type servos

"deadzone"
Input center deadzone. 0...100(%)
"""


hydraulic_multiplier_channels = [2, 3, 4, 6, 7] #output channels

center_val_servo = 90

deadzone = 15

CHANNEL_CONFIGS = {
    'trackR': {'output_channel': 0, 'type': 'angle', 'offset': -1, 'direction': 1, 'multiplier': 20, 'input_channel': 6},
    'trackL': {'output_channel': 1, 'type': 'angle', 'offset': -1, 'direction': 1, 'multiplier': 20, 'input_channel': 7},
    'scoop': {'output_channel': 2, 'type': 'angle', 'offset': 8.5, 'direction': 1, 'multiplier': 20, 'input_channel': 0},
    'lift_boom': {'output_channel': 3, 'type': 'angle', 'offset': 8, 'direction': 1, 'multiplier': 20, 'input_channel': 1},
    'tilt_boom': {'output_channel': 4, 'type': 'angle', 'offset': 4, 'direction': 1, 'multiplier': 20, 'input_channel': 4},
    'rotate': {'output_channel': 5, 'type': 'angle', 'offset': 5, 'direction': 1, 'multiplier': 20, 'input_channel': 3},  # 'reset_offset': 5} removed, not tested
    'tool1': {'output_channel': 6, 'type': 'angle', 'offset': 0, 'direction': 1, 'multiplier': 20, 'input_channel': 5},
    'tool2': {'output_channel': 7, 'type': 'angle', 'offset': 0, 'direction': 1, 'multiplier': 20, 'input_channel': 2},
    'not used1': {'output_channel': 8, 'type': 'none'},
    'pump': {'output_channel': 9, 'type': 'throttle', 'offset': 0, 'direction': 1, 'idle': -0.1, 'multiplier': 2, 'variable': 1, 'input_channel': 'none'},
    'not used2': {'output_channel': 10, 'type': 'none'},
    'not used3': {'output_channel': 11, 'type': 'none'},
    'not used4': {'output_channel': 12, 'type': 'none'},
    'not used5': {'output_channel': 13, 'type': 'none'},
    'not used6': {'output_channel': 14, 'type': 'none'},
    'not used7': {'output_channel': 15, 'type': 'none'},
}
