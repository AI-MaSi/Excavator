# old pressure reading code

# check board settings some day!

from ADCPi import ADCPi

channel_numbers = 6
decimals = 1
data = [0] * channel_numbers

try:
    adc = ADCPi(0x6E, 0x6F, 12)
except OSError as e:
    print(f"Failed to set I2C-address! Error: {e}")
    initialized = False

adc.set_conversion_mode(1)
initialized = True

def read():
    if initialized:
        for i in range(1, channel_numbers + 1):
            if adc.read_voltage(i) > 0.40:
                # round values and covert the voltages to psi (rough)
                data[i - 1] = round(((1000 * (adc.read_voltage(i) - 0.5) / (4.5 - 0.5))), decimals)
            else:
                data[i - 1] = 0
    else:
        print("ADCPi not initialized!")

    return data

