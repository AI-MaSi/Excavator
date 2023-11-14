import RPi.GPIO as GPIO
import time

# Set the GPIO mode to BCM numbering
GPIO.setmode(GPIO.BCM)

# sensor under the motor
INPUT_PIN = 4
# sensor under the center rotate
INPUT2_PIN = 17
GPIO.setup(INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(INPUT2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Number of magnets on the motor
MAGNET_COUNT = 14

# Time window to calculate RPM
SAMPLE_TIME = 0.2  # In seconds

try:
    last_state = GPIO.input(INPUT_PIN)
    toggle_count = 0
    start_time = time.time()

    while True:
        current_state = GPIO.input(INPUT_PIN)

        # Count the toggles from LOW to HIGH
        if current_state == GPIO.HIGH and last_state == GPIO.LOW:
            toggle_count += 1

        last_state = current_state

        # Calculate the RPM if SAMPLE_TIME has passed
        if time.time() - start_time >= SAMPLE_TIME:
            rpm = (toggle_count / MAGNET_COUNT) * (60 / SAMPLE_TIME)
            print(f"RPM: {rpm:.1f}")
            print(f"Center rotate sensor: {GPIO.input(INPUT2_PIN)}")
            toggle_count = 0  # Reset the toggle count
            start_time = time.time()  # Reset the start time

        time.sleep(0.001)  # Short delay to prevent excessive CPU usage

except KeyboardInterrupt:
    # Cleanup the GPIO pins when the script is interrupted
    GPIO.cleanup()
    print("\nGPIO cleaned up and script terminated.")
