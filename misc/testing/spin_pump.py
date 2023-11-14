from time import sleep
from adafruit_servokit import ServoKit


kit = ServoKit(channels=16)

kit.continuous_servo[9].throttle = -1

sleep(3)

kit.continuous_servo[9].throttle = -0.05