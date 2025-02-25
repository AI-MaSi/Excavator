import board
import adafruit_tca9548a
import time
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX


def scan_multiplexer():
    i2c = board.I2C()

    # Try to find multiplexer at address 0x71
    try:
        mux = adafruit_tca9548a.TCA9548A(i2c, address=0x71)
        print(f"Found multiplexer at address 0x71")
    except Exception as e:
        print(f"Could not find multiplexer: {e}")
        return

    print("\nScanning each multiplexer channel for IMU...")

    # Try each channel
    for channel in range(8):
        try:
            # Select channel
            print(f"\nTesting channel {channel}:")
            channel_bus = mux[channel]

            # Try to initialize IMU
            try:
                imu = ISM330DHCX(channel_bus)
                # Try reading to verify it works
                accel = imu.acceleration
                print(f"✓ Found working IMU")
                print(f"  Test reading: accel_x={accel[0]:.2f}")
            except:
                print("  No IMU found")

            time.sleep(0.1)  # Small delay between channels

        except Exception as e:
            print(f"  Error accessing channel: {e}")


if __name__ == "__main__":
    scan_multiplexer()