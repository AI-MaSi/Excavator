
import sensor_manager
from time import sleep


# init pressure sensors
pressures = sensor_manager.PressureSensor()

# init IMUs
imus = sensor_manager.IMUSensorManager(simulation_mode=False)

# init RPM check
rpm = sensor_manager.RPMSensor()


while True:

    print(f"IMUs: {imus.read_all()}")
    print(f"PRESSUREs: {pressures.read_pressure()}")
    print(f"RPM: {rpm.read_rpm()}")


    sleep(1)



