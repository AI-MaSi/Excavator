# UPD / TCP hybrid connection demo
# handshake with TCP, data transmission with UDP

import universal_connection_manager
import masi_driver
from sensor_manager import PressureSensor
from time import time, sleep


addr = '192.168.0.136'
port = 5111

# Who am I. Check config for names
identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
# My (num) inputs received from the other end
inputs = 20
# My (num) outputs im going to send
outputs = 0



# init pwm driver
controller = masi_driver.ExcavatorController(inputs=inputs,
                                             simulation_mode=False,
                                             config_file='driver_config_Motionplatform.yaml',
                                             pump_variable=True,)

# init socket
masi_manager = universal_connection_manager.MasiSocketManager(identification_number, inputs, outputs)

# init pressure sensors
sensors = PressureSensor()


# set up Excavator as server.
if not masi_manager.setup_socket(addr, port, socket_type='server'):
    raise Exception("could not setup socket!")
    # setup done

# receive handshake and extra arguments (loop_frequency)
handshake_result, extra_args = masi_manager.handshake(test_arg=69)

# receive used loop_frequency from other device
loop_frequency = extra_args[0]

# Set Excavator safety threshold to be 50% of the loop freq
controller.set_threshold(loop_frequency/2)

if not handshake_result:
    raise Exception("could not make handshake!")

# switcheroo
masi_manager.tcp_to_udp()


def run():
    # start receiving
    masi_manager.start_data_recv_thread()

    last_print_time = time()  # Initialize the last print time

    while True:
        # get latest received joystick values
        value_list = masi_manager.get_latest_received()

        # Update the controller with new values if any
        controller.update_values(value_list)
        #if value_list is not None:
            #print(f"Received: {value_list}")

        # Print sensor values (placeholder function call, replace with actual usage)
        pressures = sensors.read_pressure(return_names=True)

        print(pressures)

        # Check if 3 seconds have elapsed since the last print
        # WIP, shows wrong hz!!
        current_time = time()
        if current_time - last_print_time >= 3:
            # Retrieve and print the current receiving frequency
            current_hz = controller.get_current_hz()
            print(f"Current Receiving Hz: {current_hz if current_hz is not None else 'N/A'}")
            last_print_time = current_time  # Reset the last print time


        #sleep(0.001)

if __name__ == "__main__":
    run()
