# UPD / TCP hybrid connection demo
# handshake with TCP, data transmission with UDP

import universal_connection_manager
import masi_driver
from sensor_manager import PressureSensor
from time import sleep


addr = '127.0.0.1'
port = 5111

# Who am I. Check config for names
identification_number = 0  # 0 excavator, 1 Mevea, 2 Motion Platform, more can be added...
# My (num) inputs received from the other end
inputs = 20
# My (num) outputs im going to send
outputs = 0



# init pwm driver
controller = masi_driver.ExcavatorController(inputs=inputs,
                                             simulation_mode= True,
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


    while True:
        # get latest received joystick values
        value_list = masi_manager.get_latest_received()

        # update PWM driver values
        controller.update_values(value_list)
        if value_list is not None:
            print(f"Received: {value_list}")


        pressures = sensors.read_pressure()

        sleep(0.001)

if __name__ == "__main__":
    run()
