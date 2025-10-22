# Base code from: https://github.com/kevinhughes27/TensorKart/blob/master/utils.py

import math
import threading
import time
from inputs import get_gamepad, UnpluggedError


"""
MAPPING (this is the index read() function returns):
0: 'LeftJoystickY'
1: 'LeftJoystickX'
2: 'RightJoystickY'
3: 'RightJoystickX'
4: 'LeftTrigger'
5: 'RightTrigger'
6: 'LeftBumper'
7: 'RightBumper'
8: 'A'
9: 'X'
10: 'Y'
11: 'B'
12: 'LeftThumb'
13: 'RightThumb'
14: 'Back'
15: 'Start'
16: 'LeftDPad'
17: 'RightDPad'
18: 'UpDPad'
19: 'DownDPad'
"""


class XboxController:
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)
    MAX_RECONNECT_ATTEMPTS = 5  # Maximum number of reconnection attempts

    def __init__(self):
        self.reset_values()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._connected = False
        self._reconnect_count = 0
        self.start_monitoring()

    def reset_values(self):
        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

    def start_monitoring(self):
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_controller)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

    def read(self):
        if not self._connected:
            #print("[Warning] Controller is not connected! (you might need to press any button to connect)")
            self.reset_values()
        return {
            'LeftJoystickY': self.LeftJoystickY,
            'LeftJoystickX': self.LeftJoystickX,
            'RightJoystickY': self.RightJoystickY,
            'RightJoystickX': self.RightJoystickX,
            'LeftTrigger': self.LeftTrigger,
            'RightTrigger': self.RightTrigger,
            'LeftBumper': self.LeftBumper,
            'RightBumper': self.RightBumper,
            'A': self.A,
            'X': self.X,
            'Y': self.Y,
            'B': self.B,
            'LeftThumb': self.LeftThumb,
            'RightThumb': self.RightThumb,
            'Back': self.Back,
            'Start': self.Start,
            'LeftDPad': self.LeftDPad,
            'RightDPad': self.RightDPad,
            'UpDPad': self.UpDPad,
            'DownDPad': self.DownDPad
        }

    def _monitor_controller(self):
        while not self._stop_event.is_set():
            try:
                events = get_gamepad()
                self._connected = True
                self._reconnect_count = 0  # Reset reconnection counter on successful connection
                for event in events:
                    self._process_event(event)
            except UnpluggedError:
                if self._connected:
                    print("[JOYSTICK] Controller disconnected. Attempting to reconnect...")
                    self._connected = False
                    self.reset_values()
                if not self._attempt_reconnect():
                    print("[ERROR] Maximum reconnection attempts reached. Stopping controller monitoring.")
                    self._stop_event.set()
                    break

    def _process_event(self, event):
        if event.code == 'ABS_Y':
            self.LeftJoystickY = event.state / self.MAX_JOY_VAL
        elif event.code == 'ABS_X':
            self.LeftJoystickX = event.state / self.MAX_JOY_VAL
        elif event.code == 'ABS_RY':
            self.RightJoystickY = event.state / self.MAX_JOY_VAL
        elif event.code == 'ABS_RX':
            self.RightJoystickX = event.state / self.MAX_JOY_VAL
        elif event.code == 'ABS_Z':
            self.LeftTrigger = event.state / self.MAX_TRIG_VAL
        elif event.code == 'ABS_RZ':
            self.RightTrigger = event.state / self.MAX_TRIG_VAL
        elif event.code == 'BTN_TL':
            self.LeftBumper = event.state
        elif event.code == 'BTN_TR':
            self.RightBumper = event.state
        elif event.code == 'BTN_SOUTH':
            self.A = event.state
        elif event.code == 'BTN_NORTH':
            self.Y = event.state
        elif event.code == 'BTN_WEST':
            self.X = event.state
        elif event.code == 'BTN_EAST':
            self.B = event.state
        elif event.code == 'BTN_THUMBL':
            self.LeftThumb = event.state
        elif event.code == 'BTN_THUMBR':
            self.RightThumb = event.state
        elif event.code == 'BTN_SELECT':
            self.Back = event.state
        elif event.code == 'BTN_START':
            self.Start = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY1':
            self.LeftDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY2':
            self.RightDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY3':
            self.UpDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY4':
            self.DownDPad = event.state

    def _attempt_reconnect(self):
        wait_delay = 3

        while not self._stop_event.is_set() and self._reconnect_count < self.MAX_RECONNECT_ATTEMPTS:
            try:
                get_gamepad()
                print("[JOYSTICK] Controller reconnected successfully!")
                self._connected = True
                return True
            except UnpluggedError:
                self._reconnect_count += 1
                remaining_attempts = self.MAX_RECONNECT_ATTEMPTS - self._reconnect_count
                print(f"[JOYSTICK] Reconnection attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS} "
                      f"failed. {remaining_attempts} attempts remaining. "
                      f"Retrying in {wait_delay} seconds...")
                time.sleep(wait_delay)

        return False


    def is_connected(self):
        return self._connected

    def __del__(self):
        self.stop_monitoring()

if __name__ == "__main__":
    # initialize the controller
    controller = XboxController()

    while True:
        # read values
        print(controller.read())
        # check connection status
        print(f"Connected: {controller.is_connected()}")

        time.sleep(0.1)
