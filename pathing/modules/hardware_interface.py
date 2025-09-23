#!/usr/bin/env python3
"""
Hardware Interface

...

Usage:
    ...
"""



import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import threading

from .PCA9685_controller import PWMController
from .usb_serial_reader import USBSerialReader


class HardwareInterfaceBase(ABC):
    """Abstract base class for hardware interfaces."""
    
    @abstractmethod
    def read_imu_data(self) -> Optional[List[np.ndarray]]:
        """
        Read current IMU quaternion data.
        
        Returns:
            List of quaternions [q1, q2, q3] as numpy arrays, or None if unavailable
        """
        pass
    
    @abstractmethod  
    def send_pwm_commands(self, commands: List[float]) -> bool:
        """
        Send PWM commands to actuators.
        
        Args:
            commands: List of 8 PWM values [-1.0 to 1.0]
            
        Returns:
            True if commands sent successfully
        """
        pass

    @abstractmethod
    def is_hardware_ready(self) -> bool:
        """Check if hardware is ready for operation."""
        pass
    
    def reset(self, reset_pump: bool = True) -> None:
        """Reset hardware to safe state (default implementation)."""
        pass

class HardwareInterface(HardwareInterfaceBase):
    """Real hardware interface using existing PWM and IMU modules."""
    
    def __init__(self, 
                 config_file: str = "configuration_files/linear_config.yaml",
                 pump_variable: bool = False,
                 toggle_channels: bool = False,
                 input_rate_threshold: int = 10):
        """
        Initialize real hardware interface.
        
        Args:
            config_file: Path to PWM controller configuration
            pump_variable: Whether to use variable pump control
            toggle_channels: Whether to toggle PWM channels
            input_rate_threshold: Input rate threshold for PWM controller
        """
        self.config_file = config_file
        
        # Initialize PWM controller
        if PWMController is not None:
            try:
                self.pwm_controller = PWMController(
                    config_file=config_file,
                    pump_variable=pump_variable,
                    toggle_channels=toggle_channels,
                    input_rate_threshold=input_rate_threshold
                )
                self.pwm_ready = True
                print("PWM controller initialized")
            except Exception as e:
                print(f"PWM controller initialization failed: {e}")
                self.pwm_controller = None
                self.pwm_ready = False
        else:
            print("PWM controller not available")
            self.pwm_controller = None
            self.pwm_ready = False
        
        # Initialize IMU reader
        self.imu_ready = False
        self.latest_imu_data = None
        self._imu_lock = threading.Lock()
        self._start_imu_reader()
    
    def _check_imu_streaming(self, timeout: float = 2.0) -> bool:
        """Check if IMUs are already streaming valid data."""
        if self.usb_reader is None:
            return False
            
        start_time = time.time()
        valid_count = 0
        
        while (time.time() - start_time) < timeout:
            imu_data = self.usb_reader.read_imus()
            if imu_data is not None and len(imu_data) >= 3:
                valid_count += 1
                if valid_count >= 3:  # Got multiple valid readings
                    return True
            time.sleep(0.1)
        
        return False
    
    def _start_imu_reader(self) -> None:
        """Initialize IMU reader and start background thread."""
        if USBSerialReader is not None:
            try:
                self.usb_reader = USBSerialReader()
                
                # Check if IMUs are already streaming data
                print("Checking if IMUs are already streaming...")
                already_streaming = self._check_imu_streaming()
                
                if not already_streaming:
                    # Send handshake configuration to start IMU data streaming
                    print("IMUs not streaming - sending handshake configuration...")
                    if self.usb_reader.send_handshake_config():
                        print("IMU handshake sent successfully")
                    else:
                        print("IMU handshake failed")
                else:
                    print("IMUs already streaming - skipping handshake")
                
                # Start background thread for IMU reading
                self.imu_thread = threading.Thread(
                    target=self._imu_reader_thread, 
                    daemon=True
                )
                self.imu_thread.start()
                print("IMU reader thread started")
                
            except Exception as e:
                print(f"IMU initialization failed: {e}")
                self.usb_reader = None
        else:
            print("IMU reader not available")
            self.usb_reader = None
            
    def _imu_reader_thread(self) -> None:
        """Background thread for continuous IMU reading."""
        while True:
            try:
                quaternions = self.usb_reader.read_imus()
                if quaternions is not None and len(quaternions) >= 3:
                    # Convert to numpy arrays
                    quaternion_arrays = [np.array(q, dtype=np.float32) for q in quaternions]
                    
                    # Atomically update latest data
                    with self._imu_lock:
                        self.latest_imu_data = quaternion_arrays[:3]  # Take first 3
                        if not self.imu_ready:
                            self.imu_ready = True
                            print("IMU data acquisition started")
                        
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                print(f"IMU reader thread error: {e}")
                with self._imu_lock:
                    self.imu_ready = False
                time.sleep(0.1)  # Longer sleep on error
    
    def read_imu_data(self) -> Optional[List[np.ndarray]]:
        """Read latest IMU data."""
        # Get latest data atomically - check ready flag inside lock
        try:
            with self._imu_lock:
                if not self.imu_ready:
                    return None
                    
                if self.latest_imu_data is not None:
                    # Return copy to avoid race conditions
                    return [q.copy() for q in self.latest_imu_data]
                else:
                    return None
                    
        except Exception as e:
            print(f"IMU read error: {e}")
            return None
    
    def send_pwm_commands(self, commands: List[float]) -> bool:
        """Send PWM commands to actuators."""
        if not self.pwm_ready or self.pwm_controller is None:
            return False
            
        try:
            # Ensure we have 8 commands
            if len(commands) < 8:
                commands = list(commands) + [0.0] * (8 - len(commands))

            #print(f"DEBUG: Sending PWM commands: {commands}")
            self.pwm_controller.update_values(commands[:8])
            return True
            
        except Exception as e:
            print(f"PWM command error: {e}")
            return False

    def reset(self, reset_pump: bool = False) -> None:
        """Reset hardware to safe state."""
        if self.pwm_controller is not None:
            try:
                self.pwm_controller.reset(reset_pump=reset_pump)
            except Exception as e:
                print(f"Hardware reset error: {e}")
    
    def is_hardware_ready(self) -> bool:
        """Check if hardware is ready."""
        return self.pwm_ready and self.imu_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get hardware status for monitoring."""
        return {
            'pwm_ready': self.pwm_ready,
            'imu_ready': self.imu_ready,
            'latest_imu_timestamp': time.time() if self.latest_imu_data is not None else None
        }
