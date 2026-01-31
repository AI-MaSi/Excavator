#!/usr/bin/env python3
"""
Test client that sends sine waves to the server for hardware validation.

Run this on the client machine. The server (simple_drive_logging.py) should
be running on the Pi at 192.168.0.132:8080.

Controller data format (9 floats, all in [-1, 1]):
  [0] button_0  - Start/Stop logging toggle
  [1] button_1  - IMU display toggle
  [2] button_2  - Pump toggle
  [3] left_paddle  - left track
  [4] right_paddle - right track
  [5] left_ud   - tilt_boom
  [6] left_rl   - rotate
  [7] right_ud  - lift_boom
  [8] right_rl  - scoop
"""

import sys
import os

# Add parent directory to path if running from a different location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.udp_socket import UDPSocket
import time
import math

# ============================================================
# CONFIGURATION
# ============================================================
# SERVER_IP = "192.168.0.132"  # Pi's IP address (production network)
SERVER_IP = "192.168.0.105"  # Pi's IP address (test network)
SERVER_PORT = 8080
SEND_RATE_HZ = 100  # Match server's expected rate

# Sine wave parameters
SINE_FREQUENCY_HZ = 0.2  # Slow oscillation (5 second period)
SINE_AMPLITUDE = 0.8     # Max command value [-1, 1]

# Which channels to send sine waves to (set True to enable)
CHANNELS = {
    'scoop': True,       # [8] right_rl
    'lift_boom': True,   # [7] right_ud
    'tilt_boom': True,   # [5] left_ud
    'rotate': False,     # [6] left_rl - careful with rotation!
    'trackL': False,     # [3] left_paddle
    'trackR': False,     # [4] right_paddle
}

# Phase offsets (radians) - stagger the sine waves
PHASE_OFFSETS = {
    'scoop': 0.0,
    'lift_boom': math.pi / 3,      # 60 degrees offset
    'tilt_boom': 2 * math.pi / 3,  # 120 degrees offset
    'rotate': math.pi,
    'trackL': 0.0,
    'trackR': math.pi,
}


def generate_sine(t, freq, amplitude, phase_offset=0.0):
    """Generate sine wave value at time t."""
    return amplitude * math.sin(2 * math.pi * freq * t + phase_offset)


def main():
    print("=" * 60)
    print("  SINE WAVE TEST CLIENT")
    print("=" * 60)
    print(f"Server: {SERVER_IP}:{SERVER_PORT}")
    print(f"Send rate: {SEND_RATE_HZ} Hz")
    print(f"Sine frequency: {SINE_FREQUENCY_HZ} Hz (period: {1/SINE_FREQUENCY_HZ:.1f}s)")
    print(f"Amplitude: {SINE_AMPLITUDE}")
    print()
    print("Active channels:")
    for ch, enabled in CHANNELS.items():
        if enabled:
            print(f"  - {ch}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Initialize UDP client
    client = UDPSocket(local_id=1)
    client.setup(SERVER_IP, SERVER_PORT, num_inputs=0, num_outputs=9, is_server=False)

    # Perform handshake
    print("\nAttempting handshake with server...")
    if not client.handshake(timeout=30.0):
        print("Handshake failed! Is the server running?")
        return

    print("Handshake successful!")
    print("\nSending sine waves... (Ctrl+C to stop)\n")

    # Start receiving (for bidirectional communication if needed)
    client.start_receiving()

    start_time = time.time()
    loop_period = 1.0 / SEND_RATE_HZ
    next_send_time = time.perf_counter()

    try:
        while True:
            t = time.time() - start_time

            # Build command array (9 floats)
            # Format: [btn0, btn1, btn2, left_paddle, right_paddle, left_ud, left_rl, right_ud, right_rl]
            commands = [0.0] * 9

            # Generate sine values for enabled channels
            if CHANNELS['scoop']:
                commands[8] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['scoop'])

            if CHANNELS['lift_boom']:
                commands[7] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['lift_boom'])

            if CHANNELS['tilt_boom']:
                commands[5] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['tilt_boom'])

            if CHANNELS['rotate']:
                commands[6] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['rotate'])

            if CHANNELS['trackL']:
                commands[3] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['trackL'])

            if CHANNELS['trackR']:
                commands[4] = generate_sine(t, SINE_FREQUENCY_HZ, SINE_AMPLITUDE, PHASE_OFFSETS['trackR'])

            # Send to server
            client.send_floats(commands)

            # Status print every second
            if int(t) != int(t - loop_period) and int(t) % 1 == 0:
                active_vals = []
                if CHANNELS['scoop']:
                    active_vals.append(f"scoop={commands[8]:+.2f}")
                if CHANNELS['lift_boom']:
                    active_vals.append(f"lift={commands[7]:+.2f}")
                if CHANNELS['tilt_boom']:
                    active_vals.append(f"tilt={commands[5]:+.2f}")
                if CHANNELS['rotate']:
                    active_vals.append(f"rot={commands[6]:+.2f}")
                print(f"[t={t:6.1f}s] {' | '.join(active_vals)}")

            # Maintain send rate
            next_send_time += loop_period
            sleep_time = next_send_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_send_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Send zero commands before exiting
    print("Sending zero commands...")
    for _ in range(10):
        client.send_floats([0.0] * 9)
        time.sleep(0.01)

    print("Done!")


if __name__ == "__main__":
    main()
