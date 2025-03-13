#!/usr/bin/env python3
"""
Terminal based 3D cube simulation using ASCII characters. Looks cool in SSH!
Demo for reading Adafruit ISM330DHCX IMU connected to tca9548a I2C multiplexer.
Idea from: https://github.com/msadeqsirjani/ascii-3d-cube
"""

import math
import time
import curses
import numpy as np
import sys
import io
from contextlib import contextmanager
from imu_reader import QuatIMUReader


@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to temporarily suppress stdout and stderr output.
    Used to hide debug prints from imported modules.
    """
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create null device to redirect output
    null_device = io.StringIO()

    try:
        # Redirect stdout and stderr to the null device
        sys.stdout = null_device
        sys.stderr = null_device
        yield
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class Cube:
    def __init__(self, scale=2.2):
        # Define cube vertices with scaling
        self.vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]
        # Apply scaling
        self.vertices = [[x * scale, y * scale, z * scale] for x, y, z in self.vertices]

        # Define faces as triangles for rendering
        self.faces = [
            # Front face
            (0, 1, 2), (0, 2, 3),
            # Back face
            (4, 6, 5), (4, 7, 6),
            # Right face
            (1, 5, 6), (1, 6, 2),
            # Left face
            (4, 0, 3), (4, 3, 7),
            # Top face
            (3, 2, 6), (3, 6, 7),
            # Bottom face
            (4, 5, 1), (4, 1, 0)
        ]

        # Define edges for wireframe rendering
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
        ]

    def rotate_vertex_quat(self, vertex, quat):
        """Rotate a vertex using a quaternion [w, x, y, z]"""
        w, x, y, z = quat
        vx, vy, vz = vertex

        # Optimized quaternion rotation for speed
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        # Apply rotation formula
        new_x = (1 - 2 * (yy + zz)) * vx + 2 * (xy - wz) * vy + 2 * (xz + wy) * vz
        new_y = 2 * (xy + wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz
        new_z = 2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz

        return [new_x, new_y, new_z]

    def get_face_normal(self, face, rotated_vertices):
        """Calculate the normal vector for a face"""
        v1, v2, v3 = [rotated_vertices[i] for i in face]

        # Calculate two edges
        edge1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
        edge2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]

        # Cross product to get normal
        normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ]

        # Normalize
        length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        if length > 0:
            normal = [n / length for n in normal]

        return normal


class FastTerminalRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Simplified character set for faster rendering
        self.chars = " .,:;i1tfLCG08@"

        # Pre-allocate and initialize buffers just once
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.z_buffer = [[float('inf') for _ in range(width)] for _ in range(height)]

        # Pre-compute and cache projection parameters
        self.z_offset = 5
        self.scale = 15
        self.half_width = width // 2
        self.half_height = height // 2

    def clear(self):
        """Fast buffer clearing"""
        for y in range(self.height):
            for x in range(self.width):
                self.buffer[y][x] = ' '
                self.z_buffer[y][x] = float('inf')

    def project(self, point):
        """Fast projection of 3D point to 2D screen coordinates"""
        x, y, z = point

        # Apply perspective projection (optimized)
        z_with_offset = z + self.z_offset
        perspective = self.scale / max(0.1, z_with_offset)
        screen_x = int(x * perspective + self.half_width)
        screen_y = int(y * perspective + self.half_height)

        # Fast bounds checking
        if screen_x < 0:
            screen_x = 0
        elif screen_x >= self.width:
            screen_x = self.width - 1

        if screen_y < 0:
            screen_y = 0
        elif screen_y >= self.height:
            screen_y = self.height - 1

        return screen_x, screen_y, z

    def draw_line(self, start, end, char='@'):
        """Optimized Bresenham line algorithm"""
        x1, y1, z1 = start
        x2, y2, z2 = end

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        # Calculate z increment
        line_length = max(1, dx + dy)
        dz = (z2 - z1) / line_length
        z = z1

        x, y = x1, y1
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Only draw if closer than what's already there
                if z < self.z_buffer[y][x]:
                    self.buffer[y][x] = char
                    self.z_buffer[y][x] = z

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            z += dz

    def draw_filled_triangle(self, v1, v2, v3, shade_char):
        """Optimized triangle filling algorithm"""
        # Project the points
        p1 = self.project(v1)
        p2 = self.project(v2)
        p3 = self.project(v3)

        # Get bounding box
        min_x = max(0, min(p1[0], p2[0], p3[0]))
        max_x = min(self.width - 1, max(p1[0], p2[0], p3[0]))
        min_y = max(0, min(p1[1], p2[1], p3[1]))
        max_y = min(self.height - 1, max(p1[1], p2[1], p3[1]))

        # Quick check if triangle is too small
        if max_x - min_x < 2 or max_y - min_y < 2:
            # Just draw the vertices to ensure small triangles are visible
            for p in [p1, p2, p3]:
                x, y, z = p
                if z < self.z_buffer[y][x]:
                    self.buffer[y][x] = shade_char
                    self.z_buffer[y][x] = z
            return

        # Pre-compute barycentric denominators
        x1, y1, _ = p1
        x2, y2, _ = p2
        x3, y3, _ = p3

        # For each point in bounding box, check if it's in the triangle
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Fast point-in-triangle test
                if self.is_point_in_triangle(x, y, p1, p2, p3):
                    # Calculate z (approximate)
                    z = (p1[2] + p2[2] + p3[2]) / 3

                    # Z-buffer test
                    if z < self.z_buffer[y][x]:
                        self.z_buffer[y][x] = z
                        self.buffer[y][x] = shade_char

    def is_point_in_triangle(self, x, y, p1, p2, p3):
        """Simplified point-in-triangle test for speed"""
        x1, y1, _ = p1
        x2, y2, _ = p2
        x3, y3, _ = p3

        # Edge functions - if all have same sign, point is inside
        e1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
        e2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
        e3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)

        # Check if all have same sign (point is inside)
        return (e1 >= 0 and e2 >= 0 and e3 >= 0) or (e1 <= 0 and e2 <= 0 and e3 <= 0)


def main():

    # Initialize IMU reader with minimal filtering for maximum responsiveness
    # Use the context manager to suppress all debug prints from the IMU module
    with suppress_stdout_stderr():
        imu_reader = QuatIMUReader(
            filter_alpha=0.05,  # low pass filtering
            beta=2.0,  # Madgwick responsiveness
            integration_method="madgwick",  # "madgwick", "exponential", "adaptive"
            dead_band=0.0,  # dead band
            spike_threshold=0.0,  # spike filtering disabled
            moving_avg_samples=25  # moving average filter
        )

    print(f"Done! Found {imu_reader.imu_count} IMUs.")

    if imu_reader.imu_count == 0:
        print("No IMUs detected! Please check your connections.")
        return

    # Initialize cube
    cube = Cube(scale=2.2)

    # Set up curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)  # Hide cursor
    stdscr.keypad(True)
    stdscr.timeout(1)  # Almost no blocking on input (1ms timeout)

    try:
        # Get terminal size
        max_y, max_x = stdscr.getmaxyx()
        width = min(50, max_x - 1)  # Limit to 50 columns or terminal width
        height = min(50, max_y - 2)  # Limit to 50 rows or terminal height - 2 for status

        # Initialize renderer
        renderer = FastTerminalRenderer(width, height)

        # Initial orientation
        orientation = np.array([0.9239, 0.2751, 0.1382, 0.2329], dtype=np.float32)
        orientation = orientation / np.sqrt(np.sum(orientation ** 2))  # Normalize

        # FPS calculation variables
        prev_time = time.monotonic()
        fps_counter = 0
        fps_timer = prev_time
        fps = 0
        real_update_rate = 0
        update_counter = 0
        update_timer = prev_time

        # Set target frame rate to 60fps
        frame_duration = 1.0 / 60.0  # 16.67ms between frames
        last_screen_update = time.monotonic()

        # Main loop
        running = True

        while running:
            # Get current time
            current_time = time.monotonic()

            # Process IMU data every iteration
            try:
                # Suppress any print statements that might come from IMU reader
                with suppress_stdout_stderr():
                    quats = imu_reader.read_quats(decimals=5)

                if quats is not None and len(quats) > 0:
                    orientation = quats[0]  # Use raw quaternion from IMU
                    update_counter += 1
            except Exception as e:
                # Only show error on screen updates
                if (current_time - last_screen_update) >= frame_duration:
                    stdscr.addstr(height + 1, 0, f"IMU error: {str(e)[:max_x - 20]}")

            # Check for keyboard input
            key = stdscr.getch()
            if key == ord('q') or key == 27:  # 'q' or ESC
                running = False
            elif key == ord('r'):  # Reset orientation
                orientation = np.array([0.9239, 0.2751, 0.1382, 0.2329], dtype=np.float32)
                orientation = orientation / np.sqrt(np.sum(orientation ** 2))

            # Only update screen if enough time has passed (60fps cap)
            time_since_last_update = current_time - last_screen_update
            if time_since_last_update >= frame_duration:
                # Calculate real IMU update rate
                if current_time - update_timer >= 1.0:
                    real_update_rate = update_counter
                    update_counter = 0
                    update_timer = current_time

                # Calculate FPS
                fps_counter += 1
                if current_time - fps_timer >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_timer = current_time

                # Clear screen
                renderer.clear()

                # Rotate vertices
                rotated_vertices = []
                for vertex in cube.vertices:
                    rotated = cube.rotate_vertex_quat(vertex, orientation)
                    rotated_vertices.append(rotated)

                # Draw faces with depth sorting
                faces_with_z = []
                for i, face in enumerate(cube.faces):
                    v1, v2, v3 = [rotated_vertices[idx] for idx in face]
                    avg_z = (v1[2] + v2[2] + v3[2]) / 3

                    # Get face normal for backface culling
                    normal = cube.get_face_normal(face, rotated_vertices)

                    # Only draw faces pointing toward the camera
                    if normal[2] >= 0:
                        continue

                    # Calculate lighting
                    light_val = min(1.0, max(0.0, -normal[2] * 0.8 + 0.2))
                    char_idx = min(len(renderer.chars) - 1, int(light_val * (len(renderer.chars) - 1)))
                    shade_char = renderer.chars[char_idx]

                    faces_with_z.append((face, avg_z, shade_char))

                # Sort faces by depth (furthest first)
                faces_with_z.sort(key=lambda x: x[1], reverse=True)

                # Draw all faces
                for face, _, shade_char in faces_with_z:
                    v1, v2, v3 = [rotated_vertices[idx] for idx in face]
                    renderer.draw_filled_triangle(v1, v2, v3, shade_char)

                # Draw edges for better definition
                for edge in cube.edges:
                    p1 = rotated_vertices[edge[0]]
                    p2 = rotated_vertices[edge[1]]
                    p1_proj = renderer.project(p1)
                    p2_proj = renderer.project(p2)
                    renderer.draw_line(p1_proj, p2_proj, '@')

                # Display the buffer
                for y in range(height):
                    line = ''.join(renderer.buffer[y])
                    stdscr.addstr(y, 0, line)

                # Calculate screen refresh rate
                screen_fps = 1.0 / max(0.001, time_since_last_update)

                # Update the last screen update time
                last_screen_update = current_time

                # Show status line with both update rates
                status = f"Screen: {min(screen_fps, 60.0):.1f} FPS | IMU: {real_update_rate} Hz | Q: [{orientation[0]:.2f}, {orientation[1]:.2f}, {orientation[2]:.2f}, {orientation[3]:.2f}] | R=Reset | ESC=Quit"
                stdscr.addstr(height + 1, 0, status[:max_x - 1])

                # Refresh screen
                stdscr.refresh()

            # Calculate how much time to wait to maintain 60fps
            elapsed = time.monotonic() - current_time
            if elapsed < frame_duration:
                # Small sleep to avoid hogging CPU when we're ahead of schedule
                # Using a very short sleep to maintain high IMU sampling rate
                time.sleep(0.0005)  # 0.5ms - still allows many IMU reads between frames

    except Exception as e:
        # Clean up curses on error
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up curses
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
    main()