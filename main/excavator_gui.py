#!/usr/bin/env python3
"""
Excavator Position GUI - Interactive XZ Position Selection with UDP Control

Click and drag the dot to select target positions for the excavator.
The workspace limits are hardcoded based on excavator reach.
Sends position commands via UDP at 20Hz when streaming is enabled.

Usage:
    python excavator_gui.py
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import threading
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from udp_socket import UDPSocket


class ExcavatorGUI:
    """Interactive GUI for selecting excavator end-effector positions"""

    # Workspace limits in meters (adjust these based on your excavator's reach)
    WORKSPACE_LIMITS = {
        'x_min': 0.1,   # Minimum reach (meters)
        'x_max': 0.7,   # Maximum reach (meters)
        'z_min': -0.3,  # Lowest position (meters)
        'z_max': 0.4,   # Highest position (meters)
    }

    # Y is typically fixed for 2D operation
    Y_POSITION = 0.0

    # UDP Configuration
    UDP_HOST = "192.168.0.132"
    UDP_PORT = 8080
    UDP_LOCAL_ID = 3  # Different from server (which uses 2)
    UDP_SEND_RATE_HZ = 20  # Send at 20Hz
    UDP_NUM_BYTES_SEND = 8  # 2 bytes per value (x, y, z, rot_y) = 8 bytes total
    UDP_NUM_BYTES_RECV = 24  # 24 bytes for 4 joint positions (6 bytes each)

    # Default starting position
    DEFAULT_X = 0.6
    DEFAULT_Y = 0.0
    DEFAULT_Z = 0.0

    # Canvas settings
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 500
    CANVAS_MARGIN = 50

    # Visual settings
    DOT_RADIUS = 10
    GRID_SPACING_M = 0.1  # Grid spacing in meters

    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Excavator Position Control GUI - UDP Streaming")

        # Current position (start at default position)
        self.current_x = self.DEFAULT_X
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0  # Y rotation in degrees

        # Dragging state
        self.dragging = False

        # View state
        self.flip_x = False  # Flip X-axis to view from other side

        # UDP state
        self.udp_client = None
        self.udp_connected = False
        self.udp_streaming = False
        self.send_thread = None
        self.send_thread_running = False
        self.packets_sent = 0
        self.last_send_time = 0.0

        # Received joint positions for visualization
        self.joint_positions = None  # Will be [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
        self.last_joint_update_time = 0.0

        # Setup GUI
        self._create_widgets()
        self._setup_canvas()
        self._bind_events()

        # Initial draw
        self._draw_workspace()
        self._update_display()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="Excavator Position Control",
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Canvas for workspace
        self.canvas = tk.Canvas(main_frame,
                               width=self.CANVAS_WIDTH,
                               height=self.CANVAS_HEIGHT,
                               bg='white',
                               relief=tk.SUNKEN,
                               borderwidth=2)
        self.canvas.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Info frame (now spans both columns)
        info_frame = ttk.LabelFrame(main_frame, text="Current Position", padding="10")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Position display in a row
        pos_display = ttk.Frame(info_frame)
        pos_display.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.x_label = ttk.Label(pos_display, text="X: 0.000 m", font=('Courier', 12))
        self.x_label.pack(side=tk.LEFT, padx=(0, 15))

        self.y_label = ttk.Label(pos_display, text=f"Y: {self.Y_POSITION:.3f} m",
                                font=('Courier', 12))
        self.y_label.pack(side=tk.LEFT, padx=(0, 15))

        self.z_label = ttk.Label(pos_display, text="Z: 0.000 m", font=('Courier', 12))
        self.z_label.pack(side=tk.LEFT, padx=(0, 15))

        self.rot_label = ttk.Label(pos_display, text="Rot Y:   0.0°", font=('Courier', 12), width=14)
        self.rot_label.pack(side=tk.LEFT)

        # Rotation slider (wider now)
        rot_slider_frame = ttk.Frame(info_frame)
        rot_slider_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(rot_slider_frame, text="Y Rotation:", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W)

        self.rot_slider = ttk.Scale(rot_slider_frame, from_=-45, to=45, orient=tk.HORIZONTAL,
                                    command=self._on_rotation_change)
        self.rot_slider.set(0.0)
        self.rot_slider.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(2, 0))

        # Configure column weights for proper stretching
        rot_slider_frame.columnconfigure(0, weight=1)

        # Slider value labels
        slider_labels = ttk.Frame(rot_slider_frame)
        slider_labels.grid(row=2, column=0, sticky=(tk.W, tk.E))
        ttk.Label(slider_labels, text="-45°", font=('Arial', 8)).pack(side=tk.LEFT)
        ttk.Label(slider_labels, text="0°", font=('Arial', 8)).pack(side=tk.LEFT, expand=True)
        ttk.Label(slider_labels, text="+45°", font=('Arial', 8)).pack(side=tk.RIGHT)

        # Center button (below slider)
        center_btn_frame = ttk.Frame(rot_slider_frame)
        center_btn_frame.grid(row=3, column=0, pady=(5, 0))
        self.center_rot_btn = ttk.Button(center_btn_frame, text="Center",
                                         command=self._center_rotation)
        self.center_rot_btn.pack()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Copy button
        self.copy_btn = ttk.Button(button_frame, text="Copy Position",
                                   command=self._copy_command)
        self.copy_btn.grid(row=0, column=0, padx=5)

        # Reset button
        self.reset_btn = ttk.Button(button_frame, text="Reset (0.6, 0.0, 0°)",
                                    command=self._reset_position)
        self.reset_btn.grid(row=0, column=1, padx=5)

        # Flip button
        self.flip_btn = ttk.Button(button_frame, text="Flip View ⇄",
                                   command=self._toggle_flip)
        self.flip_btn.grid(row=0, column=2, padx=5)

        # UDP Control Frame
        udp_frame = ttk.LabelFrame(main_frame, text="UDP Streaming Control", padding="10")
        udp_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Connection info
        conn_info_frame = ttk.Frame(udp_frame)
        conn_info_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        ttk.Label(conn_info_frame, text=f"Target: {self.UDP_HOST}:{self.UDP_PORT}",
                 font=('Courier', 9)).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(conn_info_frame, text=f"Rate: {self.UDP_SEND_RATE_HZ} Hz",
                 font=('Courier', 9)).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))

        # Connection status
        status_frame = ttk.Frame(udp_frame)
        status_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10))

        self.conn_status_label = ttk.Label(status_frame, text="● Disconnected",
                                          font=('Arial', 10), foreground='red')
        self.conn_status_label.grid(row=0, column=0, sticky=tk.W)

        self.packets_label = ttk.Label(status_frame, text="Packets: 0",
                                      font=('Courier', 9))
        self.packets_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))

        # Control buttons
        self.connect_btn = ttk.Button(udp_frame, text="Connect",
                                     command=self._connect_udp)
        self.connect_btn.grid(row=2, column=0, padx=5, pady=5)

        self.stream_btn = ttk.Button(udp_frame, text="Start Streaming",
                                     command=self._toggle_streaming,
                                     state=tk.DISABLED)
        self.stream_btn.grid(row=2, column=1, padx=5, pady=5)

        self.disconnect_btn = ttk.Button(udp_frame, text="Disconnect",
                                        command=self._disconnect_udp,
                                        state=tk.DISABLED)
        self.disconnect_btn.grid(row=2, column=2, padx=5, pady=5)

        # Instructions
        instructions = ttk.Label(main_frame,
                                text="Click and drag the red dot to set target position",
                                font=('Arial', 10, 'italic'))
        instructions.grid(row=5, column=0, columnspan=2, pady=(10, 0))

    def _setup_canvas(self):
        """Setup canvas coordinate system"""
        # Calculate workspace dimensions
        workspace_width = self.WORKSPACE_LIMITS['x_max'] - self.WORKSPACE_LIMITS['x_min']
        workspace_height = self.WORKSPACE_LIMITS['z_max'] - self.WORKSPACE_LIMITS['z_min']

        # Calculate scaling to fit canvas with margins
        usable_width = self.CANVAS_WIDTH - 2 * self.CANVAS_MARGIN
        usable_height = self.CANVAS_HEIGHT - 2 * self.CANVAS_MARGIN

        self.scale_x = usable_width / workspace_width
        self.scale_z = usable_height / workspace_height

        # Use uniform scaling (same scale for both axes)
        self.scale = min(self.scale_x, self.scale_z)

        # Calculate actual drawable area
        self.draw_width = workspace_width * self.scale
        self.draw_height = workspace_height * self.scale

        # Calculate offsets to center the workspace
        self.offset_x = (self.CANVAS_WIDTH - self.draw_width) / 2
        self.offset_z = (self.CANVAS_HEIGHT - self.draw_height) / 2

    def _world_to_canvas(self, x, z):
        """Convert world coordinates (meters) to canvas coordinates (pixels)"""
        # X increases to the right, Z increases upward
        # If flip_x is True, mirror the X axis
        if self.flip_x:
            canvas_x = self.offset_x + (self.WORKSPACE_LIMITS['x_max'] - x) * self.scale
        else:
            canvas_x = self.offset_x + (x - self.WORKSPACE_LIMITS['x_min']) * self.scale
        canvas_z = self.offset_z + (self.WORKSPACE_LIMITS['z_max'] - z) * self.scale
        return canvas_x, canvas_z

    def _canvas_to_world(self, canvas_x, canvas_z):
        """Convert canvas coordinates (pixels) to world coordinates (meters)"""
        # If flip_x is True, mirror the X axis
        if self.flip_x:
            x = self.WORKSPACE_LIMITS['x_max'] - (canvas_x - self.offset_x) / self.scale
        else:
            x = self.WORKSPACE_LIMITS['x_min'] + (canvas_x - self.offset_x) / self.scale
        z = self.WORKSPACE_LIMITS['z_max'] - (canvas_z - self.offset_z) / self.scale
        return x, z

    def _clamp_position(self, x, z):
        """Clamp position to workspace limits"""
        x = max(self.WORKSPACE_LIMITS['x_min'], min(x, self.WORKSPACE_LIMITS['x_max']))
        z = max(self.WORKSPACE_LIMITS['z_min'], min(z, self.WORKSPACE_LIMITS['z_max']))
        return x, z

    def _draw_workspace(self):
        """Draw workspace boundaries and grid"""
        self.canvas.delete('all')

        # Draw workspace boundary
        corners = [
            self._world_to_canvas(self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['z_max']),
            self._world_to_canvas(self.WORKSPACE_LIMITS['x_max'], self.WORKSPACE_LIMITS['z_max']),
            self._world_to_canvas(self.WORKSPACE_LIMITS['x_max'], self.WORKSPACE_LIMITS['z_min']),
            self._world_to_canvas(self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['z_min'])
        ]

        self.canvas.create_rectangle(
            corners[0][0], corners[0][1], corners[2][0], corners[2][1],
            outline='black', width=2, fill='#f0f0f0'
        )

        # Draw grid lines
        # Vertical lines (constant X)
        x = self.WORKSPACE_LIMITS['x_min']
        while x <= self.WORKSPACE_LIMITS['x_max']:
            x1, z1 = self._world_to_canvas(x, self.WORKSPACE_LIMITS['z_min'])
            x2, z2 = self._world_to_canvas(x, self.WORKSPACE_LIMITS['z_max'])

            # Thicker line for x=0
            width = 2 if abs(x) < 0.001 else 1
            color = 'blue' if abs(x) < 0.001 else '#cccccc'

            self.canvas.create_line(x1, z1, x2, z2, fill=color, width=width)

            # Label
            label_x, label_z = self._world_to_canvas(x, self.WORKSPACE_LIMITS['z_min'])
            self.canvas.create_text(label_x, label_z + 15, text=f"{x:.1f}",
                                   font=('Arial', 8))

            x += self.GRID_SPACING_M

        # Horizontal lines (constant Z)
        z = self.WORKSPACE_LIMITS['z_min']
        while z <= self.WORKSPACE_LIMITS['z_max']:
            x1, z1 = self._world_to_canvas(self.WORKSPACE_LIMITS['x_min'], z)
            x2, z2 = self._world_to_canvas(self.WORKSPACE_LIMITS['x_max'], z)

            # Thicker line for z=0
            width = 2 if abs(z) < 0.001 else 1
            color = 'green' if abs(z) < 0.001 else '#cccccc'

            self.canvas.create_line(x1, z1, x2, z2, fill=color, width=width)

            # Label
            label_x, label_z = self._world_to_canvas(self.WORKSPACE_LIMITS['x_min'], z)
            self.canvas.create_text(label_x - 20, label_z, text=f"{z:.1f}",
                                   font=('Arial', 8))

            z += self.GRID_SPACING_M

        # Draw axis labels
        mid_x = (self.WORKSPACE_LIMITS['x_min'] + self.WORKSPACE_LIMITS['x_max']) / 2
        mid_z = (self.WORKSPACE_LIMITS['z_min'] + self.WORKSPACE_LIMITS['z_max']) / 2

        label_x, label_z = self._world_to_canvas(mid_x, self.WORKSPACE_LIMITS['z_min'])
        x_arrow = "←" if self.flip_x else "→"
        self.canvas.create_text(label_x, label_z + 35, text=f"X (meters) {x_arrow}",
                               font=('Arial', 10, 'bold'))

        if self.flip_x:
            label_x, label_z = self._world_to_canvas(self.WORKSPACE_LIMITS['x_max'], mid_z)
        else:
            label_x, label_z = self._world_to_canvas(self.WORKSPACE_LIMITS['x_min'], mid_z)
        x_offset = 50 if self.flip_x else -50
        self.canvas.create_text(label_x + x_offset, label_z, text="↑\nZ\n(meters)",
                               font=('Arial', 10, 'bold'))

        # Draw view indicator
        view_text = "View: Flipped (Back)" if self.flip_x else "View: Normal (Front)"
        self.canvas.create_text(self.CANVAS_WIDTH / 2, 15, text=view_text,
                               font=('Arial', 11, 'bold'), fill='#003366')

        # Draw the position dot
        self._draw_dot()

    def _draw_dot(self):
        """Draw the position indicator dot"""
        canvas_x, canvas_z = self._world_to_canvas(self.current_x, self.current_z)

        # Delete old dot
        self.canvas.delete('dot')

        # Draw new dot
        self.canvas.create_oval(
            canvas_x - self.DOT_RADIUS, canvas_z - self.DOT_RADIUS,
            canvas_x + self.DOT_RADIUS, canvas_z + self.DOT_RADIUS,
            fill='red', outline='darkred', width=2, tags='dot'
        )

        # Draw crosshair
        self.canvas.create_line(
            canvas_x - self.DOT_RADIUS - 5, canvas_z,
            canvas_x + self.DOT_RADIUS + 5, canvas_z,
            fill='darkred', width=1, tags='dot'
        )
        self.canvas.create_line(
            canvas_x, canvas_z - self.DOT_RADIUS - 5,
            canvas_x, canvas_z + self.DOT_RADIUS + 5,
            fill='darkred', width=1, tags='dot'
        )

    def _draw_robot_arm(self):
        """Draw the robot arm visualization from joint positions"""
        # Delete old arm visualization
        self.canvas.delete('robot_arm')

        if self.joint_positions is None or len(self.joint_positions) != 4:
            return

        # Joint positions: [boom_mount, arm_mount, bucket_mount, tool_mount]
        # Add base at origin (0, 0, 0) for complete visualization
        positions_3d = [[0.0, 0.0, 0.0]] + self.joint_positions

        # Compute end-effector position with offset (hardcoded from robot config)
        # ee_offset = [0.0, 0.0, -0.142] meters (142mm below bucket center)
        # This offset is in the last joint's local frame, so we need to rotate it
        EE_OFFSET_LOCAL = [0.0, 0.0, -0.142]

        bucket_mount_pos = self.joint_positions[2]  # Bucket joint position
        tool_mount_pos = self.joint_positions[3]    # Tool mount (end of bucket link)

        # Compute bucket link direction (local X-axis)
        link_vec = [
            tool_mount_pos[0] - bucket_mount_pos[0],
            tool_mount_pos[1] - bucket_mount_pos[1],
            tool_mount_pos[2] - bucket_mount_pos[2]
        ]
        link_length = math.sqrt(link_vec[0]**2 + link_vec[1]**2 + link_vec[2]**2)

        if link_length > 0.001:  # Avoid division by zero
            # Normalize to get local X-axis
            local_x = [link_vec[0]/link_length, link_vec[1]/link_length, link_vec[2]/link_length]

            # Local Y-axis is always [0, 1, 0] (rotation axis, pointing up)
            local_y = [0.0, 1.0, 0.0]

            # Local Z-axis = cross(X, Y) to complete right-handed frame
            local_z = [
                local_x[1]*local_y[2] - local_x[2]*local_y[1],
                local_x[2]*local_y[0] - local_x[0]*local_y[2],
                local_x[0]*local_y[1] - local_x[1]*local_y[0]
            ]

            # Transform ee_offset from local to world coordinates
            ee_offset_world = [
                EE_OFFSET_LOCAL[0]*local_x[0] + EE_OFFSET_LOCAL[1]*local_y[0] + EE_OFFSET_LOCAL[2]*local_z[0],
                EE_OFFSET_LOCAL[0]*local_x[1] + EE_OFFSET_LOCAL[1]*local_y[1] + EE_OFFSET_LOCAL[2]*local_z[1],
                EE_OFFSET_LOCAL[0]*local_x[2] + EE_OFFSET_LOCAL[1]*local_y[2] + EE_OFFSET_LOCAL[2]*local_z[2]
            ]

            # Apply rotated offset to tool mount position
            ee_position = [
                tool_mount_pos[0] + ee_offset_world[0],
                tool_mount_pos[1] + ee_offset_world[1],
                tool_mount_pos[2] + ee_offset_world[2]
            ]
        else:
            # Fallback if link has zero length
            ee_position = tool_mount_pos

        # Convert 3D positions to 2D canvas coordinates (project Y away, use X and Z)
        canvas_points = []
        for pos in positions_3d:
            x, y, z = pos
            # Project onto XZ plane (ignore Y for 2D view)
            canvas_x, canvas_z = self._world_to_canvas(x, z)
            canvas_points.append((canvas_x, canvas_z))

        # Draw links as thick lines
        link_colors = ['#FF6600', '#FF9933', '#FFCC66', '#FFE699']  # Orange gradient
        for i in range(len(canvas_points) - 1):
            x1, z1 = canvas_points[i]
            x2, z2 = canvas_points[i + 1]

            # Draw link shadow for depth effect
            self.canvas.create_line(
                x1 + 2, z1 + 2, x2 + 2, z2 + 2,
                fill='#888888', width=8, tags='robot_arm'
            )

            # Draw main link
            self.canvas.create_line(
                x1, z1, x2, z2,
                fill=link_colors[i], width=6, tags='robot_arm'
            )

        # Draw joints as circles
        joint_names = ['Base', 'Boom', 'Arm', 'Bucket', 'Tool']
        for i, (cx, cz) in enumerate(canvas_points):
            radius = 8 if i == 0 else 6

            # Joint shadow
            self.canvas.create_oval(
                cx - radius + 2, cz - radius + 2,
                cx + radius + 2, cz + radius + 2,
                fill='#666666', outline='', tags='robot_arm'
            )

            # Joint
            joint_color = '#003366' if i == 0 else '#0066CC'
            self.canvas.create_oval(
                cx - radius, cz - radius,
                cx + radius, cz + radius,
                fill=joint_color, outline='white', width=2, tags='robot_arm'
            )

            # Joint label (on first joint only to avoid clutter)
            if i == 0 or i == len(canvas_points) - 1:
                self.canvas.create_text(
                    cx, cz - radius - 12,
                    text=joint_names[i], font=('Arial', 8, 'bold'),
                    fill='#003366', tags='robot_arm'
                )

        # Draw end-effector position (with ee_offset applied)
        ee_canvas_x, ee_canvas_z = self._world_to_canvas(ee_position[0], ee_position[2])
        ee_radius = 7

        # EE shadow
        self.canvas.create_oval(
            ee_canvas_x - ee_radius + 2, ee_canvas_z - ee_radius + 2,
            ee_canvas_x + ee_radius + 2, ee_canvas_z + ee_radius + 2,
            fill='#666666', outline='', tags='robot_arm'
        )

        # EE point (purple/magenta to distinguish from joints)
        self.canvas.create_oval(
            ee_canvas_x - ee_radius, ee_canvas_z - ee_radius,
            ee_canvas_x + ee_radius, ee_canvas_z + ee_radius,
            fill='#CC00CC', outline='white', width=2, tags='robot_arm'
        )

        # EE crosshair for precision
        self.canvas.create_line(
            ee_canvas_x - 10, ee_canvas_z,
            ee_canvas_x + 10, ee_canvas_z,
            fill='#CC00CC', width=2, tags='robot_arm'
        )
        self.canvas.create_line(
            ee_canvas_x, ee_canvas_z - 10,
            ee_canvas_x, ee_canvas_z + 10,
            fill='#CC00CC', width=2, tags='robot_arm'
        )

        # EE label
        self.canvas.create_text(
            ee_canvas_x, ee_canvas_z - ee_radius - 12,
            text='EE', font=('Arial', 8, 'bold'),
            fill='#CC00CC', tags='robot_arm'
        )

        # Draw position data age indicator
        if self.last_joint_update_time > 0:
            age = time.time() - self.last_joint_update_time
            if age < 0.5:
                status_color = 'green'
                status_text = "● LIVE"
            elif age < 2.0:
                status_color = 'orange'
                status_text = "● OLD"
            else:
                status_color = 'red'
                status_text = "● STALE"

            self.canvas.create_text(
                self.CANVAS_WIDTH - 60, 35,
                text=status_text, font=('Arial', 10, 'bold'),
                fill=status_color, tags='robot_arm'
            )

    def _update_display(self):
        """Update position display"""
        # Update position labels
        self.x_label.config(text=f"X: {self.current_x:+.3f} m")
        self.z_label.config(text=f"Z: {self.current_z:+.3f} m")
        # Fixed width formatting to prevent layout shifts
        self.rot_label.config(text=f"Rot Y: {self.current_rot_y:+5.1f}°")

    def _on_rotation_change(self, value):
        """Handle rotation slider change"""
        self.current_rot_y = float(value)
        self._update_display()

    def _center_rotation(self):
        """Center the rotation slider to 0°"""
        self.current_rot_y = 0.0
        self.rot_slider.set(0.0)
        self._update_display()

    def _bind_events(self):
        """Bind mouse events"""
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

    def _on_click(self, event):
        """Handle mouse click"""
        # Check if click is on the dot
        canvas_x, canvas_z = self._world_to_canvas(self.current_x, self.current_z)
        distance = math.sqrt((event.x - canvas_x)**2 + (event.y - canvas_z)**2)

        if distance <= self.DOT_RADIUS + 5:
            self.dragging = True
        else:
            # Click outside dot - move dot to clicked position
            x, z = self._canvas_to_world(event.x, event.y)
            self.current_x, self.current_z = self._clamp_position(x, z)
            self._draw_dot()
            self._update_display()

    def _on_drag(self, event):
        """Handle mouse drag"""
        if self.dragging:
            x, z = self._canvas_to_world(event.x, event.y)
            self.current_x, self.current_z = self._clamp_position(x, z)
            self._draw_dot()
            self._update_display()

    def _on_release(self, event):
        """Handle mouse release"""
        self.dragging = False

    def _copy_command(self):
        """Copy position to clipboard"""
        # Generate command string
        if abs(self.current_rot_y) < 0.1:
            command = f"python set_pose.py {self.current_x:.3f} {self.Y_POSITION:.3f} {self.current_z:.3f}"
        else:
            command = f"python set_pose.py {self.current_x:.3f} {self.Y_POSITION:.3f} {self.current_z:.3f}  # Rot: {self.current_rot_y:.1f}°"

        self.root.clipboard_clear()
        self.root.clipboard_append(command)

        # Visual feedback
        original_text = self.copy_btn['text']
        self.copy_btn['text'] = "Copied!"
        self.root.after(1000, lambda: self.copy_btn.config(text=original_text))

    def _reset_position(self):
        """Reset to default position"""
        self.current_x = self.DEFAULT_X
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0
        self.rot_slider.set(0.0)
        self._draw_dot()
        self._update_display()

    def _toggle_flip(self):
        """Toggle X-axis flip to view from opposite side"""
        self.flip_x = not self.flip_x

        # Update button text to show current state
        if self.flip_x:
            self.flip_btn.config(text="Flip View ⇄ (Flipped)")
        else:
            self.flip_btn.config(text="Flip View ⇄")

        # Redraw entire workspace with flipped view
        self._draw_workspace()
        self._draw_robot_arm()  # Redraw robot arm if present

        print(f"View flipped: {'ON' if self.flip_x else 'OFF'}")

    # ============================================================
    # UDP CONNECTION METHODS
    # ============================================================

    @staticmethod
    def _encode_position_to_bytes(x, y, z, rot_y):
        """
        Encode position and rotation values [x, y, z, rot_y] into 8 bytes (2 bytes per value).

        Uses 16-bit signed integers for better precision than 8-bit.
        Position range: -32768 to 32767 per value (maps to approx -2.58m to +2.58m with 0.00008m resolution)
        Rotation range: -180° to +180° with 0.0055° resolution

        Args:
            x, y, z: Position values in meters
            rot_y: Y-axis rotation in degrees

        Returns:
            List of 8 signed bytes [x_high, x_low, y_high, y_low, z_high, z_low, rot_high, rot_low]
        """
        # Scale factor: use most of 16-bit range for typical workspace (-2m to +2m)
        # 32767 / 2.5 = ~13107 counts per meter, or ~0.08mm resolution
        SCALE_POS = 13107.0  # counts per meter
        # For rotation: 32767 / 180 = ~182 counts per degree, or ~0.0055° resolution
        SCALE_ROT = 182.0  # counts per degree

        # Convert to 16-bit integers
        x_int = int(round(x * SCALE_POS))
        y_int = int(round(y * SCALE_POS))
        z_int = int(round(z * SCALE_POS))
        rot_int = int(round(rot_y * SCALE_ROT))

        # Clamp to 16-bit signed range
        x_int = max(-32768, min(32767, x_int))
        y_int = max(-32768, min(32767, y_int))
        z_int = max(-32768, min(32767, z_int))
        rot_int = max(-32768, min(32767, rot_int))

        # Split each 16-bit value into high and low bytes
        # Using two's complement for negative numbers
        x_high = (x_int >> 8) & 0xFF
        x_low = x_int & 0xFF
        y_high = (y_int >> 8) & 0xFF
        y_low = y_int & 0xFF
        z_high = (z_int >> 8) & 0xFF
        z_low = z_int & 0xFF
        rot_high = (rot_int >> 8) & 0xFF
        rot_low = rot_int & 0xFF

        # Convert to signed bytes [-128, 127]
        def to_signed_byte(b):
            return b if b < 128 else b - 256

        return [
            to_signed_byte(x_high), to_signed_byte(x_low),
            to_signed_byte(y_high), to_signed_byte(y_low),
            to_signed_byte(z_high), to_signed_byte(z_low),
            to_signed_byte(rot_high), to_signed_byte(rot_low)
        ]

    @staticmethod
    def _decode_bytes_to_position(bytes_list):
        """
        Decode 6 bytes back to position values [x, y, z].

        This is the inverse of _encode_position_to_bytes().
        Use this on the receiver side.

        Args:
            bytes_list: List of 6 signed bytes [-128, 127]

        Returns:
            [x, y, z] position in meters
        """
        SCALE = 13107.0

        # Convert signed bytes back to unsigned
        def to_unsigned_byte(b):
            return b if b >= 0 else b + 256

        bytes_unsigned = [to_unsigned_byte(b) for b in bytes_list]

        # Reconstruct 16-bit integers
        x_int = (bytes_unsigned[0] << 8) | bytes_unsigned[1]
        y_int = (bytes_unsigned[2] << 8) | bytes_unsigned[3]
        z_int = (bytes_unsigned[4] << 8) | bytes_unsigned[5]

        # Convert back to signed 16-bit
        if x_int >= 32768:
            x_int -= 65536
        if y_int >= 32768:
            y_int -= 65536
        if z_int >= 32768:
            z_int -= 65536

        # Scale back to meters
        x = x_int / SCALE
        y = y_int / SCALE
        z = z_int / SCALE

        return [x, y, z]

    @staticmethod
    def _decode_joint_positions_from_bytes(bytes_list):
        """
        Decode 24 bytes to 4 joint positions.

        Args:
            bytes_list: List of 24 signed bytes [-128, 127]

        Returns:
            List of 4 positions [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
        """
        SCALE = 13107.0

        def to_unsigned_byte(b):
            return b if b >= 0 else b + 256

        joint_positions = []

        # Process each joint (6 bytes per joint)
        for i in range(4):
            start_idx = i * 6
            bytes_slice = bytes_list[start_idx:start_idx + 6]
            bytes_unsigned = [to_unsigned_byte(b) for b in bytes_slice]

            # Reconstruct 16-bit integers for x, y, z
            x_int = (bytes_unsigned[0] << 8) | bytes_unsigned[1]
            y_int = (bytes_unsigned[2] << 8) | bytes_unsigned[3]
            z_int = (bytes_unsigned[4] << 8) | bytes_unsigned[5]

            # Convert back to signed 16-bit
            if x_int >= 32768:
                x_int -= 65536
            if y_int >= 32768:
                y_int -= 65536
            if z_int >= 32768:
                z_int -= 65536

            # Scale back to meters
            x = x_int / SCALE
            y = y_int / SCALE
            z = z_int / SCALE

            joint_positions.append([x, y, z])

        return joint_positions

    def _connect_udp(self):
        """Connect to UDP server"""
        try:
            print(f"\nConnecting to {self.UDP_HOST}:{self.UDP_PORT}...")
            print(f"Using 16-bit encoding: 2 bytes per value (8 bytes total: x, y, z, rot_y)")

            # Create UDP client
            self.udp_client = UDPSocket(local_id=self.UDP_LOCAL_ID, max_age_seconds=0.5)

            # Setup as client: send 6 bytes (target position), receive 24 bytes (joint positions)
            self.udp_client.setup(
                self.UDP_HOST,
                self.UDP_PORT,
                num_inputs=self.UDP_NUM_BYTES_RECV,
                num_outputs=self.UDP_NUM_BYTES_SEND,
                is_server=False
            )

            # Perform handshake
            if self.udp_client.handshake(timeout=5.0):
                self.udp_connected = True
                self._update_connection_status()
                print("✓ Connected successfully!")

                # Enable streaming button
                self.stream_btn.config(state=tk.NORMAL)
                self.disconnect_btn.config(state=tk.NORMAL)
                self.connect_btn.config(state=tk.DISABLED)
            else:
                raise Exception("Handshake failed")

        except Exception as e:
            print(f"Connection failed: {e}")
            self.udp_connected = False
            self._update_connection_status()
            if self.udp_client:
                self.udp_client.close()
                self.udp_client = None

    def _disconnect_udp(self):
        """Disconnect from UDP server"""
        # Stop streaming first
        if self.udp_streaming:
            self._stop_streaming()

        # Close connection
        if self.udp_client:
            self.udp_client.close()
            self.udp_client = None

        self.udp_connected = False
        self.packets_sent = 0
        self._update_connection_status()

        # Update button states
        self.connect_btn.config(state=tk.NORMAL)
        self.stream_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.DISABLED)

        print("Disconnected from UDP server")

    def _toggle_streaming(self):
        """Toggle UDP streaming on/off"""
        if self.udp_streaming:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        """Start UDP streaming thread"""
        if not self.udp_connected or not self.udp_client:
            print("Not connected to UDP server!")
            return

        print(f"\nStarting UDP streaming at {self.UDP_SEND_RATE_HZ} Hz...")
        self.udp_streaming = True
        self.send_thread_running = True
        self.packets_sent = 0

        # Start UDP receiver thread (reads joint positions from robot)
        self.udp_client.start_receiving()

        # Start send thread
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()

        # Start receive/visualize thread
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()

        # Update UI
        self.stream_btn.config(text="Stop Streaming")
        self._update_connection_status()

    def _stop_streaming(self):
        """Stop UDP streaming thread"""
        print("\nStopping UDP streaming...")
        self.udp_streaming = False
        self.send_thread_running = False

        # Stop UDP receiver
        if self.udp_client:
            self.udp_client.stop_receiving()

        # Wait for threads to finish
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)

        # Update UI
        self.stream_btn.config(text="Start Streaming")
        self._update_connection_status()
        print("Streaming stopped")

    def _receive_loop(self):
        """Background thread to receive joint positions and update visualization"""
        while self.send_thread_running:
            try:
                # Get latest joint position data from UDP
                data = self.udp_client.get_latest()

                if data and len(data) == self.UDP_NUM_BYTES_RECV:
                    # Decode 24 bytes to 4 joint positions
                    joint_positions = self._decode_joint_positions_from_bytes(data)

                    # Update stored positions
                    self.joint_positions = joint_positions
                    self.last_joint_update_time = time.time()

                    # Trigger GUI update (redraw robot arm)
                    self.root.after(0, self._draw_robot_arm)

                time.sleep(0.01)  # Check at 100Hz

            except Exception as e:
                print(f"Receive error: {e}")
                time.sleep(0.1)

    def _send_loop(self):
        """Background thread to send position data at fixed rate"""
        send_period = 1.0 / self.UDP_SEND_RATE_HZ
        next_send_time = time.perf_counter()

        while self.send_thread_running:
            try:
                # Encode position and rotation into 8 bytes (2 bytes per value for 16-bit precision)
                encoded_bytes = self._encode_position_to_bytes(
                    self.current_x,
                    self.Y_POSITION,
                    self.current_z,
                    self.current_rot_y
                )

                # Send via UDP
                if self.udp_client:
                    self.udp_client.send(encoded_bytes)
                    self.packets_sent += 1
                    self.last_send_time = time.time()

                    # Update packet counter every 10 packets (reduce UI updates)
                    if self.packets_sent % 10 == 0:
                        self.root.after(0, self._update_packet_count)

                # Sleep until next send time
                next_send_time += send_period
                sleep_time = next_send_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Behind schedule, reset
                    next_send_time = time.perf_counter()

            except Exception as e:
                print(f"Send error: {e}")
                time.sleep(0.1)

    def _update_connection_status(self):
        """Update connection status display"""
        if self.udp_streaming:
            self.conn_status_label.config(text="● Streaming", foreground='green')
        elif self.udp_connected:
            self.conn_status_label.config(text="● Connected", foreground='orange')
        else:
            self.conn_status_label.config(text="● Disconnected", foreground='red')

        self._update_packet_count()

    def _update_packet_count(self):
        """Update packet counter display"""
        self.packets_label.config(text=f"Packets: {self.packets_sent}")

    def _on_closing(self):
        """Handle window close event"""
        print("\nShutting down...")

        # Stop streaming and disconnect
        if self.udp_streaming:
            self._stop_streaming()
        if self.udp_connected:
            self._disconnect_udp()

        # Close window
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ExcavatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
