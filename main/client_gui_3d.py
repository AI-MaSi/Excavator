#!/usr/bin/env python3
"""
Excavator 3D Position GUI - Keyboard control with UDP streaming

3D view of the workspace using Tkinter + Matplotlib.
Keyboard controls move the target in XYZ and adjust Y-rotation.

Controls:
- A / D: X− / X+
- W / S: Z+ / Z−
- Q / E: Y− / Y+
- R / F: Y-rotation + / − (degrees)
- Shift: 5× faster step

Sends position commands via UDP at 20Hz when streaming is enabled and optionally
renders joint positions received from the robot as a simple 3D linkage.

Usage:
    python client_gui_3d.py

Requires:
    - matplotlib (for 3D rendering)
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import threading
import sys
from pathlib import Path

# 3D rendering (Matplotlib embedded in Tk)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from udp_socket import UDPSocket


class ExcavatorGUI3D:
    """Interactive 3D GUI for selecting excavator end-effector pose"""

    # Workspace limits in meters (adjust based on excavator reach)
    WORKSPACE_LIMITS = {
        'x_min': 0.1,
        'x_max': 0.7,
        'y_min': -0.30,
        'y_max': 0.30,
        'z_min': -0.3,
        'z_max': 0.4,
    }

    # Defaults
    DEFAULT_X = 0.6
    DEFAULT_Y = 0.0
    DEFAULT_Z = 0.0

    # UDP Configuration
    UDP_HOST = "192.168.0.132"
    UDP_PORT = 8080
    UDP_LOCAL_ID = 3
    UDP_SEND_RATE_HZ = 20
    UDP_NUM_BYTES_SEND = 9   # x,y,z,rot_y (2 bytes each) + reload flag
    UDP_NUM_BYTES_RECV = 24  # 4 joints * (x,y,z) * 2 bytes

    # Keyboard step sizes
    KEY_STEP_POS = 0.01   # meters per tick (default)
    KEY_STEP_ROT = 1.0    # degrees per tick (default)
    FAST_MULTIPLIER = 5.0 # when Shift is held

    def __init__(self, root):
        self.root = root
        self.root.title("Excavator 3D Position Control GUI - UDP Streaming")

        # Current pose
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0  # degrees

        # Key state
        self.keys_down = set()

        # User-adjustable step sizes
        self.step_pos = self.KEY_STEP_POS
        self.step_rot = self.KEY_STEP_ROT

        # UDP state
        self.udp_client = None
        self.udp_connected = False
        self.udp_streaming = False
        self.send_thread = None
        self.recv_thread = None
        self.send_thread_running = False
        self.packets_sent = 0
        self.last_send_time = 0.0
        self.reload_config_flag = 0.0

        # Received joint positions for visualization
        self.joint_positions = None  # [[x,y,z], ...] length 4
        self.last_joint_update_time = 0.0

        # Build UI
        self._create_widgets()
        self._bind_events()
        self._init_plot()
        self._redraw_scene()

        # Start keyboard tick loop (60 FPS)
        self._schedule_tick()

        # Window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ---------------- UI ----------------
    def _create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        title = ttk.Label(main, text="Excavator 3D Position Control", font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        # Matplotlib figure embedded in Tk
        self.fig = Figure(figsize=(7.5, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Pose labels
        info = ttk.LabelFrame(main, text="Current Pose", padding="8")
        info.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        row = ttk.Frame(info)
        row.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.x_label = ttk.Label(row, text="X: 0.000 m", font=('Courier', 12))
        self.y_label = ttk.Label(row, text="Y: 0.000 m", font=('Courier', 12))
        self.z_label = ttk.Label(row, text="Z: 0.000 m", font=('Courier', 12))
        self.rot_label = ttk.Label(row, text="Rot Y:   0.0°", font=('Courier', 12), width=14)
        self.x_label.pack(side=tk.LEFT, padx=(0, 14))
        self.y_label.pack(side=tk.LEFT, padx=(0, 14))
        self.z_label.pack(side=tk.LEFT, padx=(0, 14))
        self.rot_label.pack(side=tk.LEFT)

        # Control settings (step sizes)
        settings = ttk.LabelFrame(main, text="Control Settings", padding="8")
        settings.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8,0))

        ttk.Label(settings, text="Move step (m):").grid(row=0, column=0, sticky=tk.W)
        self.pos_step_var = tk.StringVar(value=f"{self.step_pos:.3f}")
        self.pos_step_spin = tk.Spinbox(settings, from_=0.001, to=0.100, increment=0.001,
                                        textvariable=self.pos_step_var, width=8,
                                        command=self._apply_pos_step)
        self.pos_step_spin.grid(row=0, column=1, padx=(6,12))
        self.pos_step_spin.bind('<Return>', lambda e: self._apply_pos_step())
        self.pos_step_spin.bind('<FocusOut>', lambda e: self._apply_pos_step())

        ttk.Label(settings, text="Rot step (deg):").grid(row=0, column=2, sticky=tk.W)
        self.rot_step_var = tk.StringVar(value=f"{self.step_rot:.1f}")
        self.rot_step_spin = tk.Spinbox(settings, from_=0.1, to=20.0, increment=0.1,
                                        textvariable=self.rot_step_var, width=8,
                                        command=self._apply_rot_step)
        self.rot_step_spin.grid(row=0, column=3, padx=(6,12))
        self.rot_step_spin.bind('<Return>', lambda e: self._apply_rot_step())
        self.rot_step_spin.bind('<FocusOut>', lambda e: self._apply_rot_step())

        # Buttons
        btns = ttk.Frame(main)
        btns.grid(row=4, column=0, columnspan=3, pady=(8,0))
        self.copy_btn = ttk.Button(btns, text="Copy Position", command=self._copy_command)
        self.reset_btn = ttk.Button(btns, text="Reset (0.6, 0.0, 0°)", command=self._reset_position)
        self.copy_btn.grid(row=0, column=0, padx=4)
        self.reset_btn.grid(row=0, column=1, padx=4)

        # UDP controls
        udp = ttk.LabelFrame(main, text="UDP Streaming Control", padding="8")
        udp.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8,0))

        top = ttk.Frame(udp)
        top.grid(row=0, column=0, columnspan=3, pady=(0,6))
        ttk.Label(top, text=f"Target: {self.UDP_HOST}:{self.UDP_PORT}", font=('Courier', 9)).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(top, text=f"Rate: {self.UDP_SEND_RATE_HZ} Hz", font=('Courier', 9)).grid(row=0, column=1, sticky=tk.W, padx=(20,0))

        stat = ttk.Frame(udp)
        stat.grid(row=1, column=0, columnspan=3, pady=(0,6))
        self.conn_status_label = ttk.Label(stat, text="● Disconnected", font=('Arial', 10), foreground='red')
        self.packets_label = ttk.Label(stat, text="Packets: 0", font=('Courier', 9))
        self.conn_status_label.grid(row=0, column=0, sticky=tk.W)
        self.packets_label.grid(row=0, column=1, sticky=tk.W, padx=(20,0))

        self.connect_btn = ttk.Button(udp, text="Connect", command=self._connect_udp)
        self.stream_btn = ttk.Button(udp, text="Start Streaming", command=self._toggle_streaming, state=tk.DISABLED)
        self.disconnect_btn = ttk.Button(udp, text="Disconnect", command=self._disconnect_udp, state=tk.DISABLED)
        self.reload_btn = ttk.Button(udp, text="Reload Config", command=self._reload_config, state=tk.DISABLED)
        self.connect_btn.grid(row=2, column=0, padx=4, pady=4)
        self.stream_btn.grid(row=2, column=1, padx=4, pady=4)
        self.disconnect_btn.grid(row=2, column=2, padx=4, pady=4)
        self.reload_btn.grid(row=3, column=0, columnspan=3, padx=4, pady=(6,4))

        # Instructions
        ttk.Label(main,
                  text="Controls: A/D=X, W/S=Z, Q/E=Y, R/F=RotY (limit ±45°), Shift=fast",
                  font=('Arial', 10, 'italic')).grid(row=6, column=0, columnspan=3, pady=(8,0))

        # Layout stretch
        for c in range(3):
            main.columnconfigure(c, weight=1)
        main.rowconfigure(1, weight=1)

    def _bind_events(self):
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.bind('<KeyRelease>', self._on_key_release)

    # -------------- 3D Drawing --------------
    def _init_plot(self):
        self.ax.view_init(elev=25, azim=-60)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')

    def _set_axes_limits(self):
        xl = (self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['x_max'])
        yl = (self.WORKSPACE_LIMITS['y_min'], self.WORKSPACE_LIMITS['y_max'])
        zl = (self.WORKSPACE_LIMITS['z_min'], self.WORKSPACE_LIMITS['z_max'])
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)
        self.ax.set_zlim(zl)

        # Try to keep aspect roughly equal
        max_range = max(xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0])
        x_mid = sum(xl)/2
        y_mid = sum(yl)/2
        z_mid = sum(zl)/2
        r = max_range/2
        self.ax.set_xlim(x_mid-r, x_mid+r)
        self.ax.set_ylim(y_mid-r, y_mid+r)
        self.ax.set_zlim(z_mid-r, z_mid+r)

    def _draw_workspace_box(self):
        # Draw a transparent bounding box of the workspace
        x0, x1 = self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['x_max']
        y0, y1 = self.WORKSPACE_LIMITS['y_min'], self.WORKSPACE_LIMITS['y_max']
        z0, z1 = self.WORKSPACE_LIMITS['z_min'], self.WORKSPACE_LIMITS['z_max']
        edges = [
            ((x0,y0,z0),(x1,y0,z0)), ((x0,y1,z0),(x1,y1,z0)),
            ((x0,y0,z1),(x1,y0,z1)), ((x0,y1,z1),(x1,y1,z1)),
            ((x0,y0,z0),(x0,y1,z0)), ((x1,y0,z0),(x1,y1,z0)),
            ((x0,y0,z1),(x0,y1,z1)), ((x1,y0,z1),(x1,y1,z1)),
            ((x0,y0,z0),(x0,y0,z1)), ((x1,y0,z0),(x1,y0,z1)),
            ((x0,y1,z0),(x0,y1,z1)), ((x1,y1,z0),(x1,y1,z1)),
        ]
        for (xa,ya,za),(xb,yb,zb) in edges:
            self.ax.plot([xa,xb],[ya,yb],[za,zb], color='#cccccc', linewidth=1)

        # Draw axis zero lines for reference
        self.ax.plot([0,0],[y0,y1],[0,0], color='k', linestyle='--', linewidth=1)
        self.ax.plot([x0,x1],[0,0],[0,0], color='b', linestyle='--', linewidth=1)
        self.ax.plot([0,0],[0,0],[z0,z1], color='g', linestyle='--', linewidth=1)

    def _draw_target(self):
        # A red sphere marker (approximate with scatter)
        self.ax.scatter([self.current_x],[self.current_y],[self.current_z],
                        s=40, c='red', depthshade=True)

    def _draw_robot(self):
        if self.joint_positions is None or len(self.joint_positions) != 4:
            return

        # Joints: [boom_mount, arm_mount, bucket_mount, tool_mount]
        positions = [[0.0, 0.0, 0.0]] + self.joint_positions

        # Compute EE offset like 2D GUI (local offset in bucket frame)
        EE_OFFSET_LOCAL = [0.0, 0.0, -0.142]
        bucket = self.joint_positions[2]
        tool = self.joint_positions[3]
        link = [tool[0]-bucket[0], tool[1]-bucket[1], tool[2]-bucket[2]]
        ll = math.sqrt(link[0]**2 + link[1]**2 + link[2]**2)
        if ll > 1e-6:
            local_x = [link[0]/ll, link[1]/ll, link[2]/ll]
            local_y = [0.0, 1.0, 0.0]
            local_z = [
                local_x[1]*local_y[2] - local_x[2]*local_y[1],
                local_x[2]*local_y[0] - local_x[0]*local_y[2],
                local_x[0]*local_y[1] - local_x[1]*local_y[0]
            ]
            ee_offset_world = [
                EE_OFFSET_LOCAL[0]*local_x[0] + EE_OFFSET_LOCAL[1]*local_y[0] + EE_OFFSET_LOCAL[2]*local_z[0],
                EE_OFFSET_LOCAL[0]*local_x[1] + EE_OFFSET_LOCAL[1]*local_y[1] + EE_OFFSET_LOCAL[2]*local_z[1],
                EE_OFFSET_LOCAL[0]*local_x[2] + EE_OFFSET_LOCAL[1]*local_y[2] + EE_OFFSET_LOCAL[2]*local_z[2]
            ]
            ee = [tool[0]+ee_offset_world[0], tool[1]+ee_offset_world[1], tool[2]+ee_offset_world[2]]
        else:
            ee = tool

        xs = [p[0] for p in positions] + [ee[0]]
        ys = [p[1] for p in positions] + [ee[1]]
        zs = [p[2] for p in positions] + [ee[2]]
        # Draw links
        self.ax.plot(xs[:-1], ys[:-1], zs[:-1], color='#FF9933', linewidth=4)
        # End-effector segment
        self.ax.plot([positions[-1][0], ee[0]], [positions[-1][1], ee[1]], [positions[-1][2], ee[2]],
                     color='#FFCC66', linewidth=3)
        # Draw joints
        self.ax.scatter(xs[:-1], ys[:-1], zs[:-1], s=25, c='#333333')

    def _redraw_scene(self):
        self.ax.cla()
        self._set_axes_limits()
        self._draw_workspace_box()
        self._draw_robot()
        self._draw_target()
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.canvas.draw_idle()
        self._update_labels()

    # -------------- Keyboard control --------------
    def _on_key_press(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l','shift_r'):
            self.keys_down.add('shift')
        else:
            self.keys_down.add(key)

    def _on_key_release(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l','shift_r'):
            self.keys_down.discard('shift')
        else:
            self.keys_down.discard(key)

    def _schedule_tick(self):
        self._tick()
        self.root.after(16, self._schedule_tick)  # ~60Hz

    def _tick(self):
        # Compute step (fast with shift)
        mult = self.FAST_MULTIPLIER if 'shift' in self.keys_down else 1.0
        dp = self.step_pos * mult
        dr = self.step_rot * mult

        moved = False
        if 'a' in self.keys_down: self.current_x -= dp; moved = True
        if 'd' in self.keys_down: self.current_x += dp; moved = True
        if 'w' in self.keys_down: self.current_z += dp; moved = True
        if 's' in self.keys_down: self.current_z -= dp; moved = True
        if 'q' in self.keys_down: self.current_y -= dp; moved = True
        if 'e' in self.keys_down: self.current_y += dp; moved = True
        if 'r' in self.keys_down: self.current_rot_y += dr; moved = True
        if 'f' in self.keys_down: self.current_rot_y -= dr; moved = True

        if moved:
            self._clamp_pose()
            self._redraw_scene()

    def _clamp_pose(self):
        self.current_x = max(self.WORKSPACE_LIMITS['x_min'], min(self.current_x, self.WORKSPACE_LIMITS['x_max']))
        self.current_y = max(self.WORKSPACE_LIMITS['y_min'], min(self.current_y, self.WORKSPACE_LIMITS['y_max']))
        self.current_z = max(self.WORKSPACE_LIMITS['z_min'], min(self.current_z, self.WORKSPACE_LIMITS['z_max']))
        # Clamp rotation to [-45, 45] degrees (0 = horizon)
        if self.current_rot_y > 45.0:
            self.current_rot_y = 45.0
        if self.current_rot_y < -45.0:
            self.current_rot_y = -45.0

    def _update_labels(self):
        self.x_label.config(text=f"X: {self.current_x:.3f} m")
        self.y_label.config(text=f"Y: {self.current_y:.3f} m")
        self.z_label.config(text=f"Z: {self.current_z:.3f} m")
        self.rot_label.config(text=f"Rot Y: {self.current_rot_y:6.1f}°")

    def _copy_command(self):
        if abs(self.current_rot_y) < 0.1:
            cmd = f"python set_pose.py {self.current_x:.3f} {self.current_y:.3f} {self.current_z:.3f}"
        else:
            cmd = f"python set_pose.py {self.current_x:.3f} {self.current_y:.3f} {self.current_z:.3f}  # Rot: {self.current_rot_y:.1f}°"
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        orig = self.copy_btn['text']
        self.copy_btn['text'] = "Copied!"
        self.root.after(1000, lambda: self.copy_btn.config(text=orig))

    def _reset_position(self):
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0
        self._redraw_scene()

    # ---- Settings handlers ----
    def _apply_pos_step(self):
        try:
            val = float(self.pos_step_var.get())
            # reasonable bounds
            if not (0.0005 <= val <= 0.2):
                raise ValueError
            self.step_pos = val
        except Exception:
            # restore formatted string
            self.pos_step_var.set(f"{self.step_pos:.3f}")

    def _apply_rot_step(self):
        try:
            val = float(self.rot_step_var.get())
            if not (0.05 <= val <= 45.0):
                raise ValueError
            self.step_rot = val
        except Exception:
            self.rot_step_var.set(f"{self.step_rot:.1f}")

    # ---------------- UDP ----------------
    def _encode_position_to_bytes(self, x, y, z, rot_y):
        """
        Encode [x,y,z,rot_y] into 9 signed bytes (16-bit per value + 1 flag).
        Position scale ≈ 13107 counts/m; rotation ≈ 182 counts/deg.
        """
        SCALE_POS = 13107.0
        SCALE_ROT = 182.0
        x_i = int(round(x * SCALE_POS))
        y_i = int(round(y * SCALE_POS))
        z_i = int(round(z * SCALE_POS))
        r_i = int(round(rot_y * SCALE_ROT))
        x_i = max(-32768, min(32767, x_i))
        y_i = max(-32768, min(32767, y_i))
        z_i = max(-32768, min(32767, z_i))
        r_i = max(-32768, min(32767, r_i))
        def hb(v): return (v >> 8) & 0xFF
        def lb(v): return v & 0xFF
        def sb(b): return b if b < 128 else b - 256
        reload_b = int(round(self.reload_config_flag * 127.0))
        reload_b = max(-128, min(127, reload_b))
        return [sb(hb(x_i)), sb(lb(x_i)), sb(hb(y_i)), sb(lb(y_i)), sb(hb(z_i)), sb(lb(z_i)), sb(hb(r_i)), sb(lb(r_i)), reload_b]

    @staticmethod
    def _decode_joint_positions_from_bytes(bytes_list):
        SCALE = 13107.0
        def u8(b): return b if b >= 0 else b + 256
        joints = []
        for i in range(4):
            s = i*6
            bu = [u8(b) for b in bytes_list[s:s+6]]
            xi = (bu[0] << 8) | bu[1]
            yi = (bu[2] << 8) | bu[3]
            zi = (bu[4] << 8) | bu[5]
            if xi >= 32768: xi -= 65536
            if yi >= 32768: yi -= 65536
            if zi >= 32768: zi -= 65536
            joints.append([xi/SCALE, yi/SCALE, zi/SCALE])
        return joints

    def _connect_udp(self):
        try:
            print(f"\nConnecting to {self.UDP_HOST}:{self.UDP_PORT}...")
            self.udp_client = UDPSocket(local_id=self.UDP_LOCAL_ID, max_age_seconds=0.5)
            self.udp_client.setup(self.UDP_HOST, self.UDP_PORT,
                                  num_inputs=self.UDP_NUM_BYTES_RECV,
                                  num_outputs=self.UDP_NUM_BYTES_SEND,
                                  is_server=False)
            if self.udp_client.handshake(timeout=5.0):
                self.udp_connected = True
                self.stream_btn.config(state=tk.NORMAL)
                self.disconnect_btn.config(state=tk.NORMAL)
                self.connect_btn.config(state=tk.DISABLED)
                self._update_connection_status()
                print("✓ Connected successfully!")
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
        if self.udp_streaming:
            self._stop_streaming()
        if self.udp_client:
            self.udp_client.close()
            self.udp_client = None
        self.udp_connected = False
        self.packets_sent = 0
        self.connect_btn.config(state=tk.NORMAL)
        self.stream_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.reload_btn.config(state=tk.DISABLED)
        self._update_connection_status()
        print("Disconnected from UDP server")

    def _reload_config(self):
        if not self.udp_connected or not self.udp_streaming:
            print("Must be streaming to reload config!")
            return
        print("Sending reload config command...")
        self.reload_config_flag = 1.0
        orig = self.reload_btn['text']
        self.reload_btn['text'] = "✓ Reload Sent!"
        self.reload_btn.config(state=tk.DISABLED)
        def reset_btn():
            self.reload_btn['text'] = orig
            if self.udp_streaming:
                self.reload_btn.config(state=tk.NORMAL)
        self.root.after(2000, reset_btn)

    def _toggle_streaming(self):
        if self.udp_streaming:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        if not self.udp_connected or not self.udp_client:
            print("Not connected to UDP server!")
            return
        print(f"\nStarting UDP streaming at {self.UDP_SEND_RATE_HZ} Hz...")
        self.udp_streaming = True
        self.send_thread_running = True
        self.packets_sent = 0
        self.udp_client.start_receiving()
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()
        self.stream_btn.config(text="Stop Streaming")
        self.reload_btn.config(state=tk.NORMAL)
        self._update_connection_status()

    def _stop_streaming(self):
        print("\nStopping UDP streaming...")
        self.udp_streaming = False
        self.send_thread_running = False
        if self.udp_client:
            self.udp_client.stop_receiving()
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
        self.stream_btn.config(text="Start Streaming")
        self.reload_btn.config(state=tk.DISABLED)
        self._update_connection_status()
        print("Streaming stopped")

    def _receive_loop(self):
        while self.send_thread_running:
            try:
                data = self.udp_client.get_latest() if self.udp_client else None
                if data and len(data) == self.UDP_NUM_BYTES_RECV:
                    self.joint_positions = self._decode_joint_positions_from_bytes(data)
                    self.last_joint_update_time = time.time()
                    self.root.after(0, self._redraw_scene)
                time.sleep(0.01)
            except Exception as e:
                print(f"Receive error: {e}")
                time.sleep(0.1)

    def _send_loop(self):
        period = 1.0 / self.UDP_SEND_RATE_HZ
        next_t = time.perf_counter()
        while self.send_thread_running:
            try:
                payload = self._encode_position_to_bytes(self.current_x, self.current_y, self.current_z, self.current_rot_y)
                if self.udp_client:
                    self.udp_client.send(payload)
                    self.packets_sent += 1
                    self.last_send_time = time.time()
                    if self.reload_config_flag > 0.5:
                        self.reload_config_flag = 0.0
                    if self.packets_sent % 10 == 0:
                        self.root.after(0, self._update_packet_count)
                next_t += period
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    next_t = time.perf_counter()
            except Exception as e:
                print(f"Send error: {e}")
                time.sleep(0.1)

    def _update_connection_status(self):
        if self.udp_streaming:
            self.conn_status_label.config(text="● Streaming", foreground='green')
        elif self.udp_connected:
            self.conn_status_label.config(text="● Connected", foreground='orange')
        else:
            self.conn_status_label.config(text="● Disconnected", foreground='red')
        self._update_packet_count()

    def _update_packet_count(self):
        self.packets_label.config(text=f"Packets: {self.packets_sent}")

    # ---------------- Shutdown ----------------
    def _on_closing(self):
        print("\nShutting down...")
        if self.udp_streaming:
            self._stop_streaming()
        if self.udp_connected:
            self._disconnect_udp()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ExcavatorGUI3D(root)
    root.mainloop()


if __name__ == "__main__":
    main()
