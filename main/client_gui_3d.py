#!/usr/bin/env python3
"""
Excavator 3D Position GUI - Keyboard control with UDP streaming

- Uses UDPSocket handshake + timestamped packets
- Sends 9 signed bytes: x,y,z,rot_y (big-endian int16 each) + reload_flag
- Receives 30 signed bytes: 5 positions * (x,y,z) int16
- Keyboard controls:
    A / D : X− / X+
    W / S : Z+ / Z−
    Q / E : Y− / Y+
    R / F : Pitch + / − (arrow in X–Z plane)
    Shift : 5× faster step
"""

import math
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from modules.udp_socket import UDPSocket  # same module used by excv_gui_log.py


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

    # UDP Configuration (must match excv_gui_log.py)
    UDP_HOST = "192.168.0.132"
    UDP_PORT = 8080
    UDP_LOCAL_ID = 3              # server uses 2, GUI uses 3
    UDP_SEND_RATE_HZ = 20.0
    UDP_NUM_BYTES_SEND = 9        # x,y,z,rot_y (2 bytes each) + reload flag
    UDP_NUM_BYTES_RECV = 30       # 5 positions * (x,y,z) * 2 bytes

    # Encoding scales (same as decode_bytes_to_position / encode_joint_positions_to_bytes)
    SCALE_POS = 13107.0           # meters <-> int16
    SCALE_ROT = 182.0             # degrees <-> int16

    # Hz monitoring
    HZ_UPDATE_INTERVAL = 0.5      # Update Hz display every 0.5 s

    # Keyboard step sizes (user wanted these defaults)
    KEY_STEP_POS = 0.001          # meters per tick
    KEY_STEP_ROT = 0.1            # degrees per tick
    FAST_MULTIPLIER = 5.0         # when Shift is held

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Excavator 3D Position Control GUI - UDP Streaming")

        # Redraw throttling
        self._last_redraw_time = 0.0
        self._redraw_interval = 1.0 / 30.0  # seconds, limit redraws to ~30 FPS

        # Current pose
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0  # interpreted as pitch for the arrow

        # Key state
        self.keys_down = set()

        # User-adjustable step sizes
        self.step_pos = self.KEY_STEP_POS
        self.step_rot = self.KEY_STEP_ROT

        # UDP state
        self.udp: Optional[UDPSocket] = None
        self.udp_connected = False
        self.udp_streaming = False
        self.send_thread: Optional[threading.Thread] = None
        self.recv_thread: Optional[threading.Thread] = None
        self.send_thread_running = False
        self.reload_flag = 0  # 0 or 1, mapped to last byte (scaled to 0 or 127)

        # Received joint positions for visualization (list of [x,y,z] in meters)
        self.joint_positions: Optional[List[List[float]]] = None

        # Hz monitoring
        self.send_hz_actual = 0.0
        self.send_count_window = 0
        self.send_time_window_start = 0.0

        # Build UI
        self._create_widgets()
        self._bind_events()
        self._init_plot()
        self._redraw_scene()

        # Start keyboard tick loop (~30 FPS)
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
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))
        canvas_widget.bind("<Configure>", self._on_canvas_resize)

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
        settings.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        ttk.Label(settings, text="Move step (m):").grid(row=0, column=0, sticky=tk.W)
        self.pos_step_var = tk.StringVar(value=f"{self.step_pos:.3f}")
        self.pos_step_spin = tk.Spinbox(
            settings,
            from_=0.001,
            to=0.100,
            increment=0.001,
            textvariable=self.pos_step_var,
            width=8,
            command=self._apply_pos_step,
        )
        self.pos_step_spin.grid(row=0, column=1, padx=(6, 12))
        self.pos_step_spin.bind('<Return>', lambda e: self._apply_pos_step())
        self.pos_step_spin.bind('<FocusOut>', lambda e: self._apply_pos_step())

        ttk.Label(settings, text="Rot step (deg):").grid(row=0, column=2, sticky=tk.W)
        self.rot_step_var = tk.StringVar(value=f"{self.step_rot:.1f}")
        self.rot_step_spin = tk.Spinbox(
            settings,
            from_=0.1,
            to=20.0,
            increment=0.1,
            textvariable=self.rot_step_var,
            width=8,
            command=self._apply_rot_step,
        )
        self.rot_step_spin.grid(row=0, column=3, padx=(6, 12))
        self.rot_step_spin.bind('<Return>', lambda e: self._apply_rot_step())
        self.rot_step_spin.bind('<FocusOut>', lambda e: self._apply_rot_step())

        ttk.Label(settings, text="Hold Shift for 5× step size").grid(
            row=0, column=4, sticky=tk.W, padx=(16, 0)
        )

        # UDP settings
        cfg = ttk.LabelFrame(main, text="UDP Config", padding="8")
        cfg.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        ttk.Label(cfg, text="Host:").grid(row=0, column=0, sticky=tk.W)
        self.host_var = tk.StringVar(value=self.UDP_HOST)
        self.host_entry = ttk.Entry(cfg, textvariable=self.host_var, width=15)
        self.host_entry.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(cfg, text="Port:").grid(row=0, column=2, sticky=tk.W)
        self.port_var = tk.StringVar(value=str(self.UDP_PORT))
        self.port_entry = ttk.Entry(cfg, textvariable=self.port_var, width=7)
        self.port_entry.grid(row=0, column=3, padx=(4, 12))

        ttk.Label(cfg, text="Send rate (Hz):").grid(row=0, column=4, sticky=tk.W)
        self.rate_var = tk.StringVar(value=f"{self.UDP_SEND_RATE_HZ:.1f}")
        self.rate_entry = ttk.Entry(cfg, textvariable=self.rate_var, width=7)
        self.rate_entry.grid(row=0, column=5, padx=(4, 12))

        # Buttons INSIDE UDP Config, directly below inputs
        btns = ttk.Frame(cfg)
        btns.grid(row=1, column=0, columnspan=6, pady=(8, 0), sticky=tk.W)

        self.connect_btn = ttk.Button(btns, text="Connect", command=self._toggle_connect)
        self.stream_btn = ttk.Button(btns, text="Start Streaming", command=self._toggle_streaming, state=tk.DISABLED)
        self.reload_btn = ttk.Button(btns, text="Reload", command=self._trigger_reload, state=tk.DISABLED)
        self.copy_btn = ttk.Button(btns, text="Copy Position", command=self._copy_command)
        self.reset_btn = ttk.Button(btns, text="Reset (0.6, 0.0, 0°)", command=self._reset_position)

        self.connect_btn.grid(row=0, column=0, padx=(0, 4))
        self.stream_btn.grid(row=0, column=1, padx=(0, 4))
        self.reload_btn.grid(row=0, column=2, padx=(0, 4))
        self.copy_btn.grid(row=0, column=3, padx=(0, 4))
        self.reset_btn.grid(row=0, column=4, padx=(0, 4))

        # UDP info
        udp = ttk.LabelFrame(main, text="UDP Streaming Control", padding="8")
        udp.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        top = ttk.Frame(udp)
        top.grid(row=0, column=0, columnspan=3, pady=(0, 6))
        self.target_label = ttk.Label(
            top, text=f"Target: {self.UDP_HOST}:{self.UDP_PORT}", font=('Courier', 9)
        )
        self.target_label.grid(row=0, column=0, sticky=tk.W)
        self.rate_label = ttk.Label(
            top, text=f"Target Rate: {self.UDP_SEND_RATE_HZ:.1f} Hz", font=('Courier', 9)
        )
        self.rate_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))

        stat = ttk.Frame(udp)
        stat.grid(row=1, column=0, sticky=tk.W)
        self.status_label = ttk.Label(stat, text="Status: Disconnected", font=('Courier', 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        self.hz_label = ttk.Label(stat, text="Send Hz: 0.0", font=('Courier', 10))
        self.hz_label.grid(row=0, column=1, sticky=tk.W, padx=(12, 0))

        self.recv_label = ttk.Label(stat, text="Recv: None", font=('Courier', 10))
        self.recv_label.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        # Hint label
        hint = ttk.Label(
            main,
            text="Keyboard: A/D=X−/X+, W/S=Z+/Z−, Q/E=Y−/Y+, R/F=Rot+, Rot−, Shift=fast",
            font=('Courier', 9),
        )
        hint.grid(row=6, column=0, columnspan=3, pady=(6, 0), sticky=tk.W)

        # Resizing behavior
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        for c in range(3):
            main.columnconfigure(c, weight=1)

    def _on_canvas_resize(self, event):
        """Resize the Matplotlib figure when the Tk canvas changes size."""
        try:
            dpi = self.fig.get_dpi()
            w = max(event.width / dpi, 1)
            h = max(event.height / dpi, 1)
            self.fig.set_size_inches(w, h, forward=False)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception:
            pass

    def _bind_events(self):
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.bind('<KeyRelease>', self._on_key_release)

    # -------------- 3D Drawing --------------
    def _init_plot(self):
        self.ax.view_init(elev=25, azim=-60)
        self._set_axes_limits()
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _set_axes_limits(self):
        xl = (self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['x_max'])
        yl = (self.WORKSPACE_LIMITS['y_min'], self.WORKSPACE_LIMITS['y_max'])
        zl = (self.WORKSPACE_LIMITS['z_min'], self.WORKSPACE_LIMITS['z_max'])
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)
        self.ax.set_zlim(zl)

        # Try to keep aspect roughly equal
        max_range = max(xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0])
        x_mid = sum(xl) / 2
        y_mid = sum(yl) / 2
        z_mid = sum(zl) / 2
        r = max_range / 2
        self.ax.set_xlim(x_mid - r, x_mid + r)
        self.ax.set_ylim(y_mid - r, y_mid + r)
        self.ax.set_zlim(z_mid - r, z_mid + r)

    def _draw_workspace_box(self):
        x0, x1 = self.WORKSPACE_LIMITS['x_min'], self.WORKSPACE_LIMITS['x_max']
        y0, y1 = self.WORKSPACE_LIMITS['y_min'], self.WORKSPACE_LIMITS['y_max']
        z0, z1 = self.WORKSPACE_LIMITS['z_min'], self.WORKSPACE_LIMITS['z_max']
        edges = [
            ((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)),
            ((x0, y0, z1), (x1, y0, z1)), ((x0, y1, z1), (x1, y1, z1)),
            ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)),
            ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
            ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
            ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1)),
        ]
        lc = Line3DCollection(edges, colors='gray', linewidths=1.0, linestyles='dashed')
        self.ax.add_collection3d(lc)

    def _draw_robot(self):
        # Draw the robot chain from joint_positions if available
        # joint_positions contains: [boom_mount, arm_mount, bucket_mount, tool_mount, ee_actual]
        if self.joint_positions and len(self.joint_positions) >= 2:
            # Draw origin offset (from world origin to first joint) in GREEN
            first_joint = self.joint_positions[0]
            origin_offset_mag = np.linalg.norm(first_joint)
            if origin_offset_mag > 0.001:  # Only draw if offset is non-zero
                self.ax.plot([0, first_joint[0]],
                           [0, first_joint[1]],
                           [0, first_joint[2]],
                           '-', color='green', linewidth=2.5, label='Origin offset')
                self.ax.scatter([0], [0], [0], c='green', s=30, marker='s')  # World origin

            # Draw joint links (blue) - only the first 4 joints (not the ee)
            num_joints = min(4, len(self.joint_positions) - 1) if len(self.joint_positions) == 5 else len(self.joint_positions)
            xs = [p[0] for p in self.joint_positions[:num_joints]]
            ys = [p[1] for p in self.joint_positions[:num_joints]]
            zs = [p[2] for p in self.joint_positions[:num_joints]]
            self.ax.plot(xs, ys, zs, '-o', color='blue', linewidth=2, markersize=4, label='Links')

            # Draw end-effector offset (from last joint to EE) in ORANGE
            if len(self.joint_positions) == 5:  # We have both joint positions and EE
                last_joint = self.joint_positions[3]  # tool_mount (last physical joint)
                ee_pos = self.joint_positions[4]      # ee_actual (with ee_offset)
                ee_offset_mag = np.linalg.norm(np.array(ee_pos) - np.array(last_joint))
                if ee_offset_mag > 0.001:  # Only draw if offset is non-zero
                    self.ax.plot([last_joint[0], ee_pos[0]],
                               [last_joint[1], ee_pos[1]],
                               [last_joint[2], ee_pos[2]],
                               '-', color='orange', linewidth=2.5, label='EE offset')
                    self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], c='orange', s=40, marker='D')  # EE point

    def _draw_target(self):
        """
        Draw target as a small arrow showing *pitch* in the X–Z plane.

        current_rot_y is treated as pitch angle:
            0°  = horizontal
            −   = nose-up (toward +Z)
            +   = nose-down (toward −Z)
        """
        x, y, z = self.current_x, self.current_y, self.current_z
        pitch = -math.radians(self.current_rot_y)  # Negated: negative values point up

        dx = 0.05 * math.cos(pitch)
        dz = 0.05 * math.sin(pitch)
        dy = 0.0

        self.ax.scatter([x], [y], [z], c='red', s=40, label='Target')
        self.ax.quiver(
            x, y, z,
            dx, dy, dz,
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.2,
            color='red',
        )

    def _request_redraw(self):
        """Request a scene redraw, throttled to a maximum rate."""
        now = time.time()
        if now - self._last_redraw_time >= self._redraw_interval:
            self._last_redraw_time = now
            self.root.after_idle(self._redraw_scene)

    def _redraw_scene(self):
        """Redraw the 3D scene while preserving user camera and zoom."""
        try:
            elev = self.ax.elev
            azim = self.ax.azim
            dist = self.ax.dist
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()
        except Exception:
            elev, azim, dist = 25, -60, 10
            xlim = ylim = zlim = None

        self.ax.cla()

        # Restore previous limits so pan/zoom is preserved; fall back to workspace box
        if xlim is not None and ylim is not None and zlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)
        else:
            self._set_axes_limits()

        self._draw_workspace_box()
        self._draw_robot()
        self._draw_target()

        self.ax.view_init(elev=elev, azim=azim)
        try:
            self.ax.dist = dist
        except Exception:
            pass

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')

        # Add legend to show different components
        try:
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        except Exception:
            pass  # Legend may fail if no labeled elements yet

        self.canvas.draw_idle()
        self._update_labels()

    # -------------- Keyboard control --------------
    def _on_key_press(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l', 'shift_r'):
            self.keys_down.add('shift')
        else:
            self.keys_down.add(key)

    def _on_key_release(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l', 'shift_r'):
            self.keys_down.discard('shift')
        else:
            self.keys_down.discard(key)

    def _schedule_tick(self):
        self._tick()
        self.root.after(33, self._schedule_tick)  # ~30 Hz

    def _tick(self):
        mult = self.FAST_MULTIPLIER if 'shift' in self.keys_down else 1.0
        dp = self.step_pos * mult
        dr = self.step_rot * mult

        moved = False
        if 'a' in self.keys_down:
            self.current_x -= dp; moved = True
        if 'd' in self.keys_down:
            self.current_x += dp; moved = True
        if 'w' in self.keys_down:
            self.current_z += dp; moved = True
        if 's' in self.keys_down:
            self.current_z -= dp; moved = True
        if 'q' in self.keys_down:
            self.current_y -= dp; moved = True
        if 'e' in self.keys_down:
            self.current_y += dp; moved = True
        if 'r' in self.keys_down:
            self.current_rot_y += dr; moved = True
        if 'f' in self.keys_down:
            self.current_rot_y -= dr; moved = True

        if moved:
            self._clamp_pose()
            self._request_redraw()

    def _clamp_pose(self):
        self.current_x = max(self.WORKSPACE_LIMITS['x_min'], min(self.current_x, self.WORKSPACE_LIMITS['x_max']))
        self.current_y = max(self.WORKSPACE_LIMITS['y_min'], min(self.current_y, self.WORKSPACE_LIMITS['y_max']))
        self.current_z = max(self.WORKSPACE_LIMITS['z_min'], min(self.current_z, self.WORKSPACE_LIMITS['z_max']))
        # Clamp rotation to [-45, 45] degrees (0 = horizon)
        if self.current_rot_y > 45.0:
            self.current_rot_y = 45.0
        elif self.current_rot_y < -45.0:
            self.current_rot_y = -45.0

    def _update_labels(self):
        self.x_label.config(text=f"X: {self.current_x:6.3f} m")
        self.y_label.config(text=f"Y: {self.current_y:6.3f} m")
        self.z_label.config(text=f"Z: {self.current_z:6.3f} m")
        self.rot_label.config(text=f"Rot Y: {self.current_rot_y:6.1f}°")

    def _apply_pos_step(self):
        try:
            val = float(self.pos_step_var.get())
            if val <= 0:
                raise ValueError
            self.step_pos = val
        except ValueError:
            messagebox.showerror("Invalid value", "Move step must be a positive number.")
            self.pos_step_var.set(f"{self.step_pos:.3f}")

    def _apply_rot_step(self):
        try:
            val = float(self.rot_step_var.get())
            if val <= 0:
                raise ValueError
            self.step_rot = val
        except ValueError:
            messagebox.showerror("Invalid value", "Rot step must be a positive number.")
            self.rot_step_var.set(f"{self.step_rot:.1f}")

    def _copy_command(self):
        cmd = f"{self.current_x:.3f}, {self.current_y:.3f}, {self.current_z:.3f}, {self.current_rot_y:.1f}"
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.root.update()
        messagebox.showinfo("Copied", f"Copied pose: {cmd}")

    def _reset_position(self):
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0
        self._clamp_pose()
        self._request_redraw()

    # -------------- Encoding helpers (must match excv_gui_log.py) --------------
    def _encode_pose_to_bytes(self, x: float, y: float, z: float, rot_deg: float, reload_flag: int) -> List[int]:
        """
        Encode position + rotation + reload_flag as 9 signed bytes:

        [x_high, x_low, y_high, y_low, z_high, z_low, rot_high, rot_low, reload_byte]

        Where x,y,z are in meters, rot_deg in degrees.
        SCALE_POS and SCALE_ROT match the server.
        reload_flag is 0 or 1, mapped to 0 or 127.
        """
        def clamp_int16(v: int) -> int:
            return max(-32768, min(32767, v))

        # Scale to int16
        x_int = clamp_int16(int(round(x * self.SCALE_POS)))
        y_int = clamp_int16(int(round(y * self.SCALE_POS)))
        z_int = clamp_int16(int(round(z * self.SCALE_POS)))
        rot_int = clamp_int16(int(round(rot_deg * self.SCALE_ROT)))

        def split_int16(value: int) -> (int, int):
            if value < 0:
                value = (1 << 16) + value  # two's complement
            high = (value >> 8) & 0xFF
            low = value & 0xFF
            return high, low

        def to_signed_byte(b: int) -> int:
            return b if b < 128 else b - 256

        x_high, x_low = split_int16(x_int)
        y_high, y_low = split_int16(y_int)
        z_high, z_low = split_int16(z_int)
        rot_high, rot_low = split_int16(rot_int)

        reload_byte = 127 if reload_flag else 0  # maps to ~1.0 or 0.0 at receiver

        return [
            to_signed_byte(x_high), to_signed_byte(x_low),
            to_signed_byte(y_high), to_signed_byte(y_low),
            to_signed_byte(z_high), to_signed_byte(z_low),
            to_signed_byte(rot_high), to_signed_byte(rot_low),
            to_signed_byte(reload_byte),
        ]

    def _decode_joint_positions_from_bytes(self, data: List[int]) -> Optional[List[List[float]]]:
        """
        Decode 30 signed bytes to 5 positions [ [x,y,z], ... ] in meters.

        Must match encode_joint_positions_to_bytes in excv_gui_log.py.
        """
        if len(data) != self.UDP_NUM_BYTES_RECV:
            return None

        positions: List[List[float]] = []
        SCALE = self.SCALE_POS

        for i in range(0, len(data), 6):
            xh, xl, yh, yl, zh, zl = data[i:i + 6]

            def to_unsigned(b: int) -> int:
                return b if b >= 0 else b + 256

            x_int = (to_unsigned(xh) << 8) | to_unsigned(xl)
            y_int = (to_unsigned(yh) << 8) | to_unsigned(yl)
            z_int = (to_unsigned(zh) << 8) | to_unsigned(zl)

            if x_int >= 32768:
                x_int -= 65536
            if y_int >= 32768:
                y_int -= 65536
            if z_int >= 32768:
                z_int -= 65536

            x = x_int / SCALE
            y = y_int / SCALE
            z = z_int / SCALE
            positions.append([x, y, z])

        return positions

    # -------------- UDP / Streaming --------------
    def _toggle_connect(self):
        if not self.udp_connected:
            self._connect_udp()
        else:
            self._disconnect_udp()

    def _connect_udp(self):
        if self.udp_connected:
            return
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get().strip())
            rate = float(self.rate_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid config", "Host/port/rate settings are invalid.")
            return

        try:
            self.udp = UDPSocket(local_id=self.UDP_LOCAL_ID, max_age_seconds=0.5)
            self.udp.setup(host, port,
                           num_inputs=self.UDP_NUM_BYTES_RECV,
                           num_outputs=self.UDP_NUM_BYTES_SEND,
                           is_server=False)
            if not self.udp.handshake(timeout=10.0):
                messagebox.showerror("Handshake failed", "Failed to complete UDP handshake with server.")
                self.udp.close()
                self.udp = None
                return
        except Exception as e:
            messagebox.showerror("Connection error", f"Failed to create UDP socket: {e}")
            self.udp = None
            return

        self.udp_connected = True
        self.UDP_SEND_RATE_HZ = rate
        self.connect_btn.config(text="Disconnect")
        self.stream_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Connected to {host}:{port}")
        self.target_label.config(text=f"Target: {host}:{port}")

    def _disconnect_udp(self):
        if not self.udp_connected:
            return
        self._stop_streaming()
        if self.udp:
            self.udp.close()
        self.udp = None
        self.udp_connected = False
        self.connect_btn.config(text="Connect")
        self.stream_btn.config(state=tk.DISABLED)
        self.reload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Disconnected")

    def _toggle_streaming(self):
        if not self.udp_streaming:
            self._start_streaming()
        else:
            self._stop_streaming()

    def _start_streaming(self):
        if not self.udp_connected or not self.udp:
            messagebox.showwarning("Not connected", "Connect to the UDP server first.")
            return

        print(f"\nStarting UDP streaming at {self.UDP_SEND_RATE_HZ:.1f} Hz...")
        self.udp_streaming = True
        self.send_thread_running = True
        self.send_time_window_start = time.perf_counter()
        self.send_count_window = 0
        self.udp.start_receiving()

        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.send_thread.start()
        self.recv_thread.start()

        self.stream_btn.config(text="Stop Streaming")
        self.reload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Streaming")

    def _stop_streaming(self):
        if not self.udp_streaming:
            return
        print("\nStopping UDP streaming...")
        self.udp_streaming = False
        self.send_thread_running = False
        if self.udp:
            self.udp.stop_receiving()
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
        self.stream_btn.config(text="Start Streaming")
        self.reload_btn.config(state=tk.DISABLED)
        self.send_hz_actual = 0.0
        self.hz_label.config(text="Send Hz: 0.0")
        if self.udp_connected:
            self.status_label.config(text="Status: Connected (Idle)")
        else:
            self.status_label.config(text="Status: Disconnected")
        print("Streaming stopped")

    def _receive_loop(self):
        while self.send_thread_running:
            try:
                if not self.udp:
                    time.sleep(0.05)
                    continue
                data = self.udp.get_latest()
                if data and len(data) == self.UDP_NUM_BYTES_RECV:
                    joints = self._decode_joint_positions_from_bytes(data)
                    if joints is not None:
                        self.joint_positions = joints
                        self.root.after(0, self._request_redraw)
                        self.recv_label.config(text="Recv: OK")
                else:
                    # no fresh data
                    pass
                time.sleep(0.01)
            except Exception as e:
                print(f"Receive error: {e}")
                time.sleep(0.1)

    def _send_loop(self):
        if self.UDP_SEND_RATE_HZ <= 0:
            self.UDP_SEND_RATE_HZ = 20.0
        period = 1.0 / self.UDP_SEND_RATE_HZ
        next_t = time.perf_counter()
        self.send_time_window_start = time.perf_counter()
        self.send_count_window = 0

        while self.send_thread_running:
            try:
                if self.udp:
                    payload = self._encode_pose_to_bytes(
                        self.current_x, self.current_y, self.current_z,
                        self.current_rot_y, self.reload_flag
                    )
                    self.udp.send(payload)
                    self.send_count_window += 1

                    # reset reload flag after one send (edge)
                    if self.reload_flag:
                        self.reload_flag = 0

                    now = time.perf_counter()
                    if now - self.send_time_window_start >= self.HZ_UPDATE_INTERVAL:
                        self.send_hz_actual = self.send_count_window / (now - self.send_time_window_start)
                        self.send_time_window_start = now
                        self.send_count_window = 0
                        self.root.after(0, self._update_send_hz_label)

                next_t += period
                remaining = next_t - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_t = time.perf_counter()
            except Exception as e:
                print(f"Send error: {e}")
                time.sleep(0.1)

    def _update_send_hz_label(self):
        self.hz_label.config(text=f"Send Hz: {self.send_hz_actual:4.1f}")

    def _trigger_reload(self):
        print("Reload requested")
        self.reload_flag = 1

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
