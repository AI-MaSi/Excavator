"""
Client-side PID tuner GUI (20 Hz) to adjust setpoint and PID gains.

Uses modules.udp_socket.UDPSocket to communicate with robot-side tuner.

Controls:
- Joint: dropdown [slew, boom, arm, bucket]
- Setpoint (deg): entry box, accepts floats
- PID gains: sliders (kp, ki, kd)

Plots (lightweight): simple Tkinter canvas tracing target vs measured angle over time.

Protocol matches pid_tuner_robot.py.
"""

import math
import time
import tkinter as tk
from tkinter import ttk
from collections import deque

from modules.udp_socket import UDPSocket


def rad_to_norm(angle_rad: float) -> float:
    return max(-1.0, min(1.0, angle_rad / math.pi))


def norm_to_rad(n: float) -> float:
    return max(-1.0, min(1.0, n)) * math.pi


def joint_id_to_norm(jid: int) -> float:
    # Inverse of joint_norm_to_id in robot script
    jid = max(0, min(3, int(jid)))
    return (jid / 3.0) * 2.0 - 1.0


class PIDTunerClient:
    def __init__(self, host: str = "192.168.0.132", port: int = 8080):
        # UDP client (deferred; connect via UI button)
        self.sock = None
        self.default_host = host
        self.default_port = port

        # Tk root must be created before Tk variables
        self.root = tk.Tk()
        self.root.title("PID Tuner Client")

        # UI state
        self.joint_names = ['slew', 'boom', 'arm', 'bucket']
        self.selected_joint = tk.IntVar(self.root, value=0)
        # Setpoint slider (-90..90 deg)
        self.setpoint_deg_var = tk.DoubleVar(self.root, value=0.0)
        # PID controls via sliders with numeric readouts
        self.kp_var = tk.DoubleVar(self.root, value=2.0)
        self.ki_var = tk.DoubleVar(self.root, value=0.0)
        self.kd_var = tk.DoubleVar(self.root, value=0.0)
        self.kp_disp = tk.StringVar(self.root, value=f"{self.kp_var.get():.3f}")
        self.ki_disp = tk.StringVar(self.root, value=f"{self.ki_var.get():.3f}")
        self.kd_disp = tk.StringVar(self.root, value=f"{self.kd_var.get():.3f}")

        # Latched target setpoint (normalized). None until synced from feedback or send pressed.
        self.current_target_norm = None
        # Keep the last latched setpoint in degrees for plotting
        self.latched_setpoint_deg = None

        # Plot buffers (degrees)
        self.max_points = 600  # keep ~30s at 20 Hz
        self.target_hist = deque(maxlen=self.max_points)
        self.meas_hist = deque(maxlen=self.max_points)

        # Fixed plot range (deg). Hardcode as needed.
        self.plot_ymin = -90.0
        self.plot_ymax = 90

        # Snapshot settings: provide explicit Snapshot button to save current view

        # Build UI after variables
        self._build_ui()

        # 20 Hz update
        self.period_ms = 50
        self._schedule_update()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        # Connection controls
        ttk.Label(frm, text="Host:").grid(row=0, column=0, sticky="w")
        self.host_var = tk.StringVar(self.root, value=self.default_host)
        ttk.Entry(frm, textvariable=self.host_var, width=14).grid(row=0, column=1, sticky="ew")
        ttk.Label(frm, text="Port:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.port_var = tk.StringVar(self.root, value=str(self.default_port))
        ttk.Entry(frm, textvariable=self.port_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Button(frm, text="Connect", command=self._on_connect_clicked).grid(row=0, column=4, sticky="w", padx=(8, 0))

        # Joint dropdown
        ttk.Label(frm, text="Joint:").grid(row=1, column=0, sticky="w")
        joint_cb = ttk.Combobox(frm, values=self.joint_names, state="readonly")
        joint_cb.current(0)
        joint_cb.grid(row=1, column=1, sticky="ew")
        joint_cb.bind("<<ComboboxSelected>>", lambda e: self._on_joint_changed(joint_cb.current()))

        # Setpoint slider (deg)
        ttk.Label(frm, text="Setpoint (deg):").grid(row=1, column=2, sticky="w")
        ttk.Scale(frm, from_=-90.0, to=90.0, variable=self.setpoint_deg_var,
                  orient=tk.HORIZONTAL).grid(row=1, column=3, sticky="ew")
        self.sp_disp = tk.StringVar(self.root, value=f"{self.setpoint_deg_var.get():.1f}")
        ttk.Label(frm, textvariable=self.sp_disp, width=8).grid(row=1, column=4, sticky="w")
        # keep display synced
        def _on_sp_change(v=None):
            try:
                self.sp_disp.set(f"{float(self.setpoint_deg_var.get()):.1f}")
            except Exception:
                pass
        self.setpoint_deg_var.trace_add('write', lambda *args: _on_sp_change())

        # PID sliders with numeric display
        ttk.Label(frm, text="kp").grid(row=2, column=0, sticky="w")
        ttk.Scale(frm, from_=0.0, to=100.0, variable=self.kp_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.kp_disp.set(f"{float(v):.3f}"))\
            .grid(row=2, column=1, sticky="ew")
        ttk.Label(frm, textvariable=self.kp_disp, width=8).grid(row=2, column=2, sticky="w")

        ttk.Label(frm, text="ki").grid(row=3, column=0, sticky="w")
        ttk.Scale(frm, from_=0.0, to=50.0, variable=self.ki_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.ki_disp.set(f"{float(v):.3f}"))\
            .grid(row=3, column=1, sticky="ew")
        ttk.Label(frm, textvariable=self.ki_disp, width=8).grid(row=3, column=2, sticky="w")

        ttk.Label(frm, text="kd").grid(row=4, column=0, sticky="w")
        ttk.Scale(frm, from_=0.0, to=50.0, variable=self.kd_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.kd_disp.set(f"{float(v):.3f}"))\
            .grid(row=4, column=1, sticky="ew")
        ttk.Label(frm, textvariable=self.kd_disp, width=8).grid(row=4, column=2, sticky="w")

        # Canvas plot (larger for better visibility)
        self.canvas = tk.Canvas(frm, width=1200, height=400, bg="white")
        self.canvas.grid(row=5, column=0, columnspan=5, pady=(10, 0))

        # Status labels
        self.status_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.status_var).grid(row=6, column=0, columnspan=5, sticky="w", pady=(6, 0))

        # Controls row for send/sync
        ctrl_row = ttk.Frame(frm)
        ctrl_row.grid(row=7, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(6, 0))
        ttk.Button(ctrl_row, text="Send", command=self._on_send_clicked).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(ctrl_row, text="Sync to Measured", command=self._on_sync_clicked).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(ctrl_row, text="Snapshot", command=self._on_save_snapshot).pack(side=tk.LEFT)

        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _schedule_update(self):
        self.root.after(self.period_ms, self._update_loop)

    def _update_loop(self):
        # Read setpoint slider but do not apply unless user presses Send
        try:
            setpoint_deg = float(self.setpoint_deg_var.get())
        except Exception:
            setpoint_deg = 0.0
        kp = float(self.kp_var.get())
        ki = float(self.ki_var.get())
        kd = float(self.kd_var.get())

        # Normalize gains to [-1,1] matching robot mapping
        def to_norm(x, lo, hi):
            return ((x - lo) / (hi - lo)) * 2.0 - 1.0

        kp_n = to_norm(kp, 0.0, 20.0)
        ki_n = to_norm(ki, 0.0, 5.0)
        kd_n = to_norm(kd, 0.0, 5.0)
        joint_n = joint_id_to_norm(self.selected_joint.get())

        # If we have not yet latched a target, try to sync from measured feedback
        latest = None
        if self.sock is not None:
            latest = self.sock.get_latest_floats()
        measured_deg = None
        pwm = None
        if latest and len(latest) >= 2:
            measured_deg = float(latest[0]) * 180.0
            pwm = float(latest[1])

        # Only send once target is latched
        if self.current_target_norm is not None and self.sock is not None:
            try:
                self.sock.send_floats([self.current_target_norm, kp_n, ki_n, kd_n, joint_n])
            except Exception:
                pass

        # Receive latest feedback
        # measured_deg and pwm already decoded above

        # Update history and canvas
        # Append the last latched setpoint to avoid plot changes while typing
        if self.latched_setpoint_deg is not None:
            self.target_hist.append(self.latched_setpoint_deg)
        else:
            # If nothing latched yet, mirror measured to keep lines aligned
            self.target_hist.append(measured_deg if measured_deg is not None else 0.0)
        self.meas_hist.append(measured_deg if measured_deg is not None else 0.0)
        self._redraw_plot()

        # Status with connection state
        conn_txt = "connected" if self.sock is not None else "disconnected"
        self.status_var.set(
            f"[{conn_txt}] Joint={self.joint_names[self.selected_joint.get()]}  Setpoint={setpoint_deg:.1f}deg  "
            + (f"Meas={measured_deg:.1f}deg  PWM={pwm:.2f}" if measured_deg is not None else "Meas=--  PWM=--")
        )

        self._schedule_update()

    def _on_send_clicked(self):
        # Latch current entry as target
        try:
            setpoint_deg = float(self.setpoint_deg_var.get())
        except Exception:
            setpoint_deg = 0.0
        setpoint_rad = math.radians(setpoint_deg)
        self.current_target_norm = max(-1.0, min(1.0, setpoint_rad / math.pi))
        self.latched_setpoint_deg = setpoint_deg
        # No auto-snapshot on send; use Snapshot button

    def _on_sync_clicked(self):
        if self.sock is None:
            return
        latest = self.sock.get_latest_floats()
        if latest and len(latest) >= 2:
            measured_deg = float(latest[0]) * 180.0
            self.setpoint_deg_var.set(f"{measured_deg:.2f}")
            self.current_target_norm = max(-1.0, min(1.0, measured_deg / 180.0))
            self.latched_setpoint_deg = measured_deg

    def _on_joint_changed(self, idx: int):
        # Update selected joint and reset pose to measured to avoid jumps.
        self.selected_joint.set(idx)
        # Clear any latched target; do not send until user presses Send/Sync
        self.current_target_norm = None
        self.latched_setpoint_deg = None
        # Clear plot histories when switching channels
        self.target_hist.clear()
        self.meas_hist.clear()
        self._redraw_plot()
        # Optionally sync UI to current measured if available
        if self.sock is not None:
            latest = self.sock.get_latest_floats()
            if latest and len(latest) >= 2:
                measured_deg = float(latest[0]) * 180.0
                self.setpoint_deg_var.set(f"{measured_deg:.2f}")

    def _on_connect_clicked(self):
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get())
        except Exception:
            port = self.default_port
            self.port_var.set(str(port))

        try:
            sock = UDPSocket(local_id=1, max_age_seconds=0.5)
            sock.setup(host=host, port=port, num_inputs=2, num_outputs=5, is_server=False)
            sock.handshake(timeout=5.0)
            sock.start_receiving()
            self.sock = sock
            self.status_var.set(f"[connected] Host={host} Port={port}")
        except Exception as e:
            self.status_var.set(f"[error] connect failed: {e}")

    def _on_save_snapshot(self):
        # Use full current view buffers
        t_series = list(self.target_hist)
        m_series = list(self.meas_hist)

        # Save via matplotlib to a simple PNG (no fallback)
        try:
            import matplotlib.pyplot as plt  # type: ignore

            dt = self.period_ms / 1000.0
            n = len(t_series)
            # Use a recent-time axis that matches live view window
            # Right edge is 'now' at 0s, left is negative seconds back in time
            x = [-(n - 1 - i) * dt for i in range(n)]
            # Match snapshot size to current canvas dimensions (approximate)
            try:
                w_px = int(self.canvas.winfo_width())
                h_px = int(self.canvas.winfo_height())
            except Exception:
                w_px, h_px = 1200, 400
            dpi = 100
            fig_w = max(3.0, w_px / dpi)
            fig_h = max(2.0, h_px / dpi)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.plot(x, t_series, label='target', color='#1f77b4')
            ax.plot(x, m_series, label='measured', color='#d62728')
            ax.set_ylim(self.plot_ymin, self.plot_ymax)
            # Show same-ish timeframe as live view (recent window only)
            ax.set_xlim(min(x), 0.0)
            ax.set_xlabel('time (s)')
            ax.set_ylabel('deg')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Annotate with joint + PID + setpoint
            jname = self.joint_names[self.selected_joint.get()]
            sp = self.latched_setpoint_deg if self.latched_setpoint_deg is not None else float(self.setpoint_deg_var.get())
            title = f"Joint: {jname} | kp={self.kp_var.get():.3f} ki={self.ki_var.get():.3f} kd={self.kd_var.get():.3f} | setpoint={sp:.1f} deg"
            fig.suptitle(title)

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            png_path = f"snapshot_{timestamp}.png"
            fig.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
            self.status_var.set(f"[saved] {png_path}")
            return
        except Exception as e:
            self.status_var.set(f"[error] matplotlib required to save PNG: {e}. Try: pip install matplotlib")
            return

    def _redraw_plot(self):
        self.canvas.delete("all")
        self._draw_series(self.canvas, list(self.target_hist), list(self.meas_hist), self.plot_ymin, self.plot_ymax)

    def _draw_series(self, canvas: tk.Canvas, target_series, meas_series, ymin: float, ymax: float):
        w = int(canvas['width'])
        h = int(canvas['height'])

        # Axes
        canvas.create_line(40, h - 20, w - 10, h - 20, fill="#888")
        canvas.create_line(40, 10, 40, h - 20, fill="#888")

        # Avoid zero range
        if abs(ymax - ymin) < 1e-6:
            ymin -= 1.0
            ymax += 1.0

        def xy(i, val, npts):
            x = 40 + (i / max(1, npts - 1)) * (w - 50)
            y_norm = (val - ymin) / (ymax - ymin)
            y = (1.0 - y_norm) * (h - 30) + 10
            return x, y

        for series, color in ((target_series, "#1f77b4"), (meas_series, "#d62728")):
            points = []
            npts = len(series)
            for i, v in enumerate(series):
                points.extend(xy(i, v, npts))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2)

        # Show fixed min/max on axis
        canvas.create_text(42, 10, anchor="nw", text=f"{ymax:.1f}°", fill="#444")
        canvas.create_text(42, h - 20, anchor="sw", text=f"{ymin:.1f}°", fill="#444")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    import sys
    host = "192.168.0.132"
    port = 8080
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            port = int(sys.argv[2])
        except ValueError:
            pass
    PIDTunerClient(host=host, port=port).run()
