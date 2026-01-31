import tkinter as tk
from tkinter import ttk, messagebox
from modules.udp_socket import UDPSocket
import threading


class ControllerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Controller")
        self.root.resizable(False, False)

        self.axis_min = -128
        self.axis_max = 127
        self.button_on = 127

        self.client = None
        self.connected = False
        self.sending = False

        # Variables for inputs
        self.button_vars = [tk.IntVar(value=0) for _ in range(4)]
        self.axis_vars = [tk.IntVar(value=0) for _ in range(6)]
        self.axis_entry_vars = [tk.StringVar(value="0") for _ in range(6)]
        self.button_pulses = set()
        self.axis_scales = []
        self.pulse_override = None
        self.pulse_until = 0.0

        self.build_ui()

    def build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_tab = ttk.Frame(self.notebook, padding=5)
        self.tester_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(main_tab, text="Manual Control")
        self.notebook.add(self.tester_tab, text="Pulse Tester")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Connection frame
        conn_frame = ttk.LabelFrame(main_tab, text="Connection", padding=10)
        conn_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, sticky="e")
        self.host_entry = ttk.Entry(conn_frame, width=15)
        self.host_entry.insert(0, "192.168.0.132")
        self.host_entry.grid(row=0, column=1, padx=5)

        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, sticky="e")
        self.port_entry = ttk.Entry(conn_frame, width=6)
        self.port_entry.insert(0, "8080")
        self.port_entry.grid(row=0, column=3, padx=5)

        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=10)

        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=5, padx=5)

        # Buttons frame
        btn_frame = ttk.LabelFrame(main_tab, text="Commands", padding=10)
        btn_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        button_entries = [
            ("Toggle Pump (Button 2)", 2),
            ("Reload Config (Button 3)", 3),
        ]

        for row, (label, idx) in enumerate(button_entries):
            ttk.Label(btn_frame, text=label).grid(row=row, column=0, sticky="e", pady=2)
            btn = tk.Button(btn_frame, text="Push", width=6)
            btn.grid(row=row, column=1, sticky="w", padx=5)
            self._bind_momentary_button(btn, self.button_vars[idx], idx)

        # Axis frame
        axis_frame = ttk.LabelFrame(main_tab, text="Axis Values (-128 to 127)", padding=10)
        axis_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

        axis_labels = [
            "Left Track",
            "Right Track",
            "Tilt",
            "Rotate",
            "Lift",
            "Scoop",
        ]

        for i, label in enumerate(axis_labels):
            ttk.Label(axis_frame, text=label).grid(row=i, column=0, sticky="e", pady=2)

            scale = ttk.Scale(
                axis_frame,
                from_=self.axis_min,
                to=self.axis_max,
                orient="horizontal",
                length=150,
                command=lambda val, idx=i: self._on_scale(idx, val),
            )
            scale.grid(row=i, column=1, padx=5)
            scale.set(self.axis_vars[i].get())
            self.axis_scales.append(scale)

            entry = ttk.Entry(axis_frame, textvariable=self.axis_entry_vars[i], width=5)
            entry.grid(row=i, column=2, padx=5)
            entry.bind("<Return>", lambda event, idx=i: self._commit_axis_entry(idx))

        # Control frame
        ctrl_frame = ttk.Frame(main_tab, padding=10)
        ctrl_frame.grid(row=2, column=0, columnspan=2, pady=5)

        self.send_btn = ttk.Button(ctrl_frame, text="Start Sending", command=self.toggle_sending, state="disabled")
        self.send_btn.grid(row=0, column=0, padx=5)

        ttk.Button(ctrl_frame, text="Reset All", command=self.reset_values).grid(row=0, column=1, padx=5)

        ttk.Label(ctrl_frame, text="Send Rate (ms):").grid(row=0, column=2, padx=5)
        self.rate_var = tk.IntVar(value=80)
        ttk.Entry(ctrl_frame, textvariable=self.rate_var, width=5).grid(row=0, column=3)

        self._build_tester_tab(self.tester_tab)

    def _build_tester_tab(self, parent):
        pulse_frame = ttk.LabelFrame(parent, text="Pulse Channels", padding=10)
        pulse_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(pulse_frame, text="Duration (s):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.pulse_duration_var = tk.StringVar(value="1.0")
        ttk.Entry(pulse_frame, textvariable=self.pulse_duration_var, width=6).grid(row=0, column=1, sticky="w")

        channel_labels = [
            "Button 0 (Left Top Red)",
            "Button 1 (Right Rocker U)",
            "Button 2 (Right Rocker D)",
            "Button 3",
            "Left Paddle",
            "Right Paddle",
            "Left U/D",
            "Left L/R",
            "Right U/D",
            "Right L/R",
        ]

        ttk.Label(pulse_frame, text="Channel").grid(row=1, column=0, padx=5, pady=2)
        ttk.Label(pulse_frame, text="-1").grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(pulse_frame, text="+1").grid(row=1, column=2, padx=5, pady=2)

        for i, label in enumerate(channel_labels):
            ttk.Label(pulse_frame, text=label).grid(row=i + 2, column=0, sticky="e", padx=5, pady=2)
            ttk.Button(
                pulse_frame,
                text="-1",
                width=4,
                command=lambda idx=i: self._start_pulse(idx, -1),
            ).grid(row=i + 2, column=1, padx=5, pady=2)
            ttk.Button(
                pulse_frame,
                text="+1",
                width=4,
                command=lambda idx=i: self._start_pulse(idx, 1),
            ).grid(row=i + 2, column=2, padx=5, pady=2)

    def toggle_connection(self):
        if not self.connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        try:
            host = self.host_entry.get()
            port = int(self.port_entry.get())

            self.client = UDPSocket(local_id=1, max_age_seconds=0.5)
            self.client.setup(host=host, port=port, num_inputs=0, num_outputs=10, is_server=False)

            self.status_label.config(text="Handshaking...", foreground="orange")
            self.root.update()

            if self.client.handshake(timeout=10.0):
                self.connected = True
                self.status_label.config(text="Connected", foreground="green")
                self.connect_btn.config(text="Disconnect")
                self.send_btn.config(state="normal")
            else:
                self.status_label.config(text="Handshake Failed", foreground="red")
                self.client = None

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.status_label.config(text="Error", foreground="red")

    def disconnect(self):
        self.sending = False
        self.connected = False
        self.client = None
        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_btn.config(text="Connect")
        self.send_btn.config(text="Start Sending", state="disabled")

    def toggle_sending(self):
        if not self.sending:
            self.sending = True
            self.send_btn.config(text="Stop Sending")
            threading.Thread(target=self.send_loop, daemon=True).start()
        else:
            self.sending = False
            self.send_btn.config(text="Start Sending")

    def _bind_momentary_button(self, button, var, idx):
        def press(_event):
            var.set(self.button_on)
            self.button_pulses.add(idx)
            button.config(relief="sunken")
            if self.connected and not self.sending:
                self._send_one_shot_button(idx)

        def release(_event):
            button.config(relief="raised")

        button.config(relief="raised")
        button.bind("<ButtonPress-1>", press)
        button.bind("<ButtonRelease-1>", release)
        button.bind("<Leave>", release)

    def _send_one_shot_button(self, idx):
        try:
            values = [0] * 10
            values[idx] = self.button_on
            success = self.client.send(values)
            if not success:
                self.root.after(0, self.disconnect)
                return
            self.client.send([0] * 10)
        except Exception as e:
            print(f"Send error: {e}")
            self.root.after(0, self.disconnect)

    def _on_scale(self, idx, value):
        try:
            ivalue = int(round(float(value)))
        except (TypeError, ValueError):
            return
        ivalue = max(self.axis_min, min(self.axis_max, ivalue))
        self.axis_vars[idx].set(ivalue)
        self.axis_entry_vars[idx].set(str(ivalue))

    def _commit_axis_entry(self, idx):
        text = self.axis_entry_vars[idx].get().strip()
        if text in ("", "-"):
            self.axis_entry_vars[idx].set(str(self.axis_vars[idx].get()))
            return
        try:
            value = int(float(text))
        except ValueError:
            self.axis_entry_vars[idx].set(str(self.axis_vars[idx].get()))
            return
        value = max(self.axis_min, min(self.axis_max, value))
        self.axis_vars[idx].set(value)
        self.axis_entry_vars[idx].set(str(value))
        self.axis_scales[idx].set(value)

    def _on_tab_changed(self, _event):
        if self.notebook.select() == str(self.tester_tab):
            self._zero_axes()

    def _zero_axes(self):
        for i in range(len(self.axis_vars)):
            self.axis_vars[i].set(0)
            self.axis_entry_vars[i].set("0")
            if i < len(self.axis_scales):
                self.axis_scales[i].set(0)

    def _start_pulse(self, channel_index, value):
        if not self.connected or self.client is None:
            messagebox.showerror("Not Connected", "Connect before sending pulses.")
            return
        try:
            duration = float(self.pulse_duration_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Enter a valid duration in seconds.")
            return
        if duration <= 0:
            messagebox.showerror("Invalid Duration", "Duration must be > 0.")
            return

        self.pulse_override = (channel_index, value)
        import time
        self.pulse_until = time.time() + duration
        if not self.sending:
            threading.Thread(target=self._pulse_loop, daemon=True).start()

    def _pulse_loop(self):
        import time

        interval = max(0.01, self.rate_var.get() / 1000.0)
        while time.time() <= self.pulse_until and self.connected and self.client:
            values = self._build_send_values()
            success = self.client.send(values)
            if not success:
                self.root.after(0, self.disconnect)
                return
            time.sleep(interval)

        if self.connected and self.client:
            self.client.send([0] * 10)

    def send_loop(self):
        while self.sending and self.connected:
            try:
                values = self._build_send_values()

                success = self.client.send(values)
                if not success:
                    self.root.after(0, self.disconnect)
                    break

                if self.button_pulses:
                    for idx in list(self.button_pulses):
                        self.button_vars[idx].set(0)
                    self.button_pulses.clear()

            except Exception as e:
                print(f"Send error: {e}")
                self.root.after(0, self.disconnect)
                break

            # Sleep for the configured rate
            import time
            time.sleep(self.rate_var.get() / 1000.0)

    def _apply_pulse_override(self, values):
        import time
        if self.pulse_override and time.time() <= self.pulse_until:
            idx, val = self.pulse_override
            if 0 <= idx < len(values):
                values[idx] = val
            return values
        if self.pulse_override and time.time() > self.pulse_until:
            self.pulse_override = None
        return values

    def _build_send_values(self):
        values = [
            self.button_vars[0].get(),
            self.button_vars[1].get(),
            self.button_vars[2].get(),
            self.button_vars[3].get(),
            self.axis_vars[0].get(),
            self.axis_vars[1].get(),
            self.axis_vars[2].get(),
            self.axis_vars[3].get(),
            self.axis_vars[4].get(),
            self.axis_vars[5].get(),
        ]
        return self._apply_pulse_override(values)

    def reset_values(self):
        for var in self.button_vars:
            var.set(0)
        self.button_pulses.clear()
        for var in self.axis_vars:
            var.set(0)
        for i, var in enumerate(self.axis_entry_vars):
            var.set("0")
            self.axis_scales[i].set(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ControllerGUI(root)
    root.mainloop()
