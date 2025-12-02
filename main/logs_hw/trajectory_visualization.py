#!/usr/bin/env python3
"""
Trajectory Visualizer for Excavator System

Simple trajectory visualization with planned (blue) and executed (red) paths.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.signal import savgol_filter


class TrajectoryVisualizer:
    def __init__(self, excavator_base_offset=0.00):
        """
        Initialize the trajectory visualizer.
        
        Args:
            excavator_base_offset (float): Height offset of excavator base from ground (meters)
        """
        self.excavator_base_offset = excavator_base_offset
        self.fig = None
        self.ax = None
        
    def filter_outliers_and_smooth(self, trajectory, max_step_distance=0.03, smooth_window=71, polyorder=2):
        """
        Remove outlier points and smooth trajectory to reduce sensor noise.

        Uses physically-justified outlier removal based on robot kinematics and sampling rate,
        followed by Savitzky-Golay filtering which preserves trajectory shape while reducing noise.

        Args:
            trajectory (np.array): Nx3 array of trajectory points
            max_step_distance (float): Maximum allowed distance between consecutive points (meters).
                                       Default 0.03m (3cm) based on robot kinematics and sampling rate.
            smooth_window (int): Window size for Savitzky-Golay filter (must be odd). Default 21.
                                 Aggressive smoothing to remove encoder oscillations and high-frequency noise.
            polyorder (int): Polynomial order for Savitzky-Golay filter. Default 2.

        Returns:
            np.array: Filtered and smoothed trajectory
        """
        if len(trajectory) < 2:
            return trajectory

        # Calculate step distances between consecutive points
        step_distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))

        # Find outlier indices (points with large jumps)
        outlier_mask = np.zeros(len(trajectory), dtype=bool)
        outlier_mask[1:] = step_distances > max_step_distance

        # Remove outliers
        filtered_traj = trajectory[~outlier_mask]
        num_removed = np.sum(outlier_mask)

        if num_removed > 0:
            print(f"  Removed {num_removed} outlier points (step > {max_step_distance}m)")

        # Apply Savitzky-Golay smoothing to each axis
        # This preserves trajectory shape better than moving average
        if len(filtered_traj) >= smooth_window:
            smoothed_traj = np.zeros_like(filtered_traj)
            for i in range(3):  # x, y, z
                smoothed_traj[:, i] = savgol_filter(filtered_traj[:, i],
                                                    window_length=smooth_window,
                                                    polyorder=polyorder,
                                                    mode='nearest')
            print(f"  Applied Savitzky-Golay filter (window={smooth_window}, polyorder={polyorder})")
            return smoothed_traj
        else:
            return filtered_traj

    def load_trajectory_data(self, filepath, apply_filtering=False):
        """
        Load trajectory data from CSV file.

        Args:
            filepath (str): Path to CSV file containing trajectory data
            apply_filtering (bool): If True, remove outliers and smooth data

        Returns:
            tuple: (planned_trajectory, executed_trajectory) as numpy arrays
        """
        try:
            df = pd.read_csv(filepath)

            # Validate required columns
            required_cols = ['x_g', 'y_g', 'z_g', 'x_e', 'y_e', 'z_e']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Extract planned trajectory (goal points)
            planned_trajectory = df[['x_g', 'y_g', 'z_g']].values

            # Extract executed trajectory
            executed_trajectory = df[['x_e', 'y_e', 'z_e']].values

            print(f"Loaded {len(planned_trajectory)} trajectory points")

            # Apply filtering if requested
            if apply_filtering:
                print("Applying filtering...")
                planned_trajectory = self.filter_outliers_and_smooth(planned_trajectory)
                executed_trajectory = self.filter_outliers_and_smooth(executed_trajectory)
                print(f"After filtering: {len(planned_trajectory)} points remaining")

            # Apply excavator base offset (move everything up by 0.15m)
            planned_trajectory[:, 2] += self.excavator_base_offset
            executed_trajectory[:, 2] += self.excavator_base_offset

            return planned_trajectory, executed_trajectory

        except Exception as e:
            print(f"Error loading trajectory data: {e}")
            sys.exit(1)
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quat (array-like): [w, x, y, z] quaternion
            
        Returns:
            np.array: 3x3 rotation matrix
        """
        w, x, y, z = quat
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def create_box_vertices(self, size, pos, rot):
        """
        Create vertices for a 3D box obstacle.
        
        Args:
            size (np.array): [x, y, z] dimensions
            pos (np.array): [x, y, z] center position
            rot (np.array): [w, x, y, z] quaternion rotation
            
        Returns:
            np.array: 8x3 array of box vertices
        """
        # Create box vertices centered at origin
        half_size = size / 2
        vertices = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]]
        ])
        
        # Apply rotation
        R = self.quaternion_to_rotation_matrix(rot)
        rotated_vertices = vertices @ R.T
        
        # Apply translation (no base offset for obstacles)
        world_vertices = rotated_vertices + pos
        
        return world_vertices
    
    def create_box_faces(self, vertices):
        """
        Create faces for a 3D box from vertices.
        
        Args:
            vertices (np.array): 8x3 array of box vertices
            
        Returns:
            list: List of face vertex arrays
        """
        # Define the 6 faces of a box (each face is a quad)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[5], vertices[6], vertices[2]]   # right
        ]
        
        return faces
    
    def plot_trajectories(self, planned_traj, executed_traj, obstacle_data=None, algorithm_name="A*"):
        """
        Create 3D visualization of trajectories and obstacles.
        
        Args:
            planned_traj (np.array): Planned trajectory points (Nx3)
            executed_traj (np.array): Executed trajectory points (Nx3)
            obstacle_data (list): List of obstacle dictionaries with 'size', 'pos', 'rot'
            algorithm_name (str): Name of the algorithm used
        """
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot planned trajectory (blue)
        self.ax.plot(planned_traj[:, 0], planned_traj[:, 1], planned_traj[:, 2], 
                    'b-', linewidth=3, label='Planned Trajectory', alpha=1.0, zorder=5)
        self.ax.scatter(planned_traj[0, 0], planned_traj[0, 1], planned_traj[0, 2], 
                       c='blue', s=120, marker='o', label='Start')
        self.ax.scatter(planned_traj[-1, 0], planned_traj[-1, 1], planned_traj[-1, 2], 
                       c='blue', s=120, marker='s', label='Goal')
        
        # Plot executed trajectory (red)
        self.ax.plot(executed_traj[:, 0], executed_traj[:, 1], executed_traj[:, 2], 
                    'r-', linewidth=3, label='Executed Trajectory', alpha=1.0, zorder=5)
        
        # Plot obstacles
        if obstacle_data:
            for i, obstacle in enumerate(obstacle_data):
                vertices = self.create_box_vertices(
                    obstacle['size'], obstacle['pos'], obstacle['rot']
                )
                faces = self.create_box_faces(vertices)
                
                # Create 3D polygon collection for the obstacle
                obstacle_collection = Poly3DCollection(faces, alpha=1.0, 
                                                     facecolor='gray', edgecolor='black', zorder=1)
                self.ax.add_collection3d(obstacle_collection)
                # edges = [
                #     [vertices[i], vertices[j]] for i, j in [
                #         (0,1), (1,2), (2,3), (3,0),  # bottom face
                #         (4,5), (5,6), (6,7), (7,4),  # top face
                #         (0,4), (1,5), (2,6), (3,7)   # sides
                #     ]
                # ]

                # # Add as a 3D line collection (wireframe box)
                # wireframe = Line3DCollection(edges, colors='black', linewidths=2.0, zorder=2)
                # self.ax.add_collection3d(wireframe)
            
            # Add obstacles to legend
            self.ax.scatter([], [], [], c='grey', s=100, marker='s', 
                          label=f'Obstacles ({len(obstacle_data)})')
        
        # Add ground plane reference
        all_points = np.vstack([planned_traj, executed_traj])
        x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
        y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
        ground_level = 0.0  # Since we already offset everything by base_offset
        
        # Create a semi-transparent ground plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        zz = np.full_like(xx, ground_level)
        self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightgreen')
        
        # Customize plot
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.set_title(f'{algorithm_name} Trajectory Visualization', fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=10)
        
        # Set equal aspect ratio
        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set a nice viewing angle
        self.ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()


def detect_algorithm_from_filename(filepath):
    """
    Algorithm name from CSV filename.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        str: Algorithm name
    """
    filename = Path(filepath).stem.lower()  # Get filename without extension
    
    # Check patterns (order matters - check rrtstar before rrt!)
    if filename.startswith('astar'):
        return 'A*'
    elif filename.startswith('rrtstar'):
        return 'RRT*'
    elif filename.startswith('rrt'):
        return 'RRT'
    elif filename.startswith('prm'):
        return 'PRM'
    else:
        print(f"Warning: Could not detect algorithm from filename '{filename}', using 'Unknown'")
        return 'Unknown'


class TrajectoryVisualizerGUI:
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Trajectory Visualizer")
        self.root.geometry("700x500")

        # Current directory for file searching
        self.current_dir = Path.cwd()

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Excavator Trajectory Visualizer",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Directory selection
        ttk.Label(main_frame, text="Search Directory:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, pady=5)

        dir_frame = ttk.Frame(main_frame)
        dir_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.dir_var = tk.StringVar(value=str(self.current_dir))
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=60)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_directory)
        browse_btn.pack(side=tk.LEFT)

        # CSV file selection
        ttk.Label(main_frame, text="Select Trajectory File:", font=("Arial", 10)).grid(row=3, column=0, sticky=tk.W, pady=(10, 5))

        # Listbox with scrollbar for CSV files
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(list_frame, height=10, yscrollcommand=scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # Bind selection event to update metrics dropdown
        self.file_listbox.bind('<<ListboxSelect>>', self.on_trajectory_select)

        # Metrics dropdown
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(metrics_frame, text="Metrics:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.metrics_var = tk.StringVar(value="Select a trajectory file to view metrics")
        self.metrics_dropdown = ttk.Combobox(metrics_frame, textvariable=self.metrics_var, state='readonly', width=70)
        self.metrics_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Base offset input
        offset_frame = ttk.Frame(main_frame)
        offset_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        ttk.Label(offset_frame, text="Base Offset (m):").pack(side=tk.LEFT, padx=(0, 5))
        self.offset_var = tk.StringVar(value="0.0")
        offset_entry = ttk.Entry(offset_frame, textvariable=self.offset_var, width=10)
        offset_entry.pack(side=tk.LEFT)

        # Filtering checkbox
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 10))

        self.filter_var = tk.BooleanVar(value=False)
        filter_checkbox = ttk.Checkbutton(filter_frame, text="Apply Filtering (remove outliers & smooth noise)",
                                         variable=self.filter_var)
        filter_checkbox.pack(side=tk.LEFT)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=(10, 0))

        visualize_btn = ttk.Button(button_frame, text="Visualize", command=self.visualize, width=15)
        visualize_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(button_frame, text="Refresh List", command=self.populate_csv_files, width=15)
        refresh_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = ttk.Button(button_frame, text="Quit", command=root.quit, width=15)
        quit_btn.pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Populate initial CSV file list
        self.populate_csv_files()

    def browse_directory(self):
        """Open directory browser dialog."""
        directory = filedialog.askdirectory(initialdir=self.current_dir, title="Select Directory")
        if directory:
            self.current_dir = Path(directory)
            self.dir_var.set(str(self.current_dir))
            self.populate_csv_files()

    def populate_csv_files(self):
        """Find and populate CSV files in the current directory and subdirectories."""
        self.file_listbox.delete(0, tk.END)

        try:
            # Search for CSV files recursively
            csv_files = sorted(self.current_dir.rglob("*.csv"))

            # Filter out files with "metrics" in the name
            trajectory_files = [f for f in csv_files if "metrics" not in f.stem.lower()]

            if not trajectory_files:
                self.file_listbox.insert(tk.END, "No trajectory CSV files found in directory")
                return

            # Add files to listbox with relative paths
            for csv_file in trajectory_files:
                relative_path = csv_file.relative_to(self.current_dir)
                display_name = f"{relative_path}"
                self.file_listbox.insert(tk.END, display_name)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan directory: {e}")

    def on_trajectory_select(self, event):
        """Handle trajectory file selection and load corresponding metrics."""
        selection = self.file_listbox.curselection()
        if not selection:
            return

        selected_file = self.file_listbox.get(selection[0])
        if selected_file in ["No trajectory CSV files found in directory", "No CSV files found in directory"]:
            return

        trajectory_file = self.current_dir / selected_file

        # Extract run number from filename (e.g., "trajectory_1.csv" -> run 1)
        trajectory_stem = trajectory_file.stem
        run_number = None

        # Try to extract run number from filename (look for _N pattern)
        import re
        match = re.search(r'_(\d+)$', trajectory_stem)
        if match:
            run_number = int(match.group(1))

        # Find metrics.csv in the same directory as the trajectory file
        trajectory_parent = trajectory_file.parent
        metrics_file = trajectory_parent / "metrics.csv"

        if metrics_file.exists():
            try:
                # Load metrics CSV
                df = pd.read_csv(metrics_file)

                # If we found a run number, use that row; otherwise use first row
                if run_number is not None and run_number <= len(df):
                    row_index = run_number - 1  # 0-indexed
                else:
                    row_index = 0

                if row_index < len(df):
                    row = df.iloc[row_index]

                    # Format each metric as a separate dropdown item
                    metrics_list = []
                    for col in df.columns:
                        if isinstance(row[col], (int, float)):
                            metrics_list.append(f"{col}: {row[col]:.3f}")
                        else:
                            metrics_list.append(f"{col}: {row[col]}")

                    self.metrics_dropdown['values'] = metrics_list
                    if metrics_list:
                        self.metrics_var.set(metrics_list[0])
                    else:
                        self.metrics_var.set("No metrics found")
                else:
                    self.metrics_var.set("No metrics found for this run")
                    self.metrics_dropdown['values'] = []

            except Exception as e:
                self.metrics_dropdown['values'] = []
                self.metrics_var.set(f"Error loading metrics: {e}")
        else:
            self.metrics_dropdown['values'] = []
            self.metrics_var.set("No metrics.csv file found in folder")

    def visualize(self):
        """Visualize the selected trajectory file."""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a CSV file to visualize")
            return

        # Get selected file
        selected_file = self.file_listbox.get(selection[0])
        if selected_file == "No CSV files found in directory":
            return

        trajectory_file = self.current_dir / selected_file

        # Validate file exists
        if not trajectory_file.exists():
            messagebox.showerror("Error", f"File not found: {trajectory_file}")
            return

        # Get base offset
        try:
            base_offset = float(self.offset_var.get())
        except ValueError:
            messagebox.showerror("Error", "Base offset must be a valid number")
            return

        # Auto-detect algorithm
        algorithm = detect_algorithm_from_filename(str(trajectory_file))

        # Initialize visualizer
        visualizer = TrajectoryVisualizer(excavator_base_offset=base_offset)

        # Get filtering setting
        apply_filtering = self.filter_var.get()

        try:
            # Load trajectory data
            planned_traj, executed_traj = visualizer.load_trajectory_data(str(trajectory_file),
                                                                         apply_filtering=apply_filtering)

            # Example obstacle data
            obstacle_data = [
                {
                    "size": np.array([0.08, 0.500, 0.30]),
                    "pos": np.array([0.55, 0.0, -0.15]),
                    "rot": np.array([1.0, 0.0, 0.0, 0.0])
                }
            ]

            # Create visualization
            visualizer.plot_trajectories(planned_traj, executed_traj, obstacle_data, algorithm)

        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to visualize trajectory:\n{e}")


def main():
    """Main function to run the trajectory visualizer."""
    parser = argparse.ArgumentParser(description='Visualize excavator trajectories')
    parser.add_argument('trajectory_file', nargs='?', default=None,
                       help='CSV file containing trajectory data (x_g,y_g,z_g,x_e,y_e,z_e)')
    parser.add_argument('--algorithm', default=None,
                       help='Algorithm name for the title (default: auto-detect from filename)')
    parser.add_argument('--base-offset', type=float, default=0,
                       help='Excavator base height offset in meters (default: 0)')
    parser.add_argument('--filter', action='store_true',
                       help='Apply filtering to remove outliers and smooth noise')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI mode for file selection')

    args = parser.parse_args()

    # If no file specified or --gui flag, launch GUI
    if args.trajectory_file is None or args.gui:
        root = tk.Tk()
        app = TrajectoryVisualizerGUI(root)
        root.mainloop()
        return

    # Command-line mode
    # Validate input file exists
    if not Path(args.trajectory_file).exists():
        print(f"Error: Trajectory file '{args.trajectory_file}' not found!")
        sys.exit(1)

    # Auto-detect algorithm if not specified
    if args.algorithm is None:
        args.algorithm = detect_algorithm_from_filename(args.trajectory_file)
        print(f"Algorithm: {args.algorithm}")

    # Initialize visualizer
    visualizer = TrajectoryVisualizer(excavator_base_offset=args.base_offset)

    # Load trajectory data
    planned_traj, executed_traj = visualizer.load_trajectory_data(args.trajectory_file,
                                                                  apply_filtering=args.filter)

    # Example obstacle data
    obstacle_data = [
        {
            "size": np.array([0.08, 0.500, 0.30]),
            "pos": np.array([0.55, 0.0, -0.15]),
            "rot": np.array([1.0, 0.0, 0.0, 0.0])
        }
    ]

    # Create visualization
    visualizer.plot_trajectories(planned_traj, executed_traj, obstacle_data, args.algorithm)


if __name__ == "__main__":
    '''
    GUI mode (default when no arguments):
        python trajectory_visualization.py

    Command-line mode:
        python trajectory_visualization.py path/to/trajectory.csv
        python trajectory_visualization.py path/to/trajectory.csv --base-offset 0.15
        python trajectory_visualization.py path/to/trajectory.csv --filter
        python trajectory_visualization.py path/to/trajectory.csv --base-offset 0.15 --filter

    Force GUI mode:
        python trajectory_visualization.py --gui
    '''
    main()