# `pathing/`: Shared sim/IRL path planning library

This package is shared by the Isaac Sim runner (`run_sim_v2.py`) and the
hardware/IRL runner (`run_hw_v2.py`). It contains three layers, in increasing
order of how much they reshape the path:

1. **Base planners** (`path_planning_algorithms.py`): classical motion
   planning algorithms operating in full 3D.
2. **Planar variants** (`*_plane` versions): the same base planners, but
   searching in a 2D plane spanned by start, goal, and the world up axis.
3. **Radial layer** (`radial_planner.py`): *not a separate planner*. A
   wrapper that adds a cylindrical seed path and post-hoc radial corner
   compensation around whichever base/planar planner you chose.

The dispatcher in `normalized_planners.py` glues these together by algorithm
name: `[radial_]<base>[_plane]`.

---

## 1. Base algorithms

All four base planners live in `path_planning_algorithms.py` and share a
common `ObstacleChecker` for collision testing.

| Algorithm  | Class      | High-level entry                  | Notes                                                                 |
| ---------- | ---------- | --------------------------------- | --------------------------------------------------------------------- |
| **A\***    | `AStar3D`  | `create_astar_3d_trajectory`      | Grid search on a 3D voxel grid with an Euclidean heuristic.           |
| **RRT**    | `RRT`      | `create_rrt_trajectory`           | Random tree expansion. No rewiring, fast, suboptimal.                 |
| **RRT\***  | `RRTStar`  | `create_rrt_star_trajectory`      | RRT with neighborhood rewiring; converges toward an optimal path.     |
| **PRM**    | `PRM`      | `create_prm_trajectory`           | Probabilistic roadmap; samples nodes, connects neighbors, then runs Dijkstra. |

All four take a start, a goal, a list of obstacle dicts (`size`, `pos`,
`rot`), a grid resolution, and a safety margin. They return a piecewise-linear
3D path in world coordinates.

---

## 2. Planar variants (`_plane`)

Each base planner also has a `_plane` flavor:

- `a_star_plane`, `rrt_plane`, `rrt_star_plane`, `prm_plane`

These search in a **2D plane spanned by the start, the goal, and the world
up axis** (constructed via `basis_start_goal_plane` in `path_utils.py`).
Obstacles are projected onto that plane by
`compute_planar_obstacle_rectangles`, and a `PlanarWorldCollisionAdapter`
keeps collision checks consistent with the real 3D world while the search
runs in plane coordinates.

Use planar variants when the workspace is roughly tabletop-flat or when you
want a much smaller search space, the planner has one fewer degree of
freedom, so it is faster and more deterministic, but might struggle on harder paths.

This is what `-p` / `--planar` switches on in both `run_sim_v2.py` and
`run_hw_v2.py`.

---

## 3. The radial layer (`-r` / `--radial`)

The radial layer is a *stack* built on top of one of the base or planar
planners. It does not implement its own obstacle-avoidance search. The
collision-avoiding work is still done by A\*, RRT, RRT\*, or PRM (3D or
`_plane`). What the layer adds is a radius-aware seed, a retreat-and-replan
strategy, smoothing, and a final reshaping that pulls the result back onto
the seed's radius profile.

**Why it exists.** The main motivation is cheap large rotations. For a
revolute arm, most of a long yaw sweep is empty space. Only the final
approach typically interacts with obstacles. Running A\* / RRT over the
whole sweep is wasteful and tends to produce wandering paths. The radial
seed does the bulk of the travel for free along a clean `r(yaw)` arc, and
the base planner is only invoked for the short segment near the obstacle
(e.g. routing the last ~30° around a box behind the robot).

For the excavator this is also a tracking advantage. A wide azimuth change
at roughly fixed radius and height is mainly a single-joint slew motion,
rather than a Cartesian detour that couples boom, stick, bucket, and slew
together. The radial layer therefore tends to give the controller a simpler
reference: let the excavator house do the large rotation, then use the base
planner only where local obstacle avoidance actually requires arm motion.

### Pipeline (`radial_planner.plan_radial`)

1. **Cylindrical seed.** Interpolate smoothly from start to goal in
   `(yaw, r, z)` coordinates. This is the geometric ideal of an arm sweep
   No obstacle reasoning, just a smooth `r(yaw)` profile.
2. **Free-travel check.** Collision-check the seed. If clear, execute it
   directly. The base planner is never called.
3. **Detour via the base planner.** If the seed hits an obstacle, take the
   collision-free *prefix* of the seed up to a handoff point, and from that
   handoff call the configured base planner (`_plane` or full 3D) to route
   around the obstacle to the goal. The result is `seed-prefix + base-suffix`
   (the **red** path: piecewise-linear, obstacle-avoiding, but radially
   unshaped).
4. **Retreat-and-retry.** If the chosen handoff produces a merged path
   that is not collision-free, step the handoff back along the seed by a
   couple of samples and re-plan. Iterate through candidate handoffs until
   a clean detour is found, otherwise fall back to planning a raw path
   directly from start.
5. **Radial repacking (yellow).** RDP-simplify the red suffix to its
   corners, then blend each interior corner's radius toward the seed's
   profile:

   ```
   r_new = α · r_seed(yaw) + (1 − α) · r_raw     # default α = 0.8
   ```

   This reshapes the detour to follow the seed's `r(yaw)` curve as closely
   as the obstacle field allows. Endpoints (handoff and goal) are
   preserved exactly. Merging this with the seed-prefix gives the
   **yellow** path.
6. **Hermite smoothing + execution selection.** A cubic-Hermite pass over
   the piecewise path, parameterised by cumulative arc length, rounds the
   kinks left by the discrete planner. It is pure geometry, no radius
   information used. Smoothing is tried on each candidate in order:
   `smooth(yellow) → smooth(red) → red`. The first variant that survives
   collision-checking is executed. With `--radial-mode raw`, step 5 is
   skipped and the order becomes `smooth(red) → red`.

### Sim colors (`--draw` overlay in `run_sim_v2.py`)

These are the line colors drawn live in the Isaac viewport during sim runs
when `--draw` is set. Hardware/IRL runs use the same planner outputs, but do
not have a viewport overlay. Offline PNGs can be exported later with
`logs_sim/trajectory_visualization.py`.

| Viewport color | Source                          | What it represents                                                  |
| -------------- | ------------------------------- | ------------------------------------------------------------------- |
| Gray           | `seed_points_world`             | Cylindrical seed, the radius-ideal reference.                       |
| Red            | `raw_suffix_points_world`       | Base-planner suffix from handoff to goal (raw obstacle avoidance).  |
| Yellow         | `radial_suffix_points_world`    | Suffix after radial repacking onto the seed's `r(yaw)` profile.     |
| Blue           | `traj_points_world`             | The trajectory actually executed (smoothed yellow / red).     |

---

## File map

| File                          | Role                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| `path_planning_algorithms.py` | A\*, RRT, RRT\*, PRM and their 3D / `_plane` wrappers.               |
| `normalized_planners.py`      | `plan_to_target` dispatcher; resolves names like `radial_a_star_plane`. |
| `radial_planner.py`           | Seed generation, handoff search, radial compensation, smoothing.     |
| `path_utils.py`               | RDP simplify, resampling, `ObstacleChecker`, plane-basis math, pose helpers. |
