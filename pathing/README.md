# Pathing and kinematics testing scripts

This repo contains files for running kinematics/pathing demo similarly as in IsaacLab.

Control Architecture uses dual-loop control system:
  - Inner loop (joint-space): IK desired angles vs IMU measured angles → PID controllers → hydraulic actuators
  - Outer loop (task-space): Target positions vs FK calculated positions → IK adjustments for accuracy


Known issues (or missing stuff to test):
- Sim uses linear revolute joints directly. With constant input the rotation velocity will be constant.
- Real excavator uses hydraulic actuators that rotate the joint by driving a linkage. With constant input the rotation velocity will vary depending on the linkage curve.
  - Fix: Use prismatic joints in simulation or create controller that adds linkage rate to the direct joint control.
- Way to measure slew (= center rotation) is missing!
- Valve simulation is missing!
- Some logging issues ja bugs


## Files

- `example.py` - Basic IK example, travel between two points
- `example_astar.py` - A* pathfinding algorithm implementation
- `pathing_config.py` - Configuration settings for pathfinding algorithms. This files is compatible with the simulation version as well!


