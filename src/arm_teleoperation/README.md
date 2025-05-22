# Arm Teleoperation

This ROS package provides a node for teleoperating a Kinova robotic arm using an Oculus VR controller. It can control both a virtual PyBullet simulation and a real Kinova arm simultaneously.

## Prerequisites

Before running the package, make sure you have installed the required dependencies:

```bash
pip install numpy pybullet scipy pynput
```

Additionally, you need the Kinova ROS packages installed for controlling the real robot.

## Usage

### Simulation Only

To run the arm teleoperation node with only simulation (default):

```bash
roslaunch arm_teleoperation arm_teleoperation.launch
```

### Simulation + Real Robot Control

To enable control of the real robot while maintaining the simulation:

```bash
roslaunch arm_teleoperation arm_teleoperation.launch control_real_robot:=true
```

**IMPORTANT:** When controlling the real robot, ensure there is sufficient clearance around the robot and be ready to release the Space key to stop the robot immediately.

### Parameters

- `use_simulator` (default: true): Whether to use PyBullet simulator for IK
- `control_real_robot` (default: false): Whether to send commands to the real robot
- `robot_type` (default: j2n6s300): Type of Kinova robot
- `urdf_path` (default: from kinova_description): Path to the robot URDF file
- `update_rate` (default: 30): Control loop update rate in Hz
- `position_limit_margin` (default: 0.05): Safety margin for position limits in meters

### Controls

- **Space bar**: Press and hold to engage arm teleoperation
- **Oculus controller**: Move the right controller to control the arm's end effector position

## Safety Features

The node includes several safety features when controlling the real robot:

1. **Control timeout**: If no commands are received for 0.5 seconds, the robot stops
2. **Workspace limits**: The end effector position is checked against predefined workspace boundaries
3. **Joint limits**: The joint positions are checked against the robot's joint limits
4. **Large movement detection**: Commands that would cause large joint movements are rejected
5. **Emergency stop**: Releasing the Space key immediately stops the robot

## Notes

- The node publishes joint angles and end effector pose for the Kinova arm
- In simulation mode, a red ball marker shows the target position
- When controlling the real robot, the simulation provides a real-time preview of movements
