# Arm Teleoperation

This ROS package provides a node for teleoperating a Kinova robotic arm using an Oculus VR controller. It uses PyBullet for computing inverse kinematics.

## Prerequisites

Before running the package, make sure you have installed the required dependencies:

```bash
pip install numpy pybullet scipy pynput
```

## Usage

To run the arm teleoperation node:

```bash
roslaunch arm_teleoperation arm_teleoperation.launch
```

### Parameters

- `use_simulator` (default: true): Whether to use PyBullet simulator for IK
- `urdf_path` (default: from kinova_description): Path to the robot URDF file

### Controls

- **Space bar**: Press and hold to engage arm teleoperation
- **Oculus controller**: Move the right controller to control the arm's end effector position

## Notes

- The node can operate with or without a physical arm connected
- In simulation mode, a red ball marker shows the target position
- The node publishes joint angles and end effector pose for the Kinova arm
