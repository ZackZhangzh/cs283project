#!/usr/bin/env python3
import pybullet as p
import numpy as np
import os
import time
import rospy
import math
import rospkg
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from pynput.keyboard import Key, Listener
import sys
from scipy.spatial.transform import Rotation as R
import actionlib
from kinova_msgs.msg import ArmJointAnglesAction, ArmJointAnglesGoal
from kinova_msgs.msg import ArmPoseAction, ArmPoseGoal
from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal

try:
    from oculus_reader.scripts import *
    from oculus_reader.scripts.reader import OculusReader

    OCULUS_AVAILABLE = True
except ImportError:
    rospy.logwarn("Oculus Reader not available. Will use default values.")
    OCULUS_AVAILABLE = False

try:
    from leap_hand_utils.dynamixel_client import *
    import leap_hand_utils.leap_hand_utils as lhu

    LEAP_HAND_AVAILABLE = True
except ImportError:
    rospy.logwarn("LEAP Hand utils not available. Hand control disabled.")
    LEAP_HAND_AVAILABLE = False

"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""

control_enabled = True  # Default to enabled control state
space_pressed = True  # Default to enabled (changed from hold-to-enable to toggle mode)
keyboard_control = {
    "pos_delta": 0.01,  # Position change increment in meters
    "rot_delta": 0.05,  # Rotation change increment in radians
    "current_pos": np.array([0.4, 0.0, 0.4]),  # Default position [x, y, z]
    "current_rot": np.array([1.0, 0.0, 0.0, 0.0]),  # Default orientation [w, x, y, z]
    "keys_pressed": set(),  # Track which keys are currently pressed
}

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


# Convert 3*3 matrix to quaternion
def mat2quat(mat):
    q = np.zeros([4])
    q[0] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[1] = (mat[2][1] - mat[1][2]) / (4 * q[0])
    q[2] = (mat[0][2] - mat[2][0]) / (4 * q[0])
    q[3] = (mat[1][0] - mat[0][1]) / (4 * q[0])
    return q


# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.0 * pose[2][3]
    pos[1] = -1.0 * pose[0][3]
    pos[2] = +1.0 * pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.0 * pose[2][:3]
    mat[1][:] = +1.0 * pose[0][:3]
    mat[2][:] = -1.0 * pose[1][:3]

    return pos, mat2quat(mat)


# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = np.zeros([3])
    pos[0] = +1.0 * pose[2][3]
    pos[1] = +1.0 * pose[0][3]
    pos[2] = +1.0 * pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.0 * pose[2][:3]
    mat[1][:] = +1.0 * pose[0][:3]
    mat[2][:] = -1.0 * pose[1][:3]

    return pos, mat2quat(mat)


def vr2mj(pose):
    pos = np.zeros([3])
    pos[0] = pose[0][3]
    pos[1] = pose[1][3]
    pos[2] = pose[2][3]
    pos = pos @ OPERATOR2MANO_RIGHT
    mat = pose[:3, :3] @ OPERATOR2MANO_RIGHT
    return pos, mat2quat(mat)


def negQuat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    return res


def diffQuat(quat1, quat2):
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return diff


class LeapNode:
    def __init__(self):
        if not LEAP_HAND_AVAILABLE:
            rospy.logwarn("LEAP Hand utils not available. Hand control disabled.")
            return

        ####Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB0", 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB1", 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, "COM13", 4000000)
                self.dxl_client.connect()
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kP, 84, 2
        )  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kI, 82, 2
        )  # Igain
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kD, 80, 2
        )  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        if not LEAP_HAND_AVAILABLE:
            return
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        if not LEAP_HAND_AVAILABLE:
            return
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        if not LEAP_HAND_AVAILABLE:
            return
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        if not LEAP_HAND_AVAILABLE:
            return np.zeros(16)
        return self.dxl_client.read_pos()

    # read velocity
    def read_vel(self):
        if not LEAP_HAND_AVAILABLE:
            return np.zeros(16)
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        if not LEAP_HAND_AVAILABLE:
            return np.zeros(16)
        return self.dxl_client.read_cur()


class KinovaPybulletIK:
    def __init__(self):
        rospy.init_node("arm_teleoperation_node")

        # Get ROS parameters
        self.use_simulator = rospy.get_param("~use_simulator", True)
        self.control_real_robot = rospy.get_param("~control_real_robot", False)
        self.robot_type = rospy.get_param("~robot_type", "j2n6s300")
        self.urdf_path = rospy.get_param(
            "~urdf_path", "../kinova_description/urdf/robot.urdf"
        )
        self.update_rate = rospy.get_param("~update_rate", 30)  # Hz
        self.position_limit_margin = rospy.get_param(
            "~position_limit_margin", 0.05
        )  # meters

        # Append robot_type prefix if needed
        robot_prefix = self.robot_type + "_"

        # Publishers for joint commands (for simulation and monitoring)
        self.joint_cmd_pub = rospy.Publisher(
            "/" + robot_prefix + "driver/joint_angles/cmd",
            Float64MultiArray,
            queue_size=1,
        )
        self.pose_cmd_pub = rospy.Publisher(
            "/" + robot_prefix + "driver/tool_pose/cmd", PoseStamped, queue_size=1
        )

        # Action clients for real robot control
        if self.control_real_robot:
            rospy.loginfo("Setting up action clients for real robot control")
            self.joint_action_client = actionlib.SimpleActionClient(
                "/" + robot_prefix + "driver/joints_action/joint_angles",
                ArmJointAnglesAction,
            )
            self.pose_action_client = actionlib.SimpleActionClient(
                "/" + robot_prefix + "driver/pose_action/tool_pose",
                ArmPoseAction,
            )
            self.fingers_action_client = actionlib.SimpleActionClient(
                "/" + robot_prefix + "driver/fingers_action/finger_positions",
                SetFingersPositionAction,
            )

            # Wait for action servers to become available
            try:
                timeout = rospy.Duration(5.0)
                rospy.loginfo("Waiting for joint_angles action server...")
                self.joint_action_client.wait_for_server(timeout)
                rospy.loginfo("Waiting for tool_pose action server...")
                self.pose_action_client.wait_for_server(timeout)
                rospy.loginfo("Waiting for finger_positions action server...")
                self.fingers_action_client.wait_for_server(timeout)
                rospy.loginfo("All action servers connected!")
            except:
                rospy.logwarn(
                    "Failed to connect to one or more action servers. Will continue with simulation only."
                )
                self.control_real_robot = False

        # Subscribe to joint states if controlling real robot to monitor position
        if self.control_real_robot:
            self.joint_states_sub = rospy.Subscriber(
                "/" + robot_prefix + "driver/out/joint_state",
                JointState,
                self.joint_state_callback,
            )
            self.current_joint_positions = None
            self.joint_position_limits = {
                # Define joint limits here based on Kinova documentation
                "lower": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                "upper": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            }

        # Initialize simulator if enabled
        if self.use_simulator:
            # start pybullet
            p.connect(p.GUI)
            self.glove_to_leap_mapping_scale = 1.6
            self.kinovaEndEffectorIndex = 9

            # Find the URDF file
            if not os.path.exists(self.urdf_path):
                rospack = rospkg.RosPack()
                try:
                    kinova_path = rospack.get_path("kinova_description")
                    self.urdf_path = os.path.join(kinova_path, "urdf/robot.urdf")
                except:
                    rospy.logwarn(
                        "Could not find kinova_description package, using relative path"
                    )
                    self.urdf_path = "../kinova_description/urdf/robot.urdf"

            rospy.loginfo(f"Loading URDF from: {self.urdf_path}")

            self.kinovaId = p.loadURDF(
                self.urdf_path,
                [0.0, 0.0, 0.0],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )

            self.numJoints = p.getNumJoints(self.kinovaId)
            p.setGravity(0, 0, 0)
            useRealTimeSimulation = 0
            p.setRealTimeSimulation(useRealTimeSimulation)
            self.create_target_vis()

            for i in range(2, 8):
                p.resetJointState(self.kinovaId, i, np.pi)

            for i in range(self.numJoints):
                info = p.getJointInfo(self.kinovaId, i)
                rospy.loginfo("Joint {}: {}".format(i, info[1].decode("utf-8")))

        self.operator2mano = OPERATOR2MANO_RIGHT

        # Initialize LEAP hand if available
        if LEAP_HAND_AVAILABLE:
            self.leap_node = LeapNode()
        else:
            self.leap_node = None

        # Initialize Oculus reader if available
        if OCULUS_AVAILABLE:
            self.oculus_reader = OculusReader()
            rospy.loginfo("OculusReader initialized: %s", self.oculus_reader)
        else:
            self.oculus_reader = None
            rospy.logwarn("OculusReader not available. Using keyboard control only.")

        # Set up keyboard listener
        self.keyboard_listener = Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.keyboard_listener.start()

        # For safety, track when control was last enabled
        self.last_control_time = None
        self.control_timeout = rospy.Duration(
            0.5
        )  # timeout if no updates for 0.5 seconds

    def joint_state_callback(self, msg):
        # Store current joint positions for safety checks
        self.current_joint_positions = list(msg.position)[:6]  # Only use the arm joints

    def on_press(self, key):
        global space_pressed, keyboard_control

        # Space key toggles control mode
        if key == Key.space:
            space_pressed = not space_pressed
            if space_pressed:
                rospy.loginfo("Control ENABLED")
                self.last_control_time = rospy.Time.now()

                # Initialize keyboard control with current robot position when enabling
                if self.use_simulator:
                    link_state = p.getLinkState(
                        self.kinovaId, self.kinovaEndEffectorIndex
                    )
                    keyboard_control["current_pos"] = np.array(link_state[4])
                    keyboard_control["current_rot"] = np.array(link_state[5])
            else:
                rospy.loginfo("Control DISABLED")
                if self.control_real_robot:
                    self.stop_robot()

        # Track keyboard controls for manual operation
        if key == Key.shift:
            keyboard_control["keys_pressed"].add("shift")

        try:
            key_char = key.char.lower()
            keyboard_control["keys_pressed"].add(key_char)
        except AttributeError:
            # Special keys (like arrows) don't have a char
            pass

    def on_release(self, key):
        global keyboard_control

        # Track keyboard controls for manual operation
        if key == Key.shift:
            if "shift" in keyboard_control["keys_pressed"]:
                keyboard_control["keys_pressed"].remove("shift")

        try:
            key_char = key.char.lower()
            if key_char in keyboard_control["keys_pressed"]:
                keyboard_control["keys_pressed"].remove(key_char)
        except AttributeError:
            # Special keys don't have a char
            pass

    def stop_robot(self):
        """Send a stop command to the robot"""
        if self.control_real_robot:
            try:
                # Use current positions to avoid jerky stops
                if self.current_joint_positions:
                    goal = ArmJointAnglesGoal()
                    goal.angles.joint1 = self.current_joint_positions[0]
                    goal.angles.joint2 = self.current_joint_positions[1]
                    goal.angles.joint3 = self.current_joint_positions[2]
                    goal.angles.joint4 = self.current_joint_positions[3]
                    goal.angles.joint5 = self.current_joint_positions[4]
                    goal.angles.joint6 = self.current_joint_positions[5]
                    self.joint_action_client.send_goal(goal)
            except Exception as e:
                rospy.logerr(f"Failed to stop robot: {e}")

    def update_target_vis(self, hand_pos):
        if self.use_simulator:
            _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
            p.resetBasePositionAndOrientation(
                self.ballMbt[0], hand_pos, current_orientation
            )

    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=small_ball_radius
        )
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]

        self.ballMbt = []
        for i in range(0, 1):
            self.ballMbt.append(
                p.createMultiBody(
                    baseMass=baseMass,
                    baseCollisionShapeIndex=ball_shape,
                    basePosition=basePosition,
                )
            )  # for base and finger tip joints
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(
                self.ballMbt[i], -1, no_collision_group, no_collision_mask
            )
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1])

    def check_joint_safety(self, jointPoses):
        """Check if joint positions are within safe limits"""
        if self.current_joint_positions is None:
            return False

        # Check if any joint is out of limits
        for i in range(len(jointPoses)):
            if (
                jointPoses[i] < self.joint_position_limits["lower"][i]
                or jointPoses[i] > self.joint_position_limits["upper"][i]
            ):
                rospy.logwarn(f"Joint {i} out of safety limits: {jointPoses[i]}")
                return False

        # Check for large changes
        joint_change = np.abs(
            np.array(jointPoses) - np.array(self.current_joint_positions)
        )
        if np.max(joint_change) > 0.5:  # 0.5 radians is about 30 degrees
            rospy.logwarn(f"Large joint change detected: {np.max(joint_change)} rad")
            return False

        return True

    def check_pose_safety(self, pose_pos):
        """Check if end effector position is within safe workspace"""
        # Define a safe workspace cube
        workspace_min = np.array([0.1, -0.5, 0.1])  # adjust based on your robot
        workspace_max = np.array([0.7, 0.5, 0.7])

        pos_array = np.array([pose_pos[0], pose_pos[1], pose_pos[2]])

        # Check if position is within safety margin of workspace limits
        if np.any(pos_array < workspace_min + self.position_limit_margin) or np.any(
            pos_array > workspace_max - self.position_limit_margin
        ):
            rospy.logwarn(f"Position {pos_array} outside safe workspace")
            return False

        return True

    def compute_IK(self, arm_pos, arm_rot):
        if self.use_simulator:
            # Enforce position limits for keyboard control
            workspace_min = np.array([0.1, -0.5, 0.1])
            workspace_max = np.array([0.7, 0.5, 0.7])

            # Clamp position to safe workspace with margin
            arm_pos_array = np.array(arm_pos)
            arm_pos_array = np.clip(
                arm_pos_array,
                workspace_min + self.position_limit_margin,
                workspace_max - self.position_limit_margin,
            )
            arm_pos = arm_pos_array

            p.stepSimulation()

            jointPoses = p.calculateInverseKinematics(
                self.kinovaId,
                self.kinovaEndEffectorIndex,
                arm_pos,
                arm_rot,
                solver=p.IK_DLS,
                maxNumIterations=50,
                residualThreshold=0.0001,
            )

            qpos_now = []
            for i in range(2, 8):
                qpos_now.append(p.getJointState(self.kinovaId, i)[0])
            qpos_now = np.array(qpos_now)

            if any(math.isnan(pose) for pose in jointPoses):
                rospy.logwarn("IK solution is None")
                jointPoses = qpos_now
            else:
                jointPoses = np.array(jointPoses[:6])
                sum = 0.0
                for i in range(6):
                    sum += (jointPoses[i] - qpos_now[i]) * (jointPoses[i] - qpos_now[i])
                qpos_arm_err = math.sqrt(sum)

                if qpos_arm_err > 0.5:
                    rospy.logwarn(
                        "Jump detected. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm currently is to avoid sudden jumps.".format(
                            qpos_arm_err
                        )
                    )
                    jointPoses = qpos_now

            # update the hand joints in simulator
            for i in range(6):
                p.setJointMotorControl2(
                    bodyIndex=self.kinovaId,
                    jointIndex=i + 2,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=jointPoses[i],
                    targetVelocity=0,
                    force=500,
                    positionGain=0.3,
                    velocityGain=1,
                )

            # Publish joint commands to ROS topics
            joint_cmd_msg = Float64MultiArray()
            joint_cmd_msg.data = jointPoses
            self.joint_cmd_pub.publish(joint_cmd_msg)

            # Also publish end effector pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.robot_type + "_link_base"
            pose_msg.pose.position.x = arm_pos[0]
            pose_msg.pose.position.y = arm_pos[1]
            pose_msg.pose.position.z = arm_pos[2]
            pose_msg.pose.orientation.w = arm_rot[0]
            pose_msg.pose.orientation.x = arm_rot[1]
            pose_msg.pose.orientation.y = arm_rot[2]
            pose_msg.pose.orientation.z = arm_rot[3]
            self.pose_cmd_pub.publish(pose_msg)

            # Send commands to the real robot if enabled
            if self.control_real_robot and space_pressed:
                # Only send commands if within safe limits and timeout hasn't been reached
                current_time = rospy.Time.now()
                if self.last_control_time and (
                    current_time - self.last_control_time < self.control_timeout
                ):
                    # Perform safety checks
                    pose_safe = self.check_pose_safety(arm_pos)
                    joint_safe = self.check_joint_safety(jointPoses)

                    if pose_safe and joint_safe:
                        # Update the time of last valid control
                        self.last_control_time = current_time

                        # Send joint angles command
                        joint_goal = ArmJointAnglesGoal()
                        joint_goal.angles.joint1 = jointPoses[0]
                        joint_goal.angles.joint2 = jointPoses[1]
                        joint_goal.angles.joint3 = jointPoses[2]
                        joint_goal.angles.joint4 = jointPoses[3]
                        joint_goal.angles.joint5 = jointPoses[4]
                        joint_goal.angles.joint6 = jointPoses[5]
                        self.joint_action_client.send_goal(joint_goal)

                        # Also send Cartesian pose command for better control
                        pose_goal = ArmPoseGoal()
                        pose_goal.pose = pose_msg
                        self.pose_action_client.send_goal(pose_goal)
                    else:
                        rospy.logwarn(
                            "Command not sent to real robot due to safety constraints"
                        )
                else:
                    if self.last_control_time:
                        rospy.logwarn(
                            "Control timeout reached, not sending commands to real robot"
                        )
                    self.stop_robot()

    def process_keyboard_control(self):
        """Process keyboard inputs to control the arm when Oculus is not available"""
        global keyboard_control

        # Position control keys
        # WASD for X/Y plane movement, R/F for Z axis
        # QEIJKL for orientation control

        # Speed adjustment based on shift key
        fast_mode = False
        if "shift" in keyboard_control["keys_pressed"]:
            fast_mode = True

        pos_delta = keyboard_control["pos_delta"] * (3.0 if fast_mode else 1.0)
        rot_delta = keyboard_control["rot_delta"] * (3.0 if fast_mode else 1.0)
        keys = keyboard_control["keys_pressed"]

        # Position control (X, Y, Z)
        if "w" in keys:  # Forward (+X)
            keyboard_control["current_pos"][0] += pos_delta
            rospy.logdebug(
                f"Moving forward, position: {keyboard_control['current_pos']}"
            )
        if "s" in keys:  # Backward (-X)
            keyboard_control["current_pos"][0] -= pos_delta
            rospy.logdebug(
                f"Moving backward, position: {keyboard_control['current_pos']}"
            )
        if "d" in keys:  # Right (+Y)
            keyboard_control["current_pos"][1] += pos_delta
            rospy.logdebug(f"Moving right, position: {keyboard_control['current_pos']}")
        if "a" in keys:  # Left (-Y)
            keyboard_control["current_pos"][1] -= pos_delta
            rospy.logdebug(f"Moving left, position: {keyboard_control['current_pos']}")
        if "r" in keys:  # Up (+Z)
            keyboard_control["current_pos"][2] += pos_delta
            rospy.logdebug(f"Moving up, position: {keyboard_control['current_pos']}")
        if "f" in keys:  # Down (-Z)
            keyboard_control["current_pos"][2] -= pos_delta
            rospy.logdebug(f"Moving down, position: {keyboard_control['current_pos']}")

        # Rotation control using quaternions
        # Create rotation matrices for small increments around each axis
        if any(k in keys for k in ["q", "e", "i", "k", "j", "l"]):
            # Get current rotation as quaternion
            current_quat = keyboard_control["current_rot"]

            # Convert to rotation matrix
            r = R.from_quat(
                [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
            )
            rotation_matrix = r.as_matrix()

            # Apply incremental rotations based on keypresses
            if "q" in keys:  # Roll left
                delta_r = R.from_euler("x", -rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix
            if "e" in keys:  # Roll right
                delta_r = R.from_euler("x", rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix
            if "i" in keys:  # Pitch up
                delta_r = R.from_euler("y", rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix
            if "k" in keys:  # Pitch down
                delta_r = R.from_euler("y", -rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix
            if "j" in keys:  # Yaw left
                delta_r = R.from_euler("z", -rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix
            if "l" in keys:  # Yaw right
                delta_r = R.from_euler("z", rot_delta)
                rotation_matrix = delta_r.as_matrix() @ rotation_matrix

            # Convert back to quaternion
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()  # Returns x, y, z, w

            # Update current rotation (w, x, y, z format)
            keyboard_control["current_rot"] = np.array(
                [quat[3], quat[0], quat[1], quat[2]]
            )

        # Return the current position and orientation
        return keyboard_control["current_pos"], keyboard_control["current_rot"]

    def run(self):
        rate = rospy.Rate(self.update_rate)  # Hz
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None

        # Initialize keyboard control with current robot position
        if self.use_simulator:
            link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
            keyboard_control["current_pos"] = np.array(link_state[4])
            keyboard_control["current_rot"] = np.array(link_state[5])

        rospy.loginfo("Arm teleoperation node running. Press SPACE to control the arm.")
        if self.control_real_robot:
            rospy.loginfo("REAL ROBOT CONTROL ENABLED - BE CAREFUL!")

        # For visual feedback on control state
        control_text_id = None

        while not rospy.is_shutdown():
            # Visual feedback for control state
            if self.use_simulator:
                if control_text_id is not None:
                    p.removeUserDebugItem(control_text_id)

                control_text = (
                    "CONTROL ENABLED" if space_pressed else "CONTROL DISABLED"
                )
                # Fix color format: Use 3-element RGB array instead of 4-element RGBA
                text_color = [0, 1, 0] if space_pressed else [1, 0, 0]
                control_text_id = p.addUserDebugText(
                    control_text,
                    [0, 0, 0.8],  # Position above the robot
                    textColorRGB=text_color,
                    textSize=1.5,
                )

            if OCULUS_AVAILABLE and self.oculus_reader:
                joint_pos = self.oculus_reader.get_joint_transformations()[1]
                transformations, buttons = (
                    self.oculus_reader.get_transformations_and_buttons()
                )

                # kinova
                if transformations and "r" in transformations:
                    right_controller_pose = transformations["r"]
                    VRpos, VRquat = vrfront2mj(right_controller_pose)

                    if space_pressed:
                        # dVRP/R = VRP/Rt - VRP/R0
                        dVRP = VRpos - VRP0
                        # dVRR = VRquat - VRR0
                        dVRR = diffQuat(VRR0, VRquat)
                        # MJP/Rt =  MJP/R0 + dVRP/R

                        curr_pos = MJP0 + dVRP
                        curr_quat = mulQuat(MJR0, dVRR)

                    # Adjust origin if not engaged
                    else:
                        # RP/R0 = RP/Rt
                        if self.use_simulator:
                            link_state = p.getLinkState(self.kinovaId, 9)
                            curr_pos = link_state[4]
                            curr_quat = link_state[5]

                        else:
                            # In non-simulator mode, use a fixed starting position
                            curr_pos = np.array([0.4, 0.0, 0.4])  # Default position
                            curr_quat = np.array(
                                [1.0, 0.0, 0.0, 0.0]
                            )  # Default orientation (w,x,y,z)

                        MJP0 = curr_pos  # real kinova pos origin
                        MJR0 = curr_quat  # real kinova quat origin

                        # VP/R0 = VP/Rt
                        VRP0 = VRpos
                        VRR0 = VRquat

                    # update desired pos
                    target_pos = curr_pos
                    target_quat = curr_quat

                    self.compute_IK(target_pos, target_quat)
                    self.update_target_vis(target_pos)
            else:
                # Process keyboard control if Oculus is not available
                if space_pressed:
                    target_pos, target_quat = self.process_keyboard_control()
                    self.compute_IK(target_pos, target_quat)
                    self.update_target_vis(target_pos)
                else:
                    # When control is disabled, update keyboard control to current position
                    if self.use_simulator:
                        link_state = p.getLinkState(
                            self.kinovaId, self.kinovaEndEffectorIndex
                        )
                        keyboard_control["current_pos"] = np.array(link_state[4])
                        keyboard_control["current_rot"] = np.array(link_state[5])

            rate.sleep()


def main():
    try:
        kinova_controller = KinovaPybulletIK()
        kinova_controller.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
