#!/usr/bin/env python3
import pybullet as p
import numpy as np
import os
import rospy
import math
import rospkg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from pynput.keyboard import Key, Listener
from scipy.spatial.transform import Rotation as R
import actionlib
from kinova_msgs.msg import ArmJointAnglesAction, ArmJointAnglesGoal
from kinova_msgs.msg import ArmPoseAction, ArmPoseGoal
from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal

# Try to import optional dependencies
try:
    from oculus_reader.scripts import *
    from oculus_reader.scripts.reader import OculusReader

    OCULUS_AVAILABLE = True
except ImportError:
    rospy.logwarn("Oculus Reader not available. Will use keyboard control.")
    OCULUS_AVAILABLE = False

try:
    from leap_hand_utils.dynamixel_client import *
    import leap_hand_utils.leap_hand_utils as lhu

    LEAP_HAND_AVAILABLE = True
except ImportError:
    rospy.logwarn("LEAP Hand utils not available. Hand control disabled.")
    LEAP_HAND_AVAILABLE = False

"""
Arm teleoperation node for Kinova robotic arm.
Controls a virtual PyBullet simulation and optionally a real Kinova arm simultaneously.
Can be controlled using an Oculus VR controller or keyboard.

Toggle control mode with Space key. When in control mode:
- WASD: Move in X/Y plane
- R/F: Move up/down (Z axis)
- QEIJKL: Control orientation
- Shift: Hold for faster movement
"""

# Global state variables
control_enabled = True  # Default to enabled control state
space_pressed = True  # Default to enabled (toggle mode)
keyboard_control = {
    "pos_delta": 0.01,  # Position change increment in meters
    "rot_delta": 0.05,  # Rotation change increment in radians
    "current_pos": np.array([0.4, 0.0, 0.4]),  # Default position [x, y, z]
    "current_rot": np.array([1.0, 0.0, 0.0, 0.0]),  # Default orientation [w, x, y, z]
    "keys_pressed": set(),  # Track which keys are currently pressed
}

# Coordinate transformation matrices
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


# Utility functions for coordinate transformations and quaternion operations
def mat2quat(mat):
    """Convert 3x3 rotation matrix to quaternion"""
    q = np.zeros([4])
    q[0] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[1] = (mat[2][1] - mat[1][2]) / (4 * q[0])
    q[2] = (mat[0][2] - mat[2][0]) / (4 * q[0])
    q[3] = (mat[1][0] - mat[0][1]) / (4 * q[0])
    return q


def vrfront2mj(pose):
    """VR to MuJoCo mapping when teleOp user is standing in front of the robot"""
    pos = np.zeros([3])
    pos[0] = -1.0 * pose[2][3]
    pos[1] = -1.0 * pose[0][3]
    pos[2] = +1.0 * pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.0 * pose[2][:3]
    mat[1][:] = +1.0 * pose[0][:3]
    mat[2][:] = -1.0 * pose[1][:3]

    return pos, mat2quat(mat)


def negQuat(quat):
    """Negate a quaternion"""
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def mulQuat(qa, qb):
    """Multiply two quaternions"""
    res = np.zeros(4)
    res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    return res


def diffQuat(quat1, quat2):
    """Difference between two quaternions"""
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return diff


# Optional LEAP hand controller class
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

        # Publishers for joint commands
        self.joint_cmd_pub = rospy.Publisher(
            "/" + robot_prefix + "driver/joint_angles/cmd",
            Float64MultiArray,
            queue_size=1,
        )
        self.pose_cmd_pub = rospy.Publisher(
            "/" + robot_prefix + "driver/tool_pose/cmd", PoseStamped, queue_size=1
        )

        # Setup for real robot control
        if self.control_real_robot:
            self._setup_real_robot_control(robot_prefix)

        # Initialize simulator
        if self.use_simulator:
            self._init_simulator()

        self.operator2mano = OPERATOR2MANO_RIGHT

        # Initialize optional components
        if LEAP_HAND_AVAILABLE:
            self.leap_node = LeapNode()
        else:
            self.leap_node = None

        if OCULUS_AVAILABLE:
            self.oculus_reader = OculusReader()
            rospy.loginfo("OculusReader initialized: %s", self.oculus_reader)
        else:
            self.oculus_reader = None
            rospy.loginfo("Using keyboard control for arm teleoperation")

        # Set up keyboard listener
        self.keyboard_listener = Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.keyboard_listener.start()

        # Safety tracking
        self.last_control_time = rospy.Time.now()
        self.control_timeout = rospy.Duration(
            0.5
        )  # timeout if no updates for 0.5 seconds

        # Debug display
        self.control_text_id = None

    def _setup_real_robot_control(self, robot_prefix):
        """Initialize components for real robot control"""
        rospy.loginfo("Setting up action clients for real robot control")

        # Action clients
        self.joint_action_client = actionlib.SimpleActionClient(
            "/" + robot_prefix + "driver/joints_action/joint_angles",
            ArmJointAnglesAction,
        )
        self.pose_action_client = actionlib.SimpleActionClient(
            "/" + robot_prefix + "driver/pose_action/tool_pose", ArmPoseAction
        )
        self.fingers_action_client = actionlib.SimpleActionClient(
            "/" + robot_prefix + "driver/fingers_action/finger_positions",
            SetFingersPositionAction,
        )

        # Wait for action servers
        try:
            timeout = rospy.Duration(5.0)
            rospy.loginfo("Waiting for joint_angles action server...")
            self.joint_action_client.wait_for_server(timeout)
            rospy.loginfo("Waiting for tool_pose action server...")
            self.pose_action_client.wait_for_server(timeout)
            rospy.loginfo("Waiting for finger_positions action server...")
            self.fingers_action_client.wait_for_server(timeout)
            rospy.loginfo("All action servers connected!")

            # Subscribe to joint states for safety checks
            self.joint_states_sub = rospy.Subscriber(
                "/" + robot_prefix + "driver/out/joint_state",
                JointState,
                self.joint_state_callback,
            )
            self.current_joint_positions = None
            self.joint_position_limits = {
                "lower": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                "upper": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            }
        except:
            rospy.logwarn(
                "Failed to connect to action servers. Will continue with simulation only."
            )
            self.control_real_robot = False

    def _init_simulator(self):
        """Initialize the PyBullet simulator"""
        # Start PyBullet
        p.connect(p.GUI)
        self.kinovaEndEffectorIndex = 9

        # Find URDF file
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

        # Load robot model
        self.kinovaId = p.loadURDF(
            self.urdf_path,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        self.numJoints = p.getNumJoints(self.kinovaId)
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        # Create visual target marker
        self.create_target_vis()

        # Initialize robot pose
        for i in range(2, 8):
            p.resetJointState(self.kinovaId, i, np.pi)

        # Log joint info
        for i in range(self.numJoints):
            info = p.getJointInfo(self.kinovaId, i)
            rospy.logdebug("Joint {}: {}".format(i, info[1].decode("utf-8")))

    def joint_state_callback(self, msg):
        """Store current joint positions for safety checks"""
        self.current_joint_positions = list(msg.position)[:6]  # Only arm joints

    def on_press(self, key):
        """Handle keyboard press events"""
        global space_pressed, keyboard_control

        # Space key toggles control mode
        if key == Key.space:
            space_pressed = not space_pressed
            if space_pressed:
                rospy.loginfo("Control ENABLED")
                self.last_control_time = rospy.Time.now()

                # Initialize keyboard control with current robot position
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

        # Track shift key
        if key == Key.shift:
            keyboard_control["keys_pressed"].add("shift")

        # Track regular keys
        try:
            key_char = key.char.lower()
            keyboard_control["keys_pressed"].add(key_char)
        except AttributeError:
            pass  # Not a character key

    def on_release(self, key):
        """Handle keyboard release events"""
        global keyboard_control

        # Track shift key
        if key == Key.shift:
            if "shift" in keyboard_control["keys_pressed"]:
                keyboard_control["keys_pressed"].remove("shift")

        # Track regular keys
        try:
            key_char = key.char.lower()
            if key_char in keyboard_control["keys_pressed"]:
                keyboard_control["keys_pressed"].remove(key_char)
        except AttributeError:
            pass  # Not a character key

    def stop_robot(self):
        """Send a stop command to the real robot"""
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

    def create_target_vis(self):
        """Create a visual marker for the target position"""
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]

        self.ballMbt = []
        self.ballMbt.append(
            p.createMultiBody(
                baseMass=baseMass,
                baseCollisionShapeIndex=ball_shape,
                basePosition=basePosition,
            )
        )

        # Disable collisions
        no_collision_group = 0
        no_collision_mask = 0
        p.setCollisionFilterGroupMask(
            self.ballMbt[0], -1, no_collision_group, no_collision_mask
        )

        # Set color to red
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1])

    def update_target_vis(self, hand_pos):
        """Update the visual marker position"""
        if self.use_simulator:
            _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
            p.resetBasePositionAndOrientation(
                self.ballMbt[0], hand_pos, current_orientation
            )

    def check_joint_safety(self, jointPoses):
        """Check if joint positions are within safe limits"""
        if self.current_joint_positions is None:
            return False

        # Check joint limits
        for i in range(len(jointPoses)):
            if (
                jointPoses[i] < self.joint_position_limits["lower"][i]
                or jointPoses[i] > self.joint_position_limits["upper"][i]
            ):
                rospy.logwarn(f"Joint {i} out of safety limits: {jointPoses[i]}")
                return False

        # Check for large movements
        joint_change = np.abs(
            np.array(jointPoses) - np.array(self.current_joint_positions)
        )
        if np.max(joint_change) > 0.5:  # 0.5 radians is about 30 degrees
            rospy.logwarn(f"Large joint change detected: {np.max(joint_change)} rad")
            return False

        return True

    def check_pose_safety(self, pose_pos):
        """Check if end effector position is within safe workspace"""
        # Define safe workspace
        workspace_min = np.array([0.1, -0.5, 0.1])
        workspace_max = np.array([0.7, 0.5, 0.7])
        pos_array = np.array(pose_pos)

        # Check workspace limits
        if np.any(pos_array < workspace_min + self.position_limit_margin) or np.any(
            pos_array > workspace_max - self.position_limit_margin
        ):
            rospy.logwarn(f"Position {pos_array} outside safe workspace")
            return False

        return True

    def process_keyboard_control(self):
        """Process keyboard inputs for arm control"""
        global keyboard_control

        # Speed adjustment for shift key
        fast_mode = "shift" in keyboard_control["keys_pressed"]
        pos_delta = keyboard_control["pos_delta"] * (3.0 if fast_mode else 1.0)
        rot_delta = keyboard_control["rot_delta"] * (3.0 if fast_mode else 1.0)
        keys = keyboard_control["keys_pressed"]

        # Position control (X, Y, Z)
        if "w" in keys:  # Forward (+X)
            keyboard_control["current_pos"][0] += pos_delta
        if "s" in keys:  # Backward (-X)
            keyboard_control["current_pos"][0] -= pos_delta
        if "d" in keys:  # Right (+Y)
            keyboard_control["current_pos"][1] += pos_delta
        if "a" in keys:  # Left (-Y)
            keyboard_control["current_pos"][1] -= pos_delta
        if "r" in keys:  # Up (+Z)
            keyboard_control["current_pos"][2] += pos_delta
        if "f" in keys:  # Down (-Z)
            keyboard_control["current_pos"][2] -= pos_delta

        # Orientation control
        if any(k in keys for k in ["q", "e", "i", "k", "j", "l"]):
            current_quat = keyboard_control["current_rot"]
            r = R.from_quat(
                [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
            )
            rotation_matrix = r.as_matrix()

            # Apply rotations based on keypresses
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
            quat = r.as_quat()  # x, y, z, w
            keyboard_control["current_rot"] = np.array(
                [quat[3], quat[0], quat[1], quat[2]]
            )  # w, x, y, z

        return keyboard_control["current_pos"], keyboard_control["current_rot"]

    def compute_IK(self, arm_pos, arm_rot):
        """Compute inverse kinematics and update robot pose"""
        if self.use_simulator:
            # Enforce position limits
            workspace_min = np.array([0.1, -0.5, 0.1])
            workspace_max = np.array([0.7, 0.5, 0.7])
            arm_pos_array = np.array(arm_pos)
            arm_pos_array = np.clip(
                arm_pos_array,
                workspace_min + self.position_limit_margin,
                workspace_max - self.position_limit_margin,
            )
            arm_pos = arm_pos_array

            p.stepSimulation()

            # Calculate IK
            jointPoses = p.calculateInverseKinematics(
                self.kinovaId,
                self.kinovaEndEffectorIndex,
                arm_pos,
                arm_rot,
                solver=p.IK_DLS,
                maxNumIterations=50,
                residualThreshold=0.0001,
            )

            # Get current joint positions
            qpos_now = []
            for i in range(2, 8):
                qpos_now.append(p.getJointState(self.kinovaId, i)[0])
            qpos_now = np.array(qpos_now)

            # Validate IK solution
            if any(math.isnan(pose) for pose in jointPoses):
                rospy.logwarn("IK solution is None")
                jointPoses = qpos_now
            else:
                jointPoses = np.array(jointPoses[:6])

                # Check for large joint changes
                sum_sq_diff = 0.0
                for i in range(6):
                    sum_sq_diff += (jointPoses[i] - qpos_now[i]) ** 2
                qpos_arm_err = math.sqrt(sum_sq_diff)

                if qpos_arm_err > 0.5:
                    rospy.logwarn(
                        f"Jump detected. Joint error {qpos_arm_err}. Resetting goal to current position."
                    )
                    jointPoses = qpos_now

            # Update simulation
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

            # Publish to ROS topics
            self._publish_joint_commands(jointPoses, arm_pos, arm_rot)

            # Send commands to real robot if enabled
            if self.control_real_robot and space_pressed:
                self._send_commands_to_real_robot(jointPoses, arm_pos, arm_rot)

    def _publish_joint_commands(self, jointPoses, arm_pos, arm_rot):
        """Publish joint commands to ROS topics"""
        # Joint commands
        joint_cmd_msg = Float64MultiArray()
        joint_cmd_msg.data = jointPoses
        self.joint_cmd_pub.publish(joint_cmd_msg)

        # End effector pose
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

        return pose_msg

    def _send_commands_to_real_robot(self, jointPoses, arm_pos, arm_rot):
        """Send commands to the real robot with safety checks"""
        current_time = rospy.Time.now()

        # Check control timeout
        if self.last_control_time and (
            current_time - self.last_control_time < self.control_timeout
        ):
            # Perform safety checks
            pose_safe = self.check_pose_safety(arm_pos)
            joint_safe = self.check_joint_safety(jointPoses)

            if pose_safe and joint_safe:
                # Update the time of last valid control
                self.last_control_time = current_time

                # Create and send pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = current_time
                pose_msg.header.frame_id = self.robot_type + "_link_base"
                pose_msg.pose.position.x = arm_pos[0]
                pose_msg.pose.position.y = arm_pos[1]
                pose_msg.pose.position.z = arm_pos[2]
                pose_msg.pose.orientation.w = arm_rot[0]
                pose_msg.pose.orientation.x = arm_rot[1]
                pose_msg.pose.orientation.y = arm_rot[2]
                pose_msg.pose.orientation.z = arm_rot[3]

                # Send joint angles command
                joint_goal = ArmJointAnglesGoal()
                joint_goal.angles.joint1 = jointPoses[0]
                joint_goal.angles.joint2 = jointPoses[1]
                joint_goal.angles.joint3 = jointPoses[2]
                joint_goal.angles.joint4 = jointPoses[3]
                joint_goal.angles.joint5 = jointPoses[4]
                joint_goal.angles.joint6 = jointPoses[5]
                self.joint_action_client.send_goal(joint_goal)

                # Also send Cartesian pose command
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

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(self.update_rate)
        VRP0 = None  # VR position origin
        VRR0 = None  # VR rotation origin
        MJP0 = None  # Robot position origin
        MJR0 = None  # Robot rotation origin

        # Initialize keyboard control with current robot position
        if self.use_simulator:
            link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
            keyboard_control["current_pos"] = np.array(link_state[4])
            keyboard_control["current_rot"] = np.array(link_state[5])

        rospy.loginfo("Arm teleoperation node running. Press SPACE to toggle control.")
        if self.control_real_robot:
            rospy.loginfo("REAL ROBOT CONTROL ENABLED - BE CAREFUL!")

        while not rospy.is_shutdown():
            # Update visual feedback for control state
            self._update_control_state_display()

            # Process control inputs
            if OCULUS_AVAILABLE and self.oculus_reader:
                self._process_oculus_control(VRP0, VRR0, MJP0, MJR0)
            else:
                self._process_keyboard_control()

            rate.sleep()

    def _update_control_state_display(self):
        """Update the visual display of control state in simulator"""
        if self.use_simulator:
            if self.control_text_id is not None:
                p.removeUserDebugItem(self.control_text_id)

            control_text = "CONTROL ENABLED" if space_pressed else "CONTROL DISABLED"
            text_color = [0, 1, 0] if space_pressed else [1, 0, 0]  # Green/Red
            self.control_text_id = p.addUserDebugText(
                control_text,
                [0, 0, 0.8],  # Position above the robot
                textColorRGB=text_color,
                textSize=1.5,
            )

    def _process_oculus_control(self, VRP0, VRR0, MJP0, MJR0):
        """Process control inputs from Oculus controller"""
        joint_pos = self.oculus_reader.get_joint_transformations()[1]
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()

        if transformations and "r" in transformations:
            right_controller_pose = transformations["r"]
            VRpos, VRquat = vrfront2mj(right_controller_pose)

            if space_pressed:
                # Calculate position and rotation changes
                dVRP = VRpos - VRP0
                dVRR = diffQuat(VRR0, VRquat)

                # Apply changes to robot pose
                curr_pos = MJP0 + dVRP
                curr_quat = mulQuat(MJR0, dVRR)
            else:
                # When not in control mode, update reference positions
                if self.use_simulator:
                    link_state = p.getLinkState(
                        self.kinovaId, self.kinovaEndEffectorIndex
                    )
                    curr_pos = link_state[4]
                    curr_quat = link_state[5]
                else:
                    # Default position when not in simulator
                    curr_pos = np.array([0.4, 0.0, 0.4])
                    curr_quat = np.array([1.0, 0.0, 0.0, 0.0])

                # Store reference positions
                MJP0 = curr_pos
                MJR0 = curr_quat
                VRP0 = VRpos
                VRR0 = VRquat

            # Apply to robot
            self.compute_IK(curr_pos, curr_quat)
            self.update_target_vis(curr_pos)

    def _process_keyboard_control(self):
        """Process keyboard control inputs"""
        if space_pressed:
            target_pos, target_quat = self.process_keyboard_control()
            self.compute_IK(target_pos, target_quat)
            self.update_target_vis(target_pos)
        else:
            # When not in control mode, update keyboard position to match robot
            if self.use_simulator:
                link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
                keyboard_control["current_pos"] = np.array(link_state[4])
                keyboard_control["current_rot"] = np.array(link_state[5])


def main():
    try:
        kinova_controller = KinovaPybulletIK()
        kinova_controller.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
