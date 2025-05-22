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
import sys

# Define OCULUS_AVAILABLE at the module level so it's accessible everywhere
OCULUS_AVAILABLE = False

# Try to import optional dependencies
try:
    from oculus_reader.scripts import *
    from oculus_reader.scripts.reader import OculusReader

    OCULUS_AVAILABLE = True
except ImportError:
    OCULUS_AVAILABLE = False

"""
Arm teleoperation node for Kinova robotic arm.
Controls a virtual PyBullet simulation.
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
auto_mode = False  # Whether the arm automatically moves to the red dot
keyboard_control = {
    "pos_delta": 0.01,  # Position change increment in meters
    "rot_delta": 0.05,  # Rotation change increment in radians
    "current_pos": np.array([0.4, 0.0, 0.4]),  # Default position [x, y, z]
    "current_rot": np.array([1.0, 0.0, 0.0, 0.0]),  # Default orientation [w, x, y, z]
    "keys_pressed": set(),  # Track which keys are currently pressed
    "target_pos": np.array([0.4, 0.0, 0.4]),  # Target position for auto mode
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


class KinovaPybulletIK:
    def __init__(self):
        rospy.init_node("arm_teleoperation_node")

        # Get ROS parameters
        self.update_rate = rospy.get_param("~update_rate", 30)  # Hz
        self.position_limit_margin = rospy.get_param(
            "~position_limit_margin", 0.05
        )  # meters
        self.urdf_path = rospy.get_param(
            "~urdf_path", "../kinova_description/urdf/robot.urdf"
        )
        self.robot_type = rospy.get_param("~robot_type", "j2n6s300")

        # Publishers for joint commands
        self.joint_cmd_pub = rospy.Publisher(
            "/" + self.robot_type + "_driver/joint_angles/cmd",
            Float64MultiArray,
            queue_size=1,
        )
        self.pose_cmd_pub = rospy.Publisher(
            "/" + self.robot_type + "_driver/tool_pose/cmd", PoseStamped, queue_size=1
        )

        # Initialize simulator
        self._init_simulator()

        self.operator2mano = OPERATOR2MANO_RIGHT

        # Try to initialize Oculus reader first
        self.oculus_reader = None
        if OCULUS_AVAILABLE:
            try:
                self.oculus_reader = OculusReader()
                rospy.loginfo("OculusReader initialized: %s", self.oculus_reader)

                # Test if the reader is actually working by attempting to get data
                transformations, buttons = (
                    self.oculus_reader.get_transformations_and_buttons()
                )
                if transformations is None or len(transformations) == 0:
                    rospy.logwarn(
                        "Oculus connected but no data received. Oculus controller might be off."
                    )
                    self.oculus_reader = None
                    OCULUS_AVAILABLE = False
            except Exception as e:
                rospy.logerr(f"Failed to initialize OculusReader: {e}")
                self.oculus_reader = None
                OCULUS_AVAILABLE = False

        # If Oculus is not available, ask user if they want to use keyboard control
        if not OCULUS_AVAILABLE:
            rospy.logwarn("Oculus Reader not available.")
            print("\nDo you want to proceed with keyboard control? (y/n): ")
            user_input = input().strip().lower()

            if user_input != "y" and user_input != "yes":
                rospy.loginfo(
                    "User chose not to continue with keyboard control. Exiting."
                )
                sys.exit(0)

            rospy.loginfo("Proceeding with keyboard control for arm teleoperation")

        # Set up keyboard listener
        self.keyboard_listener = Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.keyboard_listener.start()

        # Safety tracking
        self.last_control_time = rospy.Time.now()

        # Debug display
        self.control_text_id = None
        # Flag to track if we need to initialize origins when control is engaged
        self.need_init_origins = True

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

    def on_press(self, key):
        """Handle keyboard press events"""
        global space_pressed, keyboard_control, auto_mode

        # Space key toggles control mode
        if key == Key.space:
            space_pressed = not space_pressed
            # When disabling control, also disable auto mode
            if not space_pressed:
                auto_mode = False
                rospy.loginfo("Control DISABLED")
            else:
                rospy.loginfo("Control ENABLED")
                self.last_control_time = rospy.Time.now()

                # Initialize keyboard control with current robot position
                link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
                keyboard_control["current_pos"] = np.array(link_state[4])
                keyboard_control["current_rot"] = np.array(link_state[5])

        # 'T' key toggles auto mode (move to target automatically)
        elif key == Key.tab or (hasattr(key, "char") and key.char == "t"):
            if space_pressed:  # Only toggle if control is enabled
                auto_mode = not auto_mode
                if auto_mode:
                    rospy.loginfo("AUTO MODE ENABLED - Moving to red dot target")
                else:
                    rospy.loginfo("AUTO MODE DISABLED")

        # 'M' key to manually move the target (red dot)
        elif hasattr(key, "char") and key.char == "m":
            if not auto_mode:  # Only allow moving target when not in auto mode
                self.target_movement_mode = not getattr(
                    self, "target_movement_mode", False
                )
                if self.target_movement_mode:
                    rospy.loginfo(
                        "TARGET MOVEMENT MODE - Use WASD/RF to move the red dot"
                    )
                else:
                    rospy.loginfo("ARM MOVEMENT MODE - WASD/RF controls the arm")

        # Track shift key
        elif key == Key.shift:
            keyboard_control["keys_pressed"].add("shift")

        # Track regular keys
        else:
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

    def create_target_vis(self):
        """Create a visual marker for the target position"""
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.4, 0.0, 0.5]  # Default position for the target

        # Initialize target position in keyboard_control
        global keyboard_control
        keyboard_control["target_pos"] = np.array(basePosition)

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

    def update_target_vis(self, target_pos):
        """Update the visual marker position"""
        global keyboard_control
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
        p.resetBasePositionAndOrientation(
            self.ballMbt[0], target_pos, current_orientation
        )
        # Update the target position in keyboard_control
        keyboard_control["target_pos"] = np.array(target_pos)

    def move_to_target(self):
        """Move the arm to the target position (red dot)"""
        global keyboard_control

        # Get current position and target position
        link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
        current_pos = np.array(link_state[4])
        current_rot = np.array(link_state[5])
        target_pos = keyboard_control["target_pos"]

        # Calculate direction vector to target
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        # If we're close enough to the target, stop
        if distance < 0.01:
            return current_pos, current_rot

        # Normalize direction and move toward target
        if distance > 0:
            direction = direction / distance

            # Determine step size (proportional to distance for smooth approach)
            step_size = min(0.01, distance * 0.1)

            # Calculate new position
            new_pos = current_pos + direction * step_size

            # Use current orientation
            return new_pos, current_rot

        return current_pos, current_rot

    def process_keyboard_control(self):
        """Process keyboard inputs for arm control"""
        global keyboard_control, auto_mode

        # Check if we're in target movement mode
        target_movement_mode = getattr(self, "target_movement_mode", False)

        # If auto mode is enabled, move toward the target
        if auto_mode and not target_movement_mode:
            return self.move_to_target()

        # Speed adjustment for shift key
        fast_mode = "shift" in keyboard_control["keys_pressed"]
        pos_delta = keyboard_control["pos_delta"] * (3.0 if fast_mode else 1.0)
        rot_delta = keyboard_control["rot_delta"] * (3.0 if fast_mode else 1.0)
        keys = keyboard_control["keys_pressed"]

        # If in target movement mode, move the target (red dot) instead of the arm
        if target_movement_mode:
            # Store current target position
            target_pos = keyboard_control["target_pos"].copy()

            # Position control for the target
            if "w" in keys:  # Forward (+X)
                target_pos[0] += pos_delta
            if "s" in keys:  # Backward (-X)
                target_pos[0] -= pos_delta
            if "d" in keys:  # Right (+Y)
                target_pos[1] += pos_delta
            if "a" in keys:  # Left (-Y)
                target_pos[1] -= pos_delta
            if "r" in keys:  # Up (+Z)
                target_pos[2] += pos_delta
            if "f" in keys:  # Down (-Z)
                target_pos[2] -= pos_delta

            # Update the target position and visualization
            self.update_target_vis(target_pos)

            # Return current arm position (don't move the arm in target movement mode)
            link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
            return np.array(link_state[4]), np.array(link_state[5])

        # Normal arm control mode
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

        # Publish to ROS topics for visualization or debugging
        self._publish_joint_commands(jointPoses, arm_pos, arm_rot)

    def _publish_joint_commands(self, jointPoses, arm_pos, arm_rot):
        """Publish joint commands to ROS topics for visualization"""
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

    def _update_control_state_display(self):
        """Update the visual display of control state in simulator"""
        global auto_mode

        # Update control mode text
        if self.control_text_id is not None:
            p.removeUserDebugItem(self.control_text_id)

        # Show different text based on mode
        if not space_pressed:
            control_text = "CONTROL DISABLED"
            text_color = [1, 0, 0]  # Red
        elif auto_mode:
            control_text = "AUTO MODE - MOVING TO TARGET"
            text_color = [0, 0, 1]  # Blue
        elif getattr(self, "target_movement_mode", False):
            control_text = "TARGET MOVEMENT MODE"
            text_color = [1, 0.5, 0]  # Orange
        else:
            control_text = "CONTROL ENABLED"
            text_color = [0, 1, 0]  # Green

        self.control_text_id = p.addUserDebugText(
            control_text,
            [0, 0, 0.8],  # Position above the robot
            textColorRGB=text_color,
            textSize=1.5,
        )

        # Update distance text when in auto mode
        # First, check if distance_text_id exists and remove it if it does
        if hasattr(self, "distance_text_id") and self.distance_text_id is not None:
            p.removeUserDebugItem(self.distance_text_id)

        # Add new distance text if in auto mode
        if auto_mode:
            link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
            current_pos = np.array(link_state[4])
            target_pos = keyboard_control["target_pos"]
            distance = np.linalg.norm(target_pos - current_pos)

            distance_text = f"Distance to target: {distance:.3f}m"
            self.distance_text_id = p.addUserDebugText(
                distance_text,
                [0, 0, 0.75],  # Position above the robot
                textColorRGB=[1, 1, 1],  # White
                textSize=1.2,
            )
        # If not in auto mode, set distance_text_id to None
        else:
            self.distance_text_id = None

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(self.update_rate)
        VRP0 = None  # VR position origin
        VRR0 = None  # VR rotation origin
        MJP0 = None  # Robot position origin
        MJR0 = None  # Robot rotation origin

        # Initialize keyboard control with current robot position
        link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
        keyboard_control["current_pos"] = np.array(link_state[4])
        keyboard_control["current_rot"] = np.array(link_state[5])

        # Initialize UI elements
        self.target_movement_mode = False
        self.control_text_id = None
        self.distance_text_id = None

        # Show control instructions based on available methods
        rospy.loginfo("Arm teleoperation node running. Commands:")
        rospy.loginfo("- SPACE: Toggle control on/off")
        if not self.oculus_reader:
            rospy.loginfo("- T: Toggle auto mode (move to red dot)")
            rospy.loginfo("- M: Toggle target movement mode (move red dot)")
            rospy.loginfo("- WASD/RF: Move in X/Y/Z directions")
            rospy.loginfo("- QEIJKL: Control orientation")
            rospy.loginfo("- SHIFT: Hold for faster movement")
        else:
            rospy.loginfo("- Using Oculus controller for arm control")
            rospy.loginfo("  (Oculus controller trigger toggles control)")

        while not rospy.is_shutdown():
            # Update visual feedback for control state
            self._update_control_state_display()

            # Process control inputs
            if self.oculus_reader:
                VRP0, VRR0, MJP0, MJR0 = self._process_oculus_control(
                    VRP0, VRR0, MJP0, MJR0
                )
            else:
                self._process_keyboard_control()

            rate.sleep()

    def _process_oculus_control(self, VRP0, VRR0, MJP0, MJR0):
        """Process control inputs from Oculus controller"""
        joint_pos = self.oculus_reader.get_joint_transformations()[1]
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()

        if transformations and "r" in transformations:
            right_controller_pose = transformations["r"]
            VRpos, VRquat = vrfront2mj(right_controller_pose)

            if space_pressed:
                # Initialize origins if this is the first frame after engaging control
                if (
                    self.need_init_origins
                    or VRP0 is None
                    or VRR0 is None
                    or MJP0 is None
                    or MJR0 is None
                ):
                    link_state = p.getLinkState(
                        self.kinovaId, self.kinovaEndEffectorIndex
                    )
                    MJP0 = link_state[4]  # robot position origin
                    MJR0 = link_state[5]  # robot orientation origin
                    VRP0 = VRpos.copy()  # VR position origin
                    VRR0 = VRquat.copy()  # VR orientation origin
                    self.need_init_origins = False
                    rospy.loginfo("Control ENGAGED - origins initialized")

                # Calculate position and rotation changes
                dVRP = VRpos - VRP0
                dVRR = diffQuat(VRR0, VRquat)

                # Apply changes to robot pose
                curr_pos = MJP0 + dVRP
                curr_quat = mulQuat(MJR0, dVRR)
            else:
                # Reset the need_init_origins flag when control is disengaged
                self.need_init_origins = True

                # When not in control mode, update reference positions
                link_state = p.getLinkState(self.kinovaId, self.kinovaEndEffectorIndex)
                curr_pos = link_state[4]
                curr_quat = link_state[5]

            # Apply to robot
            self.compute_IK(curr_pos, curr_quat)
            self.update_target_vis(curr_pos)

            return VRP0, VRR0, MJP0, MJR0

        return VRP0, VRR0, MJP0, MJR0

    def _process_keyboard_control(self):
        """Process keyboard control inputs"""
        if space_pressed:
            target_pos, target_quat = self.process_keyboard_control()
            self.compute_IK(target_pos, target_quat)
            self.update_target_vis(target_pos)
        else:
            # When not in control mode, update keyboard position to match robot
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
