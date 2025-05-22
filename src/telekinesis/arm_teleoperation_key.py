#!/usr/bin/env python3
import pybullet as p
import numpy as np
import os
import time

from oculus_reader.scripts import *
from oculus_reader.scripts.reader import OculusReader

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
import sys

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

# import keyboard
from pynput.keyboard import Key, Listener
import rospy
import math

"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""

# Changed from space_pressed to control_engaged to reflect the toggle behavior
control_engaged = False

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
    # mat[0][:] = +1.*pose[2][:3]
    # mat[1][:] = -1.*pose[0][:3]
    # mat[2][:] = -1.*pose[1][:3]
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
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class KinovaPybulletIK:
    def __init__(self):
        # start pybullet
        # clid = p.connect(p.SHARED_MEMORY)
        # clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # load right leap hand
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.kinovaEndEffectorIndex = 9

        # path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
        path_src = "../kinova-ros/kinova_description/urdf/robot.urdf"
        ##You may have to set this path for your setup on ROS2
        self.kinovaId = p.loadURDF(
            path_src,
            # [-0.05, -0.03, -0.125],
            # [0.0,0.038,0.098],
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
            print("Joint {}: {}".format(i, info[1].decode("utf-8")))

            # print(info)

        self.operator2mano = OPERATOR2MANO_RIGHT
        # self.leap_node = LeapNode()
        try:
            self.oculus_reader = OculusReader()
            print("OculusReader initialized:", self.oculus_reader)
            self.use_oculus = True
        except Exception as e:
            print("Failed to initialize OculusReader, entering Keyboard control mode. Error:", e)
            self.use_oculus = False

    def update_target_vis(self, hand_pos):
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

    def compute_IK(self, arm_pos, arm_rot):
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
            print("IK solution is None")
            jointPoses = qpos_now
        else:
            jointPoses = np.array(jointPoses[:6])
            # qpos_arm_err = np.linalg.norm(jointPoses - qpos_now)
            sum = 0.0
            for i in range(6):
                sum += (jointPoses[i] - qpos_now[i]) * (jointPoses[i] - qpos_now[i])
            qpos_arm_err = math.sqrt(sum)

            if qpos_arm_err > 0.5:
                print(
                    "Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(
                        qpos_arm_err
                    )
                )
                jointPoses = qpos_now

        # update the hand joints
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

    def operation(self):
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None
        # Flag to track if we need to initialize origins when control is engaged
        need_init_origins = True

        while True:
            joint_pos = self.oculus_reader.get_joint_transformations()[1]
            transformations, buttons = (
                self.oculus_reader.get_transformations_and_buttons()
            )

            # kinova
            if transformations and "r" in transformations:
                print("transformations", transformations)

                right_controller_pose = transformations["r"]
                VRpos, VRquat = vrfront2mj(
                    right_controller_pose
                )  # front x, left y, up z
                # VRpos, VRquat = vrbehind2mj(right_controller_pose) # front -x, left -y, up z
                # VRpos, VRquat = vr2mj(right_controller_pose)

                # Use control_engaged state instead of space_pressed
                if control_engaged:
                    # Initialize origins if this is the first frame after engaging control
                    if need_init_origins:
                        # Store the initial positions
                        link_state = p.getLinkState(self.kinovaId, 9)
                        MJP0 = link_state[4]  # real kinova pos origin
                        MJR0 = link_state[5]  # real kinova quat origin
                        VRP0 = VRpos
                        VRR0 = VRquat
                        need_init_origins = False
                        print("Control ENGAGED - origins initialized")

                    # Calculate relative movement
                    dVRP = VRpos - VRP0
                    dVRR = diffQuat(VRR0, VRquat)

                    # Apply movement to initial robot position
                    curr_pos = MJP0 + dVRP
                    curr_quat = mulQuat(MJR0, dVRR)
                # When not engaged, just track current position
                else:
                    # Reset the need_init_origins flag when control is disengaged
                    need_init_origins = True

                    link_state = p.getLinkState(self.kinovaId, 9)
                    curr_pos = link_state[4]
                    curr_quat = link_state[5]

                    if not need_init_origins:  # Only print this once when we disengage
                        print("Control DISENGAGED")
                        need_init_origins = True

                # Update desired pos
                target_pos = curr_pos
                target_quat = curr_quat

                self.compute_IK(target_pos, target_quat)
                self.update_target_vis(target_pos)


def on_press(key):
    # We don't need to do anything on press anymore
    pass


# Modify the key release callback to toggle the control_engaged state
def on_release(key):
    global control_engaged, need_init_origins
    if key == Key.space:
        control_engaged = not control_engaged
        print(f"Control {'ENGAGED' if control_engaged else 'DISENGAGED'}")


def main(args=None):
    keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    kinovapybulletik = KinovaPybulletIK()
    kinovapybulletik.operation()


if __name__ == "__main__":
    main()
