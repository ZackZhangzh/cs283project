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
import threading
import math
import actionlib

"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""

space_pressed = False

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

camera_base_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

leap2human_rot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

correction_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

reflection_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

Rx_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


# Convert 3*3 matrix to quaternion
def mat2quat(mat):
    q = np.zeros([4])
    q[3] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[0] = (mat[2][1] - mat[1][2]) / (4 * q[3])
    q[1] = (mat[0][2] - mat[2][0]) / (4 * q[3])
    q[2] = (mat[1][0] - mat[0][1]) / (4 * q[3])
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
    pos = camera_base_rot @ pose[:3, 3]
    mat = (
        reflection_rot
        @ (
            Rx_180
            @ (
                correction_rot
                @ (leap2human_rot @ (camera_base_rot @ pose[:3, :3]))
                @ correction_rot
            )
        )
        @ reflection_rot
    )
    q = mat2quat(mat)

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


class KinovaNode:
    def __init__(self):
        rospy.init_node("teleoperation_node", anonymous=True)

        # 运动链初始化
        urdf_path = "../kinova-ros/kinova_description/urdf/robot.urdf"

        # 关节配置
        self.controlled_joints = [
            "j2n6s300_joint_1",
            "j2n6s300_joint_2",
            "j2n6s300_joint_3",
            "j2n6s300_joint_4",
            "j2n6s300_joint_5",
            "j2n6s300_joint_6",
        ]

        # ROS通信
        self.joint_state_sub = rospy.Subscriber(
            "/j2n6s300_driver/out/joint_state",
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )
        self.joint_state_pub = rospy.Publisher(
            "/teleoperation_joint_state", JointState, queue_size=1
        )
        self.joint_state = JointState()
        self.joint_state.name = self.controlled_joints

        # 状态变量
        self.current_q = None
        self.ik_solver = None
        # result = self._publish_joint_command(torch.tensor([math.pi/2, 3.1415927,3.1415927,0, 0, 0]).float().cuda())
        # print(result)
        # wait 5 sec
        rospy.sleep(3)
        print("init success")
        # init robot arm joint qpos
        # get current qpos , set goal qpos [0, 3.1415927,3.1415927,0, 0, 0]

        self.rospy_thread = threading.Thread(target=rospy.spin)
        self.rospy_thread.start()

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        q_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        self.current_qpos = q_dict
        return

    def publish_joint_command(self, q_target):
        """发布关节指令"""
        pi = math.pi
        self.joint_state.position = [
            q_target[0] * 360 / (2 * pi),
            q_target[1] * 360 / (2 * pi),
            q_target[2] * 360 / (2 * pi),
            q_target[3] * 360 / (2 * pi),
            q_target[4] * 360 / (2 * pi),
            q_target[5] * 360 / (2 * pi),
        ]

        rospy.loginfo(f"发布关节指令: {q_target}")
        self.joint_state_pub.publish(self.joint_state)


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
        return self.dxl_client.read_pos()  # 16dof

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class SystemPybulletIK:
    def __init__(self):
        # start pybullet
        p.connect(p.GUI)
        # load right leap hand
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [4, 9, 14, 19]
        self.kinovaEndEffectorIndex = 9
        self.kinova_node = KinovaNode()
        kinova_path_src = "/home/zhihao/cs283project/src/kinova-ros/kinova_description/urdf/robot.urdf"
        self.kinovaId = p.loadURDF(
            kinova_path_src,
            [0.5, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        leap_path_src = os.path.join(
            path_src, "leap_hand_mesh_right/robot_pybullet.urdf"
        )
        self.leapId = p.loadURDF(
            leap_path_src,
            [0.0, 0.038, 0.098],
            p.getQuaternionFromEuler([0, -1.57, 0]),
            useFixedBase=True,
        )

        self.numJoints = p.getNumJoints(self.kinovaId)
        self.leapnumJoints = p.getNumJoints(self.leapId)
        for i in range(2, 8):
            p.resetJointState(self.kinovaId, i, np.pi)

        for i in range(0, self.numJoints):
            print(p.getJointInfo(self.kinovaId, i))

        for i in range(0, self.leapnumJoints):
            print(p.getJointInfo(self.leapId, i))

        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        # self.create_target_vis()

        self.operator2mano = OPERATOR2MANO_RIGHT
        self.leap_node = LeapNode()
        self.oculus_reader = OculusReader()
        self.joint_names = [
            "Index1",
            "Index2",
            "Index3",
            "IndexTip",
            "Middle1",
            "Middle2",
            "Middle3",
            "MiddleTip",
            "Ring1",
            "Ring2",
            "Ring3",
            "RingTip",
            "Thumb1",
            "Thumb2",
            "Thumb3",
            "ThumbTip",
        ]

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
        for i in range(0, 16):
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
            p.changeVisualShape(self.ballMbt[i], -1, rgbaColor=[1, 0, 0, 1])

    def update_target_vis(self, hand_pos):
        for i in range(len(hand_pos)):
            _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[i])
            p.resetBasePositionAndOrientation(
                self.ballMbt[i], hand_pos[i], current_orientation
            )

    def compute_IK(self, hand_pos, rot, target_pos, target_quat):
        p.stepSimulation()

        index_mcp_pos = hand_pos[0]
        index_pip_pos = hand_pos[1]
        index_dip_pos = hand_pos[2]
        index_tip_pos = hand_pos[3]
        middle_mcp_pos = hand_pos[4]
        middle_pip_pos = hand_pos[5]
        middle_dip_pos = hand_pos[6]
        middle_tip_pos = hand_pos[7]
        ring_mcp_pos = hand_pos[8]
        ring_pip_pos = hand_pos[9]
        ring_dip_pos = hand_pos[10]
        ring_tip_pos = hand_pos[11]
        thumb_mcp_pos = hand_pos[12]
        thumb_pip_pos = hand_pos[13]
        thumb_dip_pos = hand_pos[14]
        thumb_tip_pos = hand_pos[15]

        index_mcp_rot = rot[0]
        index_pip_rot = rot[1]
        index_dip_rot = rot[2]
        index_tip_rot = rot[3]
        middle_mcp_rot = rot[4]
        middle_pip_rot = rot[5]
        middle_dip_rot = rot[6]
        middle_tip_rot = rot[7]
        ring_mcp_rot = rot[8]
        ring_pip_rot = rot[9]
        ring_dip_rot = rot[10]
        ring_tip_rot = rot[11]
        thumb_mcp_rot = rot[12]
        thumb_pip_rot = rot[13]
        thumb_dip_rot = rot[14]
        thumb_tip_rot = rot[15]

        leapEndEffectorPos = [
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            thumb_tip_pos,
        ]

        leapEndEffectorRot = [
            index_tip_rot,
            middle_tip_rot,
            ring_tip_rot,
            thumb_tip_rot,
        ]

        leap_jointPoses = p.calculateInverseKinematics2(
            self.leapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            leapEndEffectorRot,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        kinova_jointPoses = p.calculateInverseKinematics(
            self.kinovaId,
            self.kinovaEndEffectorIndex,
            target_pos,
            target_quat,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        arm_qpos_now = []
        for i in range(2, 8):
            arm_qpos_now.append(p.getJointState(self.kinovaId, i)[0])
        arm_qpos_now = tuple(arm_qpos_now)

        if any(math.isnan(pose) for pose in kinova_jointPoses):
            print("IK solution is None")
            kinova_jointPoses = arm_qpos_now
        else:
            kinova_jointPoses = kinova_jointPoses[:6]
            # qpos_arm_err = np.linalg.norm(jointPoses - qpos_now)
            sum = 0.0
            for i in range(6):
                sum += (kinova_jointPoses[i] - arm_qpos_now[i]) * (
                    kinova_jointPoses[i] - arm_qpos_now[i]
                )
            qpos_arm_err = math.sqrt(sum)

            # if qpos_arm_err > 0.5:
            #     print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
            #     kinova_jointPoses = arm_qpos_now

        combined_jointPoses = (
            (0.0,)
            + (0.0,)
            + kinova_jointPoses[0:6]
            + (0.0,)
            + (0.0,)
            + leap_jointPoses[0:4]
            + (0.0,)
            + leap_jointPoses[4:8]
            + (0.0,)
            + leap_jointPoses[8:12]
            + (0.0,)
            + leap_jointPoses[12:16]
            + (0.0,)
        )
        combined_jointPoses = list(combined_jointPoses)
        leap_whole_jointPoses = (
            leap_jointPoses[0:4]
            + (0.0,)
            + leap_jointPoses[4:8]
            + (0.0,)
            + leap_jointPoses[8:12]
            + (0.0,)
            + leap_jointPoses[12:16]
            + (0.0,)
        )
        leap_whole_jointPoses = list(leap_whole_jointPoses)
        # update the hand joints
        for i in range(0, 6):
            p.setJointMotorControl2(
                bodyIndex=self.kinovaId,
                jointIndex=i + 2,
                controlMode=p.POSITION_CONTROL,
                targetPosition=kinova_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        for i in range(10, 30):
            p.setJointMotorControl2(
                bodyIndex=self.kinovaId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        for i in range(self.leapnumJoints):
            p.setJointMotorControl2(
                bodyIndex=self.leapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=leap_whole_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        real_robot_hand_q[0:4] = leap_jointPoses[0:4]
        real_robot_hand_q[4:8] = leap_jointPoses[4:8]
        real_robot_hand_q[8:12] = leap_jointPoses[8:12]
        real_robot_hand_q[12:16] = leap_jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        self.leap_node.set_allegro(real_robot_hand_q)  # gripper_state]

        print("target_pos:", target_pos)
        print("target_quat:", target_quat)
        print("kinova_jointPoses:", kinova_jointPoses)
        self.kinova_node.publish_joint_command(kinova_jointPoses)

    def operation(self):
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None
        while True:
            joint_pos = self.oculus_reader.get_joint_transformations()[1]
            transformations, buttons = (
                self.oculus_reader.get_transformations_and_buttons()
            )
            gripper_state = []
            gripper_state[len(self.motors)] = 1
            # self.end_effector_retargeting(transformations)
            # kinova
            if transformations and "r" in transformations:
                right_controller_pose = transformations["r"]
                # VRpos, VRquat = vrfront2mj(right_controller_pose) # front x, left y, up z
                VRpos, VRquat = vrbehind2mj(
                    right_controller_pose
                )  # front -x, left -y, up z
                if space_pressed:
                    # dVRP/R = VRP/Rt - VRP/R0
                    dVRP = VRpos - VRP0
                    # dVRR = VRquat - VRR0
                    dVRR = diffQuat(VRR0, VRquat)
                    # MJP/Rt =  MJP/R0 + dVRP/R

                    curr_pos = MJP0 + dVRP
                    # curr_quat = mulQuat(MJR0, dVRR)
                    curr_quat = VRquat

                # Adjust origin if not engaged
                else:
                    # ros spin once
                    kinova_joint_state = self.kinova_node.current_qpos  # arm qpose
                    p.resetJointState(
                        self.kinovaId, 2, kinova_joint_state["j2n6s300_joint_1"]
                    )
                    p.resetJointState(
                        self.kinovaId, 3, kinova_joint_state["j2n6s300_joint_2"]
                    )
                    p.resetJointState(
                        self.kinovaId, 4, kinova_joint_state["j2n6s300_joint_3"]
                    )
                    p.resetJointState(
                        self.kinovaId, 5, kinova_joint_state["j2n6s300_joint_4"]
                    )
                    p.resetJointState(
                        self.kinovaId, 6, kinova_joint_state["j2n6s300_joint_5"]
                    )
                    p.resetJointState(
                        self.kinovaId, 7, kinova_joint_state["j2n6s300_joint_6"]
                    )

                    # RP/R0 = RP/Rt
                    link_state = p.getLinkState(self.kinovaId, 9)

                    curr_pos = link_state[4]
                    curr_quat = link_state[5]
                    # 当前 q pose

                    MJP0 = curr_pos  # real kinova pos origin
                    MJR0 = curr_quat  # real kinova quat origin

                    # VP/R0 = VP/Rt
                    VRP0 = VRpos
                    VRR0 = VRquat

                # udpate desired pos
                target_pos = curr_pos
                target_quat = VRquat

                # leaphand
                pos = []
                final_pos = []
                rot = []
                if joint_pos == {}:
                    # print("None")
                    continue
                else:
                    # print("Receive info")
                    mediapipe_wrist_rot = joint_pos["WristRoot"][:3, :3]
                    wrist_position = joint_pos["WristRoot"][:3, 3]
                    # wrist_position[2] += 0.05

                    for joint_name in self.joint_names:
                        joint_transformation = joint_pos[joint_name]

                        pos.append(joint_transformation[:3, 3] - wrist_position)
                        if joint_name == "MiddleTip":
                            pos[-1][0] += 0.009375
                        if joint_name == "RingTip":
                            pos[-1][0] -= 0.004375
                        if joint_name == "IndexTip":
                            pos[-1][0] -= 0.00125
                        if joint_name == "ThumbTip":
                            pos[-1][0] += 0.00375
                        pos[-1] = pos[-1] @ mediapipe_wrist_rot @ self.operator2mano

                        # Turn the rotation matrix into quaternion
                        rotation = (
                            joint_transformation[:3, :3]
                            @ mediapipe_wrist_rot
                            @ self.operator2mano
                        )
                        quaternion = R.from_matrix(rotation).as_quat()
                        rot.append(quaternion)

                        final_pos.append(
                            [
                                pos[-1][0] * self.glove_to_leap_mapping_scale * 1.15,
                                pos[-1][1] * self.glove_to_leap_mapping_scale,
                                pos[-1][2] * self.glove_to_leap_mapping_scale,
                            ]
                        )
                        final_pos[-1][2] -= 0.05

                self.compute_IK(final_pos, rot, target_pos, target_quat)
                # target_pos, target_quat
                # self.update_target_vis(final_pos)


from teleop.visualizer import RawScene
import h5py
import cv2

visualizer = RawScene()


def recor_depisode_data(
    episode_data,
    env_info,
    target_pos,
    target_quat,
    gripper_state,
    buttons,
    transformations,
    recording,
    episode_ended,
):
    if (
        transformations
        and "r" in transformations
        and buttons["RG"]
        and not episode_ended
        and recording
    ):

        # Traget data
        episode_data["actions"] += [
            np.concatenate([target_pos, target_quat, [gripper_state]])
        ]
        episode_data["dones"] += [False]
        episode_data["rewards"] += [False]
        # Current data
        episode_data["obs/time"] += [env_info["obs_dict"]["time"]]
        episode_data["obs/arm_qpos"] += [env_info["obs_dict"]["qp_arm"]]
        episode_data["obs/eef_pos"] += [env_info["obs_dict"]["pos_ee"]]
        episode_data["obs/eef_rot"] += [env_info["obs_dict"]["rot_ee"]]
        episode_data["obs/gripper_qpos"] += [env_info["obs_dict"]["qp_ee"]]  #

        primary = cv2.resize(
            env_info["visual_dict"]["rgb:right_cam:480x640:2d"][:, 80:-80], (92, 92)
        )
        # wrist = cv2.resize(env_info['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (92, 92))
        episode_data["obs/agentview_image"] += [primary]
        # episode_data['obs/eye_in_hand_image'] += [wrist]

        visualizer.log({k: v[-1] for k, v in episode_data.items()})

    cv2.imshow(
        "primary",
        env_info["visual_dict"]["rgb:right_cam:480x640:2d"][:, 80:-80][..., ::-1],
    )
    cv2.waitKey(1)


def on_press(key):
    global space_pressed
    if key == Key.space:
        space_pressed = True


# 键盘释放回调
def on_release(key):
    global space_pressed
    if key == Key.space:
        space_pressed = False


def main(args=None):
    keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    leappybulletik = SystemPybulletIK()
    leappybulletik.operation()


if __name__ == "__main__":
    main()
