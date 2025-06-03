#!/usr/bin/env python3
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from uuid import uuid4 
os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
import numpy as np
from pathlib import Path
import rerun as rr
from scipy.spatial.transform import Rotation
import glob
import h5py
import json
from common import h5_tree, log_angle_rot, blueprint_row_images, log_cartesian_velocity, link_to_world_transform
from rerun_loader_urdf import URDFLogger

POS_DIM_NAMES = ["x", "y", "z", "w", "q1", "q2", "q3"]
CAMERA_NAMES = ['rgb:right_cam:480x640:2d', 'rgb:wrist_cam:480x640:2d']

def extract_extrinsics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Takes a vector with dimension 6 and extracts the translation vector and the rotation matrix"""
    translation = pose[:3]
    rotation = Rotation.from_quat(np.array(pose[3:-1])[[1, 2, 3, 0]]).as_matrix()
    gripper = pose[-1:]
    return (translation, rotation, gripper)

class RawScene:
    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, np.ndarray]

    def __init__(self):
        rr.init("PandaIIT", spawn=True)
        rr.send_blueprint(blueprint_raw())
        
        # make the list of dictionaries into a dictionary of stacked arrays

    def reset(self):
        stream = rr.new_recording('PandaIIT', make_default=True, recording_id=uuid4())
        stream.connect()

        self.urdf_logger = URDFLogger(Path('/home/teleop/project/teleop/franka_description/panda.urdf'))
        self.urdf_logger.log()


    def log_cameras_next(self, step: dict) -> None:
        """
        Log data from cameras at step `i`.
        It should be noted that it logs the next camera frames that haven't been 
        read yet, this means that this method must only be called once for each step 
        and it must be called in order (log_cameras_next(0), log_cameras_next(1)). 

        The motivation behind this is to avoid storing all the frames in a `list` because
        that would take up too much memory.
        """

        rr.log(f"cameras/agentview_image", rr.Image(step['obs/agentview_image']))
        # rr.log(f"cameras/eye_in_hand_image", rr.Image(step['obs/eye_in_hand_image']))

    def log_action(self, step: dict) -> None:
        pose = step['actions']
        trans, mat, gripper = extract_extrinsics(pose)

        # mat = mat * Rotation.from_euler('y', 180, degrees=True).as_matrix()
        rr.log('action/cartesian_position/transform', rr.Transform3D(translation=trans, mat3x3=mat))
        rr.log('action/cartesian_position/origin', rr.Points3D([trans], radii=[10]))
        rr.log('action/target_gripper_position', rr.Scalar(gripper))

        for j, dim_name in enumerate(POS_DIM_NAMES):
            rr.log(f'cartesian/{dim_name}/action', rr.Scalar(pose[j]))
        
    def log_robot_state(self, step: dict, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        
        joint_angles = step['obs/arm_qpos']
        for joint_idx, angle in enumerate(joint_angles):
            log_angle_rot(entity_to_transform, joint_idx + 1, angle)
        
        rr.log('robot_state/gripper_position', rr.Scalar(step['obs/gripper_qpos']))

        for j, pos in enumerate(step['obs/arm_qpos']):
            rr.log(f"robot_state/joint_position/{j}", rr.Scalar(pos))

        trans = step['obs/eef_pos']
        mat = step['obs/eef_rot'].reshape(3, 3)
        quat = Rotation.from_matrix(mat).as_quat(canonical=True)[[-1, 0, 1, 2]]

        rr.log('robot_state/cartesian_position/transform', rr.Transform3D(translation=trans, mat3x3=mat))

        cartesian = np.concatenate([trans, quat])
        for j, dim_name in enumerate(POS_DIM_NAMES):
            rr.log(f'cartesian/{dim_name}/robot_state', rr.Scalar(cartesian[j]))


    def log(self, step) -> None:
        # time_stamps_nanos = self.trajectory['observation']['time']
        nano = (step['obs/time'] * 1e9).astype(int)
        rr.set_time_nanos("real_time", nano.item())

        self.log_action(step)
        self.log_cameras_next(step)
        self.log_robot_state(step, self.urdf_logger.entity_to_transform)

    def save(self):
        rr.save('/home/teleop/project/data/rerun/action_2_new.rrd')

def blueprint_raw():
    from rerun.blueprint import (
        Blueprint,
        BlueprintPanel,
        Horizontal,
        Vertical,
        SelectionPanel,
        Spatial3DView,
        TimePanel,
        TimeSeriesView,
        Tabs,
    )

    blueprint = Blueprint(
        Horizontal(
            Vertical(
                Spatial3DView(name="robot view", origin="/", contents=["/**"]),
                blueprint_row_images(
                    [
                        f"cameras/agentview_image",
                        f"cameras/eye_in_hand_image",
                    ]
                ),
                row_shares=[3, 1, 1],
            ),
            Tabs(
                Vertical(
                    TimeSeriesView(origin=f'robot_state/gripper_position', contents=['robot_state/gripper_position', 'action/target_gripper_position' ]), 
                    name='obs/gripper'
                ),
                 Vertical(
                    TimeSeriesView(origin='jump_trans'),
                    TimeSeriesView(origin='jump_rot'),
                    *(
                        TimeSeriesView(origin=f'cartesian/{dim_name}') for dim_name in POS_DIM_NAMES
                    ),
                    name='cartesian_position',
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_position/{i}') for i in range(7)
                    ),
                    name='joint_position',
                ),
                Vertical(
                    TimeSeriesView(origin='action/', contents=['action/gripper_position', 'action/target_gripper_position']),
                    TimeSeriesView(origin='action/gripper_velocity'),
                    name='action/gripper' 
                ),
               
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_velocities/{i}') for i in range(7)
                    ),
                    name='robot_state/joint_velocities'
                ),
            ),
            column_shares=[3, 2],
        ),
        BlueprintPanel(expanded=False),
        SelectionPanel(expanded=False),
        TimePanel(expanded=False),
        auto_space_views=False,
    )
    return blueprint


