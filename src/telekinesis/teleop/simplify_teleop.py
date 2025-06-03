import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/robohive')

from robohive.utils.quat_math import euler2quat, euler2mat, mat2quat, diffQuat, mulQuat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.robot import robot
import my_env

from pynput import keyboard
import time
import numpy as np
from scipy.spatial.transform import Rotation
import click
import gym
import signal

try:
    from oculus_reader import OculusReader
except ImportError as e:
    raise ImportError("(Missing oculus_reader. HINT: Install and perform the setup instructions from https://github.com/rail-berkeley/oculus_reader)")

flag = False


def main():
    global flag
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()


    # use oculus as input 
    oculus_reader = OculusReader()
    oculus_reader_ready = False

    while not oculus_reader_ready:
        # Get the controller and headset positions and the button being pushed
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        if transformations or buttons:
            oculus_reader_ready = True
        else:
            print("Oculus reader not ready. Check that headset is awake and controller are on")
        time.sleep(0.10)
    print('Oculus Ready!')




    env_name = 'rpFrankaRobotiqData-v1'
    # seed and load environments
    env_args = {'is_hardware': True, 
                'config_path': './my_env/franka_robotiq.config',
                # 'frame_skip': 50,
                # 'model_path': '/franka_robotiq.xml', 
                # 'target_pose': np.array([0, 0, 0, 0, 0, 0, 0, 0])
                }
    
    # visualizer = RawScene()
    # time.sleep(10)
    seed = 123
    np.random.seed(seed)
    env = gym.make(env_name, **env_args)
    env.seed(seed)
    env.env.mujoco_render_frames = False

    def close():
        env.close()
    
    signal.signal(signal.SIGINT, close)

    env.reset(blocking=False)

    num = 0
    # act[0] matters, 0 for close, 1 for open
    act = np.arange(8)
    while True:
        # read button
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        # print(transformations, buttons)

        if buttons['RG'] or buttons['rightTrig'][0] != 0:
            print("Begin!")
            flag = True
            if buttons['rightTrig'][0] != 0:
                print ('act close')
                act[0] = 0
            else:
                print('act open')
                act[0] = 1

        if flag:
            num = num + 1
            flag = False
            print("start triggering the gripper")
            env.step(act)
            print("finish num", num)

        # if num > 0:
        #     break
            # flag = False
        time.sleep(0.1)


    env.close()


if __name__ == '__main__':
    main()


