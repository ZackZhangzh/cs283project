from multiprocessing import Process
from multiprocessing.managers import BaseManager
import torch
import cv2
import dill
import numpy as np
import time
import hydra

from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from multiprocessing import Queue

class Policy(Process):

    def __init__(self) -> None:
        super().__init__()
        checkpoint = 'data/eightyK.ckpt'
        device = 'cuda:0'

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir='data')
        workspace.load_payload(payload)

        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        self.policy = policy
        self.past_obs = None
        self.action_count = 0

        BaseManager.register('get_queue')
        manager = BaseManager(address=('127.0.0.1', 12345), authkey=b'abracadabra')

        print('Connecting to manager...')
        start = time.time()

        while True:
            try:
                manager.connect()
                break
            except ConnectionRefusedError as e:
                if time.time() - start > 120:
                    print('Connection refused.')
                    raise e
                time.sleep(1)
        print('Connected to manager.')

        self.obs_queue = manager.get_queue('observations')
        self.action_queue = manager.get_queue('actions')

    def run(self):
        while True:
            print('Waiting for observation')
            data = self.obs_queue.get(block=True)
            obs_id, obs = data['obs_id'], data['obs']
            print(f'Observation {obs_id} received')

            if self.past_obs is None:
                self.past_obs = obs

            model_input = {}

            agentview_image = torch.stack([
                torch.tensor(cv2.resize(self.past_obs['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80], (84, 84))).permute(-1, 0, 1) / 255,
                torch.tensor(cv2.resize(obs['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80], (84, 84))).permute(-1, 0, 1) / 255
            ])

            # eye_in_hand_image = torch.stack([
            #     torch.tensor(cv2.resize(self.past_obs['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (84, 84))).permute(-1, 0, 1) / 255,
            #     torch.tensor(cv2.resize(obs['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (84, 84))).permute(-1, 0, 1) / 255
            # ])

            model_input['agentview_image'] = agentview_image
            # model_input['eye_in_hand_image'] = eye_in_hand_image
            model_input['eef_pos'] = np.stack([self.past_obs['obs_dict']['pos_ee'], obs['obs_dict']['pos_ee']])

            eef_quat = torch.stack([self.mat_to_quat.forward(torch.tensor(self.past_obs['obs_dict']['rot_ee'].reshape(3,3)[None]))[0],
                                    self.mat_to_quat.forward(torch.tensor(obs['obs_dict']['rot_ee'].reshape(3,3)[None]))[0]])
            
            model_input['eef_quat'] = eef_quat
            model_input['gripper_qpos'] = torch.stack([torch.tensor(self.past_obs['obs_dict']['qp_ee'][None]), torch.tensor(obs['obs_dict']['qp_ee'][None])])

            # udpate desired pos
            with torch.no_grad():
                model_input = {obs: model_input[obs][None] for obs in model_input}
                start_inference = time.perf_counter()
                action_chunk = self.policy.predict_action(model_input)['action']
                print(f"Inference time: {time.perf_counter()-start_inference}")
            action_chunk = action_chunk[0]

            action_chunk = np.concatenate([
                action_chunk[:, :3].cpu().numpy(),
                self.six_to_quat.forward(torch.tensor(action_chunk[:, 3:9])[None])[0].cpu().detach().numpy(),
                action_chunk[:, 9:10].cpu().numpy()
            ], axis=1)

            print(f'Sending actions: {list(range(obs_id, obs_id+8))}')
            self.action_queue.put(dict(zip(range(obs_id, obs_id+8), action_chunk)))
            print('Action sent')

if __name__ == '__main__':
    policy = Policy()
    policy.run()    