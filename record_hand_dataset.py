import pyk4a
from pyk4a import Config, PyK4A
import cv2
import numpy as np
import os   
import time
import imageio
# trj_data_name = f'k4a_hand_dataset_{time.time()}'
trj_data_name = 'arrange_bottle_dataset_30'
if __name__ == "__main__":
    # K = np.loadtxt('cam_K.txt').reshape(3,3)
    # k4a init
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    calibration = k4a.calibration

    K = calibration.get_camera_matrix(1) # stand for color type
    window_name = 'k4a'
    start_trigger = False
    annotation = False
    first_tracking_frame = False
    index = 0
    zfar = 2.0
    first_downscale = True
    shorter_side = 720
    recording = True
    first_recording = True

    dataset = []

    while recording :
        capture = k4a.get_capture()

        if first_downscale:
            H, W = capture.color.shape[:2]
            downscale = shorter_side / min(H, W)
            H = int(H*downscale)
            W = int(W*downscale)
            K[:2] *= downscale
            first_downscale = False        
     
     
        color = capture.color[...,:3].astype(np.uint8)
        color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST) 
        depth = capture.transformed_depth.astype(np.float32) / 1e3
        depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.01) | (depth>=zfar)] = 0

        data = {
            'color': color,
            'depth': depth,
            'index': index
        }
        index += 1
        dataset.append(data)
        cv2.imshow("k4a", color)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            recording = False
        time.sleep(1 / 10.0)
    
    k4a.stop()


    import yaml
    for data in dataset:
        # color save as png
        # depth save as csv
        # collor depth use timestamp to name
        #color depeth save in a folder
        index = data.get('index')
        color = data.get('color')
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = data.get('depth')
        pose = data.get('pose')
        color_path = f'./{trj_data_name}/color/{index}.png'
        depth_path = f'./{trj_data_name}/depth/{index}.csv'
        os.makedirs(f'./{trj_data_name}/color', exist_ok=True)
        os.makedirs(f'./{trj_data_name}/depth', exist_ok=True)
        imageio.imwrite(color_path, color)
        np.savetxt(depth_path, depth, delimiter=',')
        color_path = f'/color/{index}.png'
        depth_path = f'/depth/{index}.csv'
        data['color'] = color_path
        data['depth'] = depth_path
        data['index'] = index

    with open(f'./{trj_data_name}/hand_data.yaml', 'w') as f:
        yaml.dump(dataset, f)   