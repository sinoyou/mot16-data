import os
from trajectory_length_calc import *
import numpy as np
import pandas as pd


def frame_squeeze(traj, dataset_fps):
    """
    Given dataset, returns squeezed.
    Time gap between frames is 0.1s.
    """
    # squeeze
    max_frame = np.max(traj[:, 0])
    min_frame = np.min(traj[:, 0])
    frame_gaps = float(dataset_fps * 0.1)
    select_list = np.arange(min_frame, max_frame, frame_gaps)
    result = []
    for frame in select_list:
        frame = int(frame)
        result.append(traj[traj[:, 0] == frame, :])
    return np.concatenate(result, axis=0)

    # Standard
    # 1. 合格轨迹标准：走路行人类 && 轨迹长度大于2秒
    # 2. 格式：每个数据库单独建立，存放于MOT16Filter中，相邻帧之间间隔为0.1s，使用归一化处理。
    # [frame, id, left_x, left_y, width, height]


if __name__ == '__main__':
    print('Datasets info preparing')
    datasets_info = {}
    datasets_info['MOT16-02'] = {'fps': 30, 'width': 1920, 'height': 1080}
    datasets_info['MOT16-04'] = {'fps': 30, 'width': 1920, 'height': 1080}
    datasets_info['MOT16-05'] = {'fps': 14, 'width': 640, 'height': 480}
    datasets_info['MOT16-09'] = {'fps': 30, 'width': 1920, 'height': 1080}
    datasets_info['MOT16-10'] = {'fps': 30, 'width': 1920, 'height': 1080}
    datasets_info['MOT16-11'] = {'fps': 30, 'width': 1920, 'height': 1080}
    datasets_info['MOT16-13'] = {'fps': 25, 'width': 1920, 'height': 1080}

    print('Data Washer Working')
    # directory of data to be washed.
    wash_data_dict = os.path.join('data', 'MOT16Labels', 'train')
    # directory of washed data
    washed_data_dict = os.path.join('data', 'MOT16Filter', 'train')
    if not os.path.exists(washed_data_dict):
        os.mkdir(washed_data_dict)
    all_datasets_list = os.listdir(wash_data_dict)
    for dataset in all_datasets_list:
        if dataset not in datasets_info:
            print('No information about ', dataset)
            continue
        path = os.path.join(wash_data_dict, dataset, 'gt', 'gt.txt')
        dataset_fps = datasets_info[dataset].get('fps')
        dataset_pixel = [datasets_info[dataset].get('width'), datasets_info[dataset].get('height')]

        # get numpy data
        ori_data = pd.read_csv(path).to_numpy()
        # squeeze data to 10 fps
        squeeze_data = frame_squeeze(ori_data, dataset_fps)
        # get time dict and filter trajectory id whose length is over 2 secs
        time_dict = trajectory_length(squeeze_data, 10)
        candidate_traj_list = [k for (k, v) in time_dict.items() if v > 2]

        # data analysis
        max_ped_num_scene = 0
        for frame in np.unique(squeeze_data[:, 0]):
            objects_in_scene = squeeze_data[squeeze_data[:, 0] == frame, 1]
            objects_list = [ped for ped in candidate_traj_list if ped in objects_in_scene]
            max_ped_num_scene = max(max_ped_num_scene, len(objects_list))
        print('Num of Candidate Trajectories:{}, Max pedestrians in one scene:{}'.format(len(candidate_traj_list),
                                                                                         max_ped_num_scene))

        # generate output data
        traj_list = []
        for traj in candidate_traj_list:
            traj_data = squeeze_data[squeeze_data[:, 1] == traj, :]
            # frame normalization
            normal_data = np.zeros((traj_data.shape[0], 6))
            normal_data[:, 0] = traj_data[:, 0]  # frame id
            normal_data[:, 1] = traj_data[:, 1]  # ped id
            normal_data[:, 2] = (traj_data[:, 2] + traj_data[:, 4] / 2) / dataset_pixel[0]  # box's center x
            normal_data[:, 3] = (traj_data[:, 3] + traj_data[:, 5] / 2) / dataset_pixel[1]  # box's center y
            normal_data[:, 4] = traj_data[:, 4] / dataset_pixel[0] / 2  # box's scale x
            normal_data[:, 5] = traj_data[:, 5] / dataset_pixel[1] / 2  # box's scale y
            traj_list.append(normal_data)
        # save data
        dataset_processed = np.concatenate(traj_list, axis=0)
        dataset_processed = np.transpose(dataset_processed)
        save_path = os.path.join(washed_data_dict, dataset + '.csv')
        save_path_scale = os.path.join(washed_data_dict, dataset + '-scale' + '.csv')

        # save data only with bounding box's center coordinates
        data_frame = pd.DataFrame(dataset_processed[0:4, :])
        data_frame.to_csv(save_path, index=False, header=False)
        # save data with bounding box's scale and coordinates
        data_frame_scale = pd.DataFrame(dataset_processed)
        data_frame_scale.to_csv(save_path_scale, index=False, header=False)
