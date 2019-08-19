import os
import numpy as np
import pandas as pd


def trajectory_length(data, fps, object_type=1):
    """
    Input dataset  and fps, return a dict with trajectory's id as key and trajectory's length as value.
    :param data : numpy array [N x 9]
    :param fps: frames per sec.
    :param object_type: the kind to be considered
    :return: dictionary
    """
    # get person list
    target_person_data = data[data[:, 7] == object_type, :]
    target_person_list = np.unique(target_person_data[:, 1]).tolist()

    # for each target person, calculate its time
    time_dict = {}
    for person in target_person_list:
        single_person_data = target_person_data[target_person_data[:, 1] == person, :]
        frames_num = np.shape(single_person_data)[0]
        time = frames_num / fps
        time_dict[int(person)] = time

    return time_dict


if __name__ == '__main__':
    dataset_dict = {'02': 30, '04': 30, '05': 14, '09': 30, '11': 30, '13': 25}
    for dataset in dataset_dict.keys():
        path = 'MOT16-' + dataset
        dataset_path = os.path.join('data', 'MOT16Labels', 'train', path, 'gt', 'gt.txt')

        data = pd.read_csv(dataset_path).to_numpy()
        target_time_dict = trajectory_length(data, dataset_dict.get(dataset))
        print('Overall Trajectory Number:', len(target_time_dict.keys()))
        print('Trajectory Longer than 2 secs', len([item for item in target_time_dict.values() if item > 2.0]))
        print(target_time_dict)
