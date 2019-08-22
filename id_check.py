import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_frame_id(scene_path):
    """
    用于检查frame和行人id的连续性，同一id是否在两段以上的时间。
    检查方式：量化检查和可视化检查
    散点图格式：横坐标-帧数；纵坐标-该帧数下出现的id号
    :param scene_path: 具体场景文件夹名称
    :return:
    """
    gt_path = os.path.join('data', 'MOT16Labels', 'train', scene_path, 'gt', 'gt.txt')
    print('Ground Truth Path:', gt_path)
    gt_data = pd.read_csv(gt_path)
    gt_data = gt_data.to_numpy(dtype=float)
    print('Ground Truth data format:', gt_data.shape)
    frame_low_bound = int(np.min(gt_data[:, 0]))
    frame_up_bound = int(np.max(gt_data[:, 0]))

    # Get the unique id list
    target_select_list = np.logical_or(gt_data[:, 6] == 1, gt_data[:, 6] == 2)
    target_id_list = gt_data[target_select_list, :]
    target_id_list = np.unique(target_id_list[:, 1]).tolist()
    print('The amount of pedestrian\'s id is ', len(target_id_list))
    num_id = len(target_id_list)

    # initial a zero array to store seq info.
    id_frame = np.zeros((num_id + 1, (frame_up_bound - frame_low_bound + 1)), dtype=int)
    for index in range(0, len(target_id_list)):
        ped_id = target_id_list[index]
        cur_id_frames = gt_data[gt_data[:, 1] == ped_id, 0].tolist()
        for frame in cur_id_frames:
            id_frame[index, int(frame) - frame_low_bound] = index

    # plot
    x_label = np.array(range(frame_low_bound, frame_up_bound + 1, 1)).reshape(-1, 1)
    y_labels = np.split(id_frame, num_id + 1)
    for (i, y_label) in enumerate(y_labels):
        y_label = y_label.reshape(-1, 1)

        # quantitative check
        flag_up = 0
        flag_down = 0
        for j in range(1, frame_up_bound - frame_low_bound + 1):
            if y_label[j - 1, 0] == 0 and y_label[j, 0] > 0:
                flag_up = flag_up + 1
            elif y_label[j - 1, 0] > 0 and y_label[j, 0] == 0:
                flag_down = flag_down + 1
        if flag_up > 1 or flag_down > 1:
            print('[Warning]:{} th pedestrian\' trajectory is separate.'.format(i))

        # visibility check
        plt.scatter(x_label, y_label.reshape(-1, 1), s=3)
    plt.show()


if __name__ == '__main__':
    path = 'MOT16-' + '13'  # 04,05,09,10,11,13
    scatter_frame_id(path)
