import numpy
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
import scipy.io as io

# 遍历文件夹
def get_filelist(dir, Filelist):
    newDir = dir

    if os.path.isfile(dir):

        Filelist.append(dir)

        # # 若只是要返回文件文，使用这个

        # Filelist.append(os.path.basename(dir))

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码

            # if s == "xxx":

            # continue

            newDir = os.path.join(dir, s)

            get_filelist(newDir, Filelist)

    return Filelist

def calc_zscore(matrix):

    mean = np.mean(matrix, axis = 0)
    std = np.std(matrix, axis = 0)
    std[std==0] = 1
    zscore = (matrix - mean) / std
    zscore[np.isnan(zscore)]= 0
    return zscore
if __name__ == '__main__':
    cout = 0
    # 数据路径
    data_path = "/home/lwc/data/混合数据"
    #e
    data_lists = get_filelist(data_path, [])
    # print(len(data_lists))
    min = 116

    data = np.zeros((len(data_lists), 1, 30, min, 1))
    label = np.zeros(len(data_lists))

    # load all data
    idx = 0
    data_all = None

    for i in data_lists:
        full_sequence = np.load(i)
        data[idx, 0, :, :, 0] = full_sequence
        if "CID" == i[20:23]:
            label[idx] = 1
        else:
            label[idx] = 0
        if data_all is None:
            data_all = np.transpose(full_sequence)
        else:
            data_all = np.concatenate((data_all, np.transpose(full_sequence)), axis=1)
        idx = idx + 1

    n_regions = 116
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i, :], data_all[j, :])[0][1])  # get value from corrcoef matrix
                A[j][i] = A[i][j]

    np.save('data2/adj_matrix.npy', A)

    data = data[:idx]
    label = label[:idx]


    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2)
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.2)
    np.save('data2/train_data.npy', train_data)
    np.save('data2/train_label.npy', train_label)
    np.save('data2/test_data.npy', test_data)
    np.save('data2/test_label.npy', test_label)
    np.save('data2/val_data.npy', val_data)
    np.save('data2/val_label.npy', val_label)
