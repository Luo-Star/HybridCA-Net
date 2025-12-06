import os

import numpy as np
from scipy import stats


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

def cal_adj(path):
    fileLists = get_filelist(path, [])
    adj_all = np.zeros((116, 116))
    for item in fileLists:
        data = np.loadtxt(item)
        # e = (data > 0.8) | (data < -0.8)
        # full_sequence = np.where(e, 1, 0)
        adj_all += data
    A = divide_my_col_sum(adj_all)
    zscore = stats.zscore(adj_all)
    np.save('data/adj_matrix.npy', A)
    np.save('data/adj_matrix_unz_zscore.npy', zscore)
    pass

#
def divide_my_col_sum(martrix):
    """
    计算矩阵中每个元素除以每一列的和

    :param martrix: 输入的矩阵
    :return: 计算结果矩阵
    """
    # 计算每一列的和
    col_sum = np.sum(martrix, axis=0)

    # 计算矩阵中每个元素除以每一列的和
    result = np.divide(martrix, col_sum)

    return result
if __name__ == '__main__':
    path = "/home/lwc/data/mix_data/z"
    cal_adj(path)