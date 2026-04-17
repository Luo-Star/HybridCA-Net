import os
import numpy as np

def sliding_window(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    for i in range(0, num_windows * step_size, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def save_windows_to_files(windows, base_filename, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, window in enumerate(windows):
        new_filename = f"{base_filename}_{i+1}.txt"
        output_path = os.path.join(output_directory, new_filename)
        np.savetxt(output_path, window)  # 根据实际情况更改delimiter
        print(f"Saved: {output_path}")

def process_txt_files(input_directory, output_directory, wind_size):
    window_size = wind_size
    step_size = 5

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            try:
                data = np.loadtxt(file_path)  # 根据实际情况更改delimiter
                windows = sliding_window(data, window_size, step_size)
                base_filename = os.path.splitext(filename)[0]
                save_windows_to_files(windows, base_filename, output_directory)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 使用时，将 'input_directory_path' 替换为你要遍历的文件夹路径
# 将 'output_directory_path' 替换为你要保存结果的文件夹路径
AD = '/data/home/202221000475/AD-fMRI-std/AD'
AD_slices = '/data/home/202221000475/AD-fMRI-std-slices/AD'

MCI = '/data/home/202221000475/AD-fMRI-std/MCI'
MCI_slices = '/data/home/202221000475/AD-fMRI-std-slices/MCI'

NC = '/data/home/202221000475/AD-fMRI-std/NC'
NC_slices = '/data/home/202221000475/AD-fMRI-std-slices/NC'

process_txt_files(AD, AD_slices,80)
process_txt_files(MCI, MCI_slices,80)
process_txt_files(NC, NC_slices,80)

# CID = '/media/lwc/Lwc/data/data/second/400/data/CID'
# CID_60 = '/media/lwc/Lwc/data/data/second/400/data/CID_W60'
#
# HC = '/media/lwc/Lwc/data/data/second/400/data/HC'
# HC_60 = '/media/lwc/Lwc/data/data/second/400/data/HC_W60'
#
# process_txt_files(CID, CID_60,60)
# process_txt_files(HC, HC_60,60)
#
# CID = '/media/lwc/Lwc/data/data/second/400/data/CID'
# CID_70 = '/media/lwc/Lwc/data/data/second/400/data/CID_W70'
#
# HC = '/media/lwc/Lwc/data/data/second/400/data/HC'
# HC_70 = '/media/lwc/Lwc/data/data/second/400/data/HC_W70'
#
# process_txt_files(CID, CID_70, 70)
# process_txt_files(HC, HC_70, 70)
#
# CID = '/media/lwc/Lwc/data/data/second/400/data/CID'
# CID_80 = '/media/lwc/Lwc/data/data/second/400/data/CID_W80'
#
# HC = '/media/lwc/Lwc/data/data/second/400/data/HC'
# HC_80 = '/media/lwc/Lwc/data/data/second/400/data/HC_W80'
#
# process_txt_files(CID, CID_80, 80)
# process_txt_files(HC, HC_80, 80)
#
#
# CID = '/media/lwc/Lwc/data/data/second/400/data/CID'
# CID_90 = '/media/lwc/Lwc/data/data/second/400/data/CID_W90'
#
# HC = '/media/lwc/Lwc/data/data/second/400/data/HC'
# HC_90 = '/media/lwc/Lwc/data/data/second/400/data/HC_W90'
#
# process_txt_files(CID, CID_90, 90)
# process_txt_files(HC, HC_90, 90)
#
# CID = '/media/lwc/Lwc/data/data/second/400/data/CID'
# CID_100 = '/media/lwc/Lwc/data/data/second/400/data/CID_W100'
#
# HC = '/media/lwc/Lwc/data/data/second/400/data/HC'
# HC_100 = '/media/lwc/Lwc/data/data/second/400/data/HC_W100'
#
# process_txt_files(CID, CID_100, 100)
# process_txt_files(HC, HC_100, 100)