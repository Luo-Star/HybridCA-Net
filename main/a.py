import os

def count_non_empty_folders(root_dir):
    count = 0
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            smri_path = os.path.join(subdir_path, 'sMRI')
            fmri_path = os.path.join(subdir_path, 'fMRI')
            if os.path.isdir(smri_path) and os.path.isdir(fmri_path):
                if len(os.listdir(smri_path)) > 0 and len(os.listdir(fmri_path)) > 0:
                    count += 1
    return count

root_dir = '/media/lwc/Lwc/ADNI-raw/处理好的数据/融合/MCI'
non_empty_count = count_non_empty_folders(root_dir)
print(f"sMRI和fMRI文件夹均不为空的子文件夹个数: {non_empty_count}")
