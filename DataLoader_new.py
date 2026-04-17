import os
import random

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter


class FusionDataset(Dataset):
    def __init__(self, root_dir, fmri_cut_size=80, split='train', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 seed=42, classes=('AD', 'MCI', 'NC'), balance_method=None):
        self.root_dir = root_dir
        self.fmri_cut_size = fmri_cut_size
        self.data = []
        self.labels = {class_name: idx for idx, class_name in enumerate(classes)}
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.classes = classes
        self.balance_method = balance_method
        self._load_data()
        self._split_data()
        if split == 'train' and balance_method and set(classes) == {'MCI', 'NC'}:
            self._balance_data()

    def _load_data(self):
        for label_dir in os.listdir(self.root_dir):
            if label_dir in self.classes:
                label_path = os.path.join(self.root_dir, label_dir)
                if os.path.isdir(label_path):
                    label = self.labels.get(label_dir, None)
                    if label is not None:
                        for subject in os.listdir(label_path):
                            subject_path = os.path.join(label_path, subject)
                            if os.path.isdir(subject_path):
                                smri_dir = os.path.join(subject_path, 'sMRI')
                                fmri_dir = os.path.join(subject_path, 'fMRI')
                                smri_file = self._find_file(smri_dir, '.nii')
                                fmri_file = self._find_file(fmri_dir, '.txt')
                                if smri_file and fmri_file:
                                    self.data.append((smri_file, fmri_file, label, subject_path))
                                else:
                                    print(f"Skipping subject {subject} in {label_dir} due to missing file")

    def _find_file(self, directory, extension):
        if not os.path.isdir(directory):
            return None
        for file in os.listdir(directory):
            if file.endswith(extension):
                return os.path.join(directory, file)
        return None

    def _split_data(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.data))
        train_end = int(self.train_ratio * len(indices))
        val_end = train_end + int(self.val_ratio * len(indices))

        if self.split == 'train':
            self.data = [self.data[i] for i in indices[:train_end]]
        elif self.split == 'val':
            self.data = [self.data[i] for i in indices[train_end:val_end]]
        elif self.split == 'test':
            self.data = [self.data[i] for i in indices[val_end:]]

    def _balance_data(self):
        targets = [label for _, _, label, _ in self.data]
        class_counts = Counter(targets)
        min_count = min(class_counts.values())

        balanced_data = []
        for target_class in class_counts:
            class_data = [item for item in self.data if item[2] == target_class]
            if self.balance_method == 'undersample':
                class_data = random.sample(class_data, min_count)
            balanced_data.extend(class_data)

        self.data = balanced_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smri_path, fmri_path, label, directory = self.data[idx]
        smri_data = self._load_nii_file(smri_path)
        fmri_data = self._load_txt_file(fmri_path)
        fmri_data = self._standardize_data(fmri_data)
        return smri_data, fmri_data, label

    def get_targets(self):
        return [label for _, _, label, _ in self.data]

    def _load_nii_file(self, file_path):
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        data = np.expand_dims(data, axis=(0))
        return torch.tensor(data, dtype=torch.float)

    def _load_txt_file(self, file_path):
        data = np.loadtxt(file_path)
        if data.shape[0] > self.fmri_cut_size:
            start_idx = np.random.randint(0, data.shape[0] - self.fmri_cut_size + 1)
            data = data[start_idx:start_idx + self.fmri_cut_size, :]
        data = np.expand_dims(data, axis=(0, -1))  # Add dimensions at the beginning and end
        return torch.tensor(data, dtype=torch.float)

    def _standardize_data(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        return (data - mean) / std


def count_classes(dataset):
    counter = Counter()
    for _, _, label in dataset:
        counter[label] += 1
    return counter


if __name__ == '__main__':
    root_dir = '../data'
    fmri_cut_size = 80  # Default size is 80, modify if needed

    # Example usage with different class combinations
    train_dataset = FusionDataset(root_dir, fmri_cut_size=fmri_cut_size, split='train', classes=['MCI', 'NC'],
                                  balance_method='undersample')
    val_dataset = FusionDataset(root_dir, fmri_cut_size=fmri_cut_size, split='val', classes=['MCI', 'NC'])
    test_dataset = FusionDataset(root_dir, fmri_cut_size=fmri_cut_size, split='test', classes=['MCI', 'NC'])

    # 统计每个数据集中各类的数量
    train_class_counts = count_classes(train_dataset)
    val_class_counts = count_classes(val_dataset)
    test_class_counts = count_classes(test_dataset)

    print(f"Train dataset class counts: {train_class_counts}")
    print(f"Validation dataset class counts: {val_class_counts}")
    print(f"Test dataset class counts: {test_class_counts}")
