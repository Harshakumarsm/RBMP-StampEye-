import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, part='A', train=True, transform=None):
        self.root_dir = root_dir
        self.part = part
        self.train = train
        self.transform = transform
        if part == 'A':
            part_dir = 'part_A'
        else:
            part_dir = 'part_B'
        if train:
            img_dir = os.path.join(root_dir, part_dir, 'train_data', 'images')
            gt_dir = os.path.join(root_dir, part_dir, 'train_data', 'ground_truth')
        else:
            img_dir = os.path.join(root_dir, part_dir, 'test_data', 'images')
            gt_dir = os.path.join(root_dir, part_dir, 'test_data', 'ground_truth')
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = os.path.join(self.gt_dir, 'GT_' + os.path.basename(img_path).replace('.jpg', '.mat'))
        image = Image.open(img_path).convert('RGB')
        mat = sio.loadmat(gt_path)
        density_map = mat['density']
        if self.transform:
            image = self.transform(image)
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()
        return image, density_map

# Example transform for training
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) 