import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size=96, augment=True, scale=1):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.augment = augment
        self.scale = scale
        
        # 验证数据集中的图像尺寸
        with h5py.File(self.h5_file, 'r') as f:
            first_lr = f['lr']['0'][:]
            if first_lr.shape[0] < patch_size or first_lr.shape[1] < patch_size:
                raise ValueError(f"Image size ({first_lr.shape[0]}x{first_lr.shape[1]}) is smaller than patch size ({patch_size}x{patch_size})")

    def augment_patch(self, lr_patch, hr_patch):
        """对patch进行数据增强"""
        # 随机水平翻转
        if np.random.random() < 0.5:
            lr_patch = np.fliplr(lr_patch).copy()
            hr_patch = np.fliplr(hr_patch).copy()

        # 随机垂直翻转
        if np.random.random() < 0.5:
            lr_patch = np.flipud(lr_patch).copy()
            hr_patch = np.flipud(hr_patch).copy()

        # 随机90度旋转
        k = np.random.randint(0, 4)
        lr_patch = np.rot90(lr_patch, k).copy()
        hr_patch = np.rot90(hr_patch, k).copy()

        return lr_patch, hr_patch

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 加载原始数据并确保是uint8类型
            lr = f['lr'][str(idx)][:].astype(np.float32)  # 保持浮点数据
            hr = f['hr'][str(idx)][:].astype(np.float32)
            
            # 将数据归一化到0-255范围
            lr = np.clip(lr * 255.0, 0, 255).astype(np.uint8)
            hr = np.clip(hr * 255.0, 0, 255).astype(np.uint8)
            
            # 确保图像尺寸大于patch_size
            h, w = lr.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                # 如果图像太小，调整图像大小到patch_size
                lr_img = Image.fromarray(lr)
                hr_img = Image.fromarray(hr)
                lr = np.array(lr_img.resize((self.patch_size, self.patch_size), Image.BICUBIC))
                hr = np.array(hr_img.resize((self.patch_size, self.patch_size), Image.BICUBIC))
                h, w = self.patch_size, self.patch_size

            # 随机裁剪
            x = np.random.randint(0, max(1, h - self.patch_size + 1))
            y = np.random.randint(0, max(1, w - self.patch_size + 1))
            
            lr_patch = lr[x:x + self.patch_size, y:y + self.patch_size]
            hr_patch = hr[x:x + self.patch_size, y:y + self.patch_size]

            # 应用数据增强
            if self.augment:
                lr_patch, hr_patch = self.augment_patch(lr_patch, hr_patch)

            # 转换为torch张量，调整通道顺序为(C, H, W)并归一化到0-1
            lr_tensor = torch.from_numpy(lr_patch.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float() / 255.0

            return lr_tensor, hr_tensor

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 加载原始数据并确保是float32类型
            lr = f['lr'][str(idx)][:].astype(np.float32)
            hr = f['hr'][str(idx)][:].astype(np.float32)
            
            # 将数据归一化到0-255范围
            lr = np.clip(lr * 255.0, 0, 255).astype(np.uint8)
            hr = np.clip(hr * 255.0, 0, 255).astype(np.uint8)
            
            # 直接转换为torch张量，保持原始尺寸
            lr_tensor = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0

            return lr_tensor, hr_tensor

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])