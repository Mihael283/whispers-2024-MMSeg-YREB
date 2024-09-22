import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import rasterio
from rasterio.windows import Window

class WhisperSegDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Read file list
        with open(os.path.join(data_dir, f'{split}.txt'), 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        self.msi_dir = os.path.join(data_dir, split, 'MSI')
        self.sar_dir = os.path.join(data_dir, split, 'SAR')
        if split == 'train':
            self.label_dir = os.path.join(data_dir, split, 'label')

    def __len__(self):
        return len(self.file_list) * (256 // 64) * (256 // 64)  # Number of 64x64 patches per image

    def __getitem__(self, idx):
        file_idx = idx // ((256 // 64) * (256 // 64))
        patch_idx = idx % ((256 // 64) * (256 // 64))
        row = (patch_idx // (256 // 64)) * 64
        col = (patch_idx % (256 // 64)) * 64
        
        filename = self.file_list[file_idx]
        
        # Read MSI data
        with rasterio.open(os.path.join(self.msi_dir, filename)) as src:
            msi_data = src.read(window=Window(col, row, 64, 64))
        
        # Read SAR data
        with rasterio.open(os.path.join(self.sar_dir, filename)) as src:
            sar_data = src.read(window=Window(col, row, 64, 64))
        
        # Combine MSI and SAR data
        combined_data = np.vstack((msi_data, sar_data))
        
        if self.split == 'train':
            # Read label data
            with rasterio.open(os.path.join(self.label_dir, filename)) as src:
                label_data = src.read(window=Window(col, row, 64, 64))
            label_data = label_data.squeeze().astype(np.int64)
            label_data -= 1
            label_data[label_data < 0] = 0
        else:
            label_data = np.zeros((64, 64), dtype=np.int64)  # Dummy label for test set
        
        # Convert to torch tensors
        combined_data = torch.from_numpy(combined_data).float()
        label_data = torch.from_numpy(label_data).long()
        
        if self.transform:
            combined_data = self.transform(combined_data)
        
        return combined_data, label_data

class WhisperDataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406] + [0] * 11, std=[0.229, 0.224, 0.225] + [1] * 11),
        ])

        # Create datasets
        self.train_dataset = WhisperSegDataset(data_dir, split='train', transform=self.transform)
        
        # Split train set into train and validation
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size])
        
        self.test_dataset = WhisperSegDataset(data_dir, split='test', transform=self.transform)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    dataloader = WhisperDataLoader(data_dir, batch_size, num_workers)
    return dataloader.get_train_dataloader(), dataloader.get_val_dataloader(), dataloader.get_test_dataloader()