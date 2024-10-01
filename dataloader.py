import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import rasterio
import matplotlib.pyplot as plt

class WhisperSegDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        with open(os.path.join(data_dir, f'{split}.txt'), 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        self.msi_dir = os.path.join(data_dir, split, 'MSI')
        self.sar_dir = os.path.join(data_dir, split, 'SAR')
        if split == 'train':
            self.label_dir = os.path.join(data_dir, split, 'label')

    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        with rasterio.open(os.path.join(self.msi_dir, filename)) as src:
            msi_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))
        
        with rasterio.open(os.path.join(self.sar_dir, filename)) as src:
            sar_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))
        
        combined_data = np.vstack((msi_data, sar_data)) 

        if self.split == 'train':
            with rasterio.open(os.path.join(self.label_dir, filename)) as src:
                label_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))
            label_data = label_data.squeeze().astype(np.int64) 
            # Remove the label adjustment
            # label_data -= 1 
            # label_data[label_data < 0] = 0
        else:
            label_data = np.zeros((256, 256), dtype=np.int64) 
        
        combined_data = torch.from_numpy(combined_data).float()
        label_data = torch.from_numpy(label_data).long()
        
        if self.transform:
            combined_data = self.transform(combined_data)
        
        return combined_data, label_data

class WhisperDataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, mean=[0.485] * 13 + [0.5], std=[0.229] * 13 + [0.1], normalize=True):
        """
        Args:
            data_dir (str): Path to the data directory.
            batch_size (int): Batch size.
            num_workers (int): Number of worker threads.
            mean (list or None): List of means for normalization.
            std (list or None): List of stds for normalization.
            normalize (bool): Whether to apply normalization.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if normalize:
            if mean is None or std is None:
                raise ValueError("Mean and Std must be provided for normalization if normalize is True.")
            self.transform = transforms.Normalize(mean=mean, std=std)
        else:
            self.transform = None

        self.train_dataset = WhisperSegDataset(data_dir, split='train', transform=self.transform)
        
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        self.test_dataset = WhisperSegDataset(data_dir, split='test', transform=self.transform)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )


def get_dataloaders(data_dir, batch_size=32, num_workers=4, mean=None, std=None, normalize=True):
    """
    Returns:
        train_loader, val_loader, test_loader
    """
    dataloader = WhisperDataLoader(
        data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        mean=mean, 
        std=std, 
        normalize=normalize
    )
    return (
        dataloader.get_train_dataloader(),
        dataloader.get_val_dataloader(),
        dataloader.get_test_dataloader()
    )
