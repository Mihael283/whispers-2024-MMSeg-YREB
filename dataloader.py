import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import findpeaks

def enhanced_lee_filter(img, win_size=5, k=1.0, cu=0.523, cmax=1.73):
    """
    Apply Enhanced Lee filter to reduce speckle noise in SAR images.
    
    Args:
        img (numpy.ndarray): Input SAR image.
        win_size (int): Size of the filter window (must be odd).
        k (float): Damping factor for the filter.
        cu (float): Coefficient of variation for lee enhanced of noise.
        cmax (float): Max coefficient of variation for lee enhanced.
    
    Returns:
        numpy.ndarray: Filtered image.
    """
    img = findpeaks.stats.scale(img)
    return findpeaks.lee_enhanced_filter(img, win_size=win_size, k=k, cu=cu, cmax=cmax)


class WhisperSegDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, augment=False , apply_lee_filter=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.apply_lee_filter = apply_lee_filter
        
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

        if self.apply_lee_filter:
            # Apply Enhanced Lee filter to each band of the SAR data
            for i in range(sar_data.shape[0]):
                sar_data[i] = enhanced_lee_filter(sar_data[i])

        combined_data = np.vstack((msi_data, sar_data)) 

        if self.split == 'train':
            with rasterio.open(os.path.join(self.label_dir, filename)) as src:
                label_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))
            label_data = label_data.squeeze().astype(np.int64) 
        else:
            label_data = np.zeros((256, 256), dtype=np.int64) 
        
        combined_data = torch.from_numpy(combined_data).float()
        label_data = torch.from_numpy(label_data).long()
        
        if self.augment:
            combined_data, label_data = self.apply_augmentations(combined_data, label_data)
        
        if self.transform:
            combined_data = self.transform(combined_data)
        
        return combined_data, label_data
    
    def apply_augmentations(self, image, mask):
        mask = mask.unsqueeze(0)

        if random.random() > 0.5:
            angle = random.randint(-100, 100)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        mask = mask.squeeze(0)

        return image, mask

class WhisperDataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, mean=None, std=None, normalize=True, augment=True, apply_lee_filter=False):
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
        self.augment = augment
        self.apply_lee_filter = apply_lee_filter

        if normalize:
            if mean is None or std is None:
                raise ValueError("Mean and Std must be provided for normalization if normalize is True.")
            self.transform = transforms.Normalize(mean=mean, std=std)
        else:
            self.transform = None

        self.train_dataset = WhisperSegDataset(data_dir, split='train', transform=self.transform, augment=self.augment, apply_lee_filter=self.apply_lee_filter)
        
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        self.test_dataset = WhisperSegDataset(data_dir, split='test', transform=self.transform, apply_lee_filter=self.apply_lee_filter)

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

def get_dataloaders(data_dir, batch_size=32, num_workers=4, mean=None, std=None, normalize=True, augment=True, apply_lee_filter=False):
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
        normalize=normalize,
        augment=augment,
        apply_lee_filter=apply_lee_filter
    )
    return (
        dataloader.get_train_dataloader(),
        dataloader.get_val_dataloader(),
        dataloader.get_test_dataloader()
    )
