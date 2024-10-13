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
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter, map_coordinates



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

    def calculate_water_index(self, msi_data):
        # RE1 is B5 and S2 is B12 for Sentinel-2
        re1 = msi_data[4]  # B5 is at index 4 (0-based)
        s2 = msi_data[11]  # B12 is at index 11 (0-based)
        
        denominator = re1 + s2
        water_index = np.where(denominator != 0, (re1 - s2) / denominator, 0)
        
        return water_index.astype(np.float32)
    
    def calculate_ndsi(self, msi_data):
        # NDSI = (Green - SWIR1) / (Green + SWIR1)
        # For Sentinel-2: Green is B3, SWIR1 is B11
        green = msi_data[2]  # B3 is at index 2 (0-based)
        swir1 = msi_data[10]  # B11 is at index 10 (0-based)
        
        # Avoid division by zero
        denominator = green + swir1
        ndsi = np.where(denominator != 0, (green - swir1) / denominator, 0)
        
        return ndsi.astype(np.float32)

    def calculate_ndmi(self, msi_data):
        # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        # For Sentinel-2: NIR is B8, SWIR1 is B11
        nir = msi_data[7]    # B8 is at index 7 (0-based)
        swir1 = msi_data[10] # B11 is at index 10 (0-based)
        
        # Avoid division by zero
        denominator = nir + swir1
        ndmi = np.where(denominator != 0, (nir - swir1) / denominator, 0)
        
        return ndmi.astype(np.float32)


    def calculate_ndvi(self, msi_data):
        # NDVI = (NIR - Red) / (NIR + Red)
        red = msi_data[3]  # B4
        nir = msi_data[7]  # B8
        
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        return ndvi.astype(np.float32)

    def calculate_mndwi(self, msi_data):
        # MNDWI = (Green - SWIR) / (Green + SWIR)
        green = msi_data[2]  # B3
        swir = msi_data[10]  # B11
        
        denominator = green + swir
        mndwi = np.where(denominator != 0, (green - swir) / denominator, 0)
        return mndwi.astype(np.float32)

    def calculate_s3_snow_index(self, msi_data):
        # S3 = (Green - SWIR) / (Green + SWIR) - (Red / Green)
        green = msi_data[2]  # B3
        red = msi_data[3]    # B4
        swir = msi_data[10]  # B11
        
        denominator = green + swir
        s3 = np.where(denominator != 0, (green - swir) / denominator, 0) - np.where(green != 0, red / green, 0)
        return s3.astype(np.float32)

    def calculate_nbsims(self, msi_data):
        # NBSIMS = 0.36 * (G + R + N) - (((B + S2) / G) + S1)
        # For Sentinel-2: B is B2, G is B3, R is B4, N is B8, S1 is B11, S2 is B12
        blue = msi_data[1]   # B2 is at index 1 (0-based)
        green = msi_data[2]  # B3 is at index 2 (0-based)
        red = msi_data[3]    # B4 is at index 3 (0-based)
        nir = msi_data[7]    # B8 is at index 7 (0-based)
        swir1 = msi_data[10] # B11 is at index 10 (0-based)
        swir2 = msi_data[11] # B12 is at index 11 (0-based)
        
        nbsims = 0.36 * (green + red + nir) - (((blue + swir2) / np.maximum(green, 1e-10)) + swir1)
        
        return nbsims.astype(np.float32)
    
    def calculate_nbsi(self, msi_data):
        # NBSI = (SWIR1 + Red - NIR - Blue) / (SWIR1 + Red + NIR + Blue)
        swir1 = msi_data[10]  # B11
        red = msi_data[3]     # B4
        nir = msi_data[7]     # B8
        blue = msi_data[1]    # B2
        
        numerator = swir1 + red - nir - blue
        denominator = swir1 + red + nir + blue
        nbsi = np.where(denominator != 0, numerator / denominator, 0)
        return nbsi.astype(np.float32)

    def calculate_nbai(self, msi_data):
        # NBAI = (SWIR1 - NIR) / (SWIR1 + NIR)
        swir1 = msi_data[10]  # B11
        nir = msi_data[7]     # B8
        
        numerator = swir1 - nir
        denominator = swir1 + nir
        nbai = np.where(denominator != 0, numerator / denominator, 0)
        return nbai.astype(np.float32)

    def calculate_ndsoil(self, msi_data):
        # NDSI = (SWIR2 - Green) / (SWIR2 + Green)
        swir2 = msi_data[11]  # B12
        green = msi_data[2]   # B3
        
        numerator = swir2 - green
        denominator = swir2 + green
        ndsi = np.where(denominator != 0, numerator / denominator, 0)
        return ndsi.astype(np.float32)


    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        with rasterio.open(os.path.join(self.msi_dir, filename)) as src:
            msi_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))

        with rasterio.open(os.path.join(self.sar_dir, filename)) as src:
            sar_data = src.read(window=rasterio.windows.Window(0, 0, 256, 256))

        water_index = self.calculate_water_index(msi_data)
        ndsi = self.calculate_ndsi(msi_data)
        ndmi = self.calculate_ndmi(msi_data)
        ndvi = self.calculate_ndvi(msi_data)
        mndwi = self.calculate_mndwi(msi_data)
        s3_snow = self.calculate_s3_snow_index(msi_data)
        nbsi = self.calculate_nbsi(msi_data)
        nbai = self.calculate_nbai(msi_data)
        ndsi_soil = self.calculate_ndsoil(msi_data)

        if self.apply_lee_filter:
            # Apply Enhanced Lee filter to each band of the SAR data
            for i in range(sar_data.shape[0]):
                sar_data[i] = enhanced_lee_filter(sar_data[i])

        msi_data = np.delete(msi_data, [0, 9], axis=0)

        combined_data = np.vstack((msi_data, sar_data, 
                                   water_index[np.newaxis, ...],
                                   ndsi[np.newaxis, ...], 
                                   ndmi[np.newaxis, ...],
                                   ndvi[np.newaxis, ...],
                                   mndwi[np.newaxis, ...], 
                                   s3_snow[np.newaxis, ...],
                                   nbsi[np.newaxis, ...],
                                   nbai[np.newaxis, ...],
                                   ndsi_soil[np.newaxis, ...]))

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

        # Rotation
        if random.random() > 0.6:
            angle = random.randint(-270, 270)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
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
