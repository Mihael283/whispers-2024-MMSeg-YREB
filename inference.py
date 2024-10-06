import os
import torch
import numpy as np
import rasterio
from zipfile import ZipFile
from models.unet import UNet
from dataloader import WhisperDataLoader, WhisperSegDataset

data_dir = 'MMSeg-YREB'
model_path = 'saved_models/unet_epoch_41_0.78944.pt'
output_dir = 'predictions'
zip_filename = 'submission.zip' 
batch_size = 1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

model = UNet(n_channels=14, n_classes=10, bilinear=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define normalization parameters (same as in training)
mean = [0.485] * 13 + [0.5]
std = [0.229] * 13 + [0.1]

# Create test dataset using your custom dataloader
dataloader = WhisperDataLoader(
    data_dir, 
    batch_size=batch_size, 
    num_workers=4, 
    mean=mean, 
    std=std, 
    normalize=True,
    augment=False  # Disable augmentations for inference
)
test_loader = dataloader.get_test_dataloader()

# Create a separate WhisperSegDataset for file list
test_dataset = WhisperSegDataset(data_dir, split='test', transform=False, augment=False)

def save_prediction_as_tiff(pred_mask, filename, reference_file_path):
    """
    Save the predicted mask as a TIFF file.

    Args:
        pred_mask (numpy array): The predicted segmentation mask.
        filename (str): The output filename for the TIFF file.
        reference_file_path (str): Path to the reference file (for geospatial metadata).
    """
    with rasterio.open(reference_file_path) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
        )

        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(pred_mask.astype(rasterio.uint8), 1)

print("Starting inference...")
with torch.no_grad():
    for idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()

        filename = test_dataset.file_list[idx]
        tiff_path = os.path.join(output_dir, filename)
        reference_file_path = os.path.join(data_dir, 'test', 'MSI', filename)
        
        # Ensure pred_mask is 2D (H x W)
        pred_mask = pred_mask.squeeze()
        
        save_prediction_as_tiff(pred_mask, tiff_path, reference_file_path)

        print(f"Saved prediction for {filename} as TIFF. Shape: {pred_mask.shape}")

print("Zipping predictions...")
with ZipFile(zip_filename, 'w') as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)

print(f"All predictions saved and zipped into {zip_filename}.")