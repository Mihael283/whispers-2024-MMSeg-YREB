import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import torch
from models.unet import UNet
from dataloader import WhisperSegDataset
import warnings
from rasterio.errors import NotGeoreferencedWarning
from matplotlib.colors import ListedColormap

# Suppress the specific warning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def denormalize(img, mean, std):
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    return img * std + mean

# Define class names and colors
class_names = ['Background', 'Tree', 'Grassland', 'Cropland', 'Low Vegetation', 'Wetland', 'Water', 'Built-up', 'Bare ground']
colors = ['#000000', '#008000', '#00FF00', '#FFFF00', '#808080', '#800080', '#0000FF', '#FF0000', '#A52A2A']
cmap = ListedColormap(colors)

def visualize_prediction(input_img, prediction, ground_truth=None, sample_idx=0):
    fig, axs = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(20, 7))

    # Visualize input image (using RGB channels)
    input_img_rgb = input_img[[3, 2, 1], :, :]  # B04, B03, B02 channels for RGB
    input_img_rgb = np.clip(input_img_rgb / np.max(input_img_rgb), 0, 1)
    axs[0].imshow(input_img_rgb.transpose(1, 2, 0))
    axs[0].set_title('Input (B04, B03, B02)')
    axs[0].axis('off')

    # Visualize prediction
    im = axs[1].imshow(prediction, cmap=cmap, vmin=0, vmax=8)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    # Add colorbar as legend
    cbar = plt.colorbar(im, ax=axs[1], orientation='vertical', aspect=30)
    cbar.set_ticks(np.arange(0, 9) + 0.5)
    cbar.set_ticklabels(class_names)
    cbar.set_label('Classes')

    # Visualize ground truth if available
    if ground_truth is not None:
        im = axs[2].imshow(ground_truth, cmap=cmap, vmin=0, vmax=8)
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')

        # Add colorbar as legend
        cbar = plt.colorbar(im, ax=axs[2], orientation='vertical', aspect=30)
        cbar.set_ticks(np.arange(0, 9) + 0.5)
        cbar.set_ticklabels(class_names)
        cbar.set_label('Classes')

    plt.tight_layout()
    return fig

# Set up paths and parameters
data_dir = 'MMSeg-YREB'
model_path = 'saved_models/unet_epoch_24_0.44573.pt'
predictions_dir = 'predictions'
output_dir = 'visualizations_pred'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# Load the model
model = UNet(n_channels=14, n_classes=9, bilinear=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Set up dataset
mean = [0.485] * 13 + [0.5]
std = [0.229] * 13 + [0.1]
test_dataset = WhisperSegDataset(data_dir, split='test', transform=None)

# Visualization loop
print("Starting visualization...")
with torch.no_grad():
    for idx, (inputs, _) in enumerate(test_dataset):
        try:
            # Load and denormalize input
            input_img = denormalize(inputs.numpy(), mean, std)

            # Load prediction
            prediction_path = os.path.join(predictions_dir, test_dataset.file_list[idx])
            with rasterio.open(prediction_path) as src:
                prediction = src.read(1)

            # Load ground truth if available (for validation set)
            ground_truth = None
            ground_truth_path = os.path.join(data_dir, 'train', 'label', test_dataset.file_list[idx])
            if os.path.exists(ground_truth_path):
                with rasterio.open(ground_truth_path) as src:
                    ground_truth = src.read(1)
                ground_truth -= 1  # Adjust labels to 0-8 range
                ground_truth[ground_truth < 0] = 0

            # Create visualization
            fig = visualize_prediction(input_img, prediction, ground_truth, idx)

            # Save visualization
            output_path = os.path.join(output_dir, f'visualization_{idx}.jpg')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved visualization for sample {idx} to {output_path}")

            # Optional: limit the number of visualizations
            if idx >= 99:  # Visualize first 100 samples
                break

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

print("Visualization complete.")