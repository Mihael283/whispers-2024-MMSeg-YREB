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
import matplotlib.patches as mpatches

# Suppress the specific warning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Define class names and colors
class_names = ['Background', 'Tree', 'Grassland', 'Cropland', 'Low Vegetation', 'Wetland', 'Water', 'Built-up', 'Bare ground', 'Snow']
colors = ['#000000', '#008000', '#00FF00', '#FFFF00', '#808080', '#800080', '#0000FF', '#FF0000', '#A52A2A', '#FFFFFF']
cmap = plt.cm.colors.ListedColormap(colors)

def denormalize(img, mean, std):
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    return img * std + mean

def visualize_prediction(input_img, prediction, ground_truth=None, sample_idx=0):
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    # Visualize input image (using RGB channels)
    input_img_rgb = input_img[[3, 2, 1], :, :]  # B04, B03, B02 channels for RGB
    input_img_rgb = np.clip(input_img_rgb / 3000, 0, 1)  # Adjust clipping values as needed
    axs[0].imshow(input_img_rgb.transpose(1, 2, 0))
    axs[0].set_title('Input (B04, B03, B02)')
    axs[0].axis('off')

    # Visualize prediction
    im = axs[1].imshow(prediction, cmap=cmap, vmin=0, vmax=9)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    # Overlay view
    overlay = input_img_rgb.transpose(1, 2, 0).copy()
    overlay[prediction > 0] = cmap(prediction[prediction > 0] / 9.0)[:, :3]
    axs[2].imshow(overlay)
    axs[2].set_title('Overlay: Input + Prediction')
    axs[2].axis('off')

    # Custom legend
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust this value to fit the legend
    return fig

# Set up paths and parameters
data_dir = 'MMSeg-YREB'
predictions_dir = 'predictions'
output_dir = 'visualizations_pred'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# Set up dataset
mean = [0.485] * 13 + [0.5]
std = [0.229] * 13 + [0.1]
test_dataset = WhisperSegDataset(data_dir, split='test', transform=None)

# Visualization loop
print("Starting visualization...")
for idx, (inputs, _) in enumerate(test_dataset):
    try:
        # Load and denormalize input
        input_img = denormalize(inputs.numpy(), mean, std)

        # Load prediction
        prediction_path = os.path.join(predictions_dir, test_dataset.file_list[idx])
        with rasterio.open(prediction_path) as src:
            prediction = src.read(1)

        # Create visualization
        fig = visualize_prediction(input_img, prediction, sample_idx=idx)

        # Save visualization
        output_path = os.path.join(output_dir, f'visualization_{idx}.png')
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