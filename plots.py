import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def plot_losses(filename='plot_losses.npy', output='loss_plots.png'):
    # Load plot data
    plot_losses = np.load(filename)

    # Plot loss
    plt.figure(figsize=(12,8))
    plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation'])
    plt.savefig(output)
    plt.close()

    print(f"Loss plot saved as '{output}'")

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor using the provided mean and std.
    """
    if mean is not None and std is not None:
        mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
    return tensor

def select_channels(input_tensor, channel_indices):
    """
    Selects specific channels from the input tensor.
    """
    return input_tensor[:, channel_indices, :, :]

def visualize_batch(model, x, y, preds, batch_idx, epoch, mean=None, std=None):
    n_samples = min(3, x.shape[0])
    
    # Denormalize the inputs
    x_denorm = denormalize(x.clone(), mean, std)

    # Select channels for visualization
    rgb_channels = [3, 2, 1]  # B04, B03, B02 for RGB
    sar_channel = [13]  # Assuming SAR is the last channel

    x_rgb = select_channels(x_denorm, rgb_channels)
    x_sar = select_channels(x_denorm, sar_channel)

    os.makedirs('./visualizations', exist_ok=True)

    for i in range(n_samples):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # RGB Image
        rgb_image = x_rgb[i].permute(1, 2, 0).cpu().numpy()
        rgb_min, rgb_max = rgb_image.min(), rgb_image.max()
        if rgb_max - rgb_min > 0:
            rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb_image = np.zeros_like(rgb_image)
        
        axs[0].imshow(rgb_image)
        axs[0].set_title('RGB (B04, B03, B02)')
        axs[0].axis('off')

        # SAR Image
        sar_image = x_sar[i].squeeze().cpu().numpy()
        sar_min, sar_max = sar_image.min(), sar_image.max()
        if sar_max - sar_min > 0:
            sar_image = (sar_image - sar_min) / (sar_max - sar_min)
        else:
            sar_image = np.zeros_like(sar_image)
        
        axs[1].imshow(sar_image, cmap='gray')
        axs[1].set_title('SAR Image')
        axs[1].axis('off')

        # Ground Truth
        axs[2].imshow(y[i].cpu(), cmap='jet', vmin=0, vmax=9)
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')
        
        # Prediction
        axs[3].imshow(preds[i].cpu(), cmap='jet', vmin=0, vmax=9)
        axs[3].set_title('Prediction')
        axs[3].axis('off')

        plt.suptitle(f'Epoch {epoch+1}, Batch {batch_idx}')
        plt.tight_layout()
    
        save_path = f'./visualizations/epoch_{epoch+1}_batch_{batch_idx}_sample_{i}.png'
        plt.savefig(save_path)
        plt.close()

# Usage example (within your training loop):
# visualize_batch(model, x, y, preds, batch_idx, epoch, mean=mean, std=std)