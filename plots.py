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
    
    x_denorm = denormalize(x.clone(), mean, std)

    msi_channels = list(range(10))
    sar_channel = [10]
    water_index_channel = [11]
    ndsi_channel = [12]
    ndmi_channel = [13]
    ndvi_channel = [14]
    mndwi_channel = [15]
    s3_snow_channel = [16]
    nbsi_channel = [17]
    nbai_channel = [18]
    ndsi_soil_channel = [19]
    rgb_channels = [2, 1, 0]

    x_msi = select_channels(x_denorm, msi_channels)
    x_sar = select_channels(x_denorm, sar_channel)
    x_water_index = select_channels(x_denorm, water_index_channel)
    x_ndsi = select_channels(x_denorm, ndsi_channel)
    x_ndmi = select_channels(x_denorm, ndmi_channel)
    x_ndvi = select_channels(x_denorm, ndvi_channel)
    x_mndwi = select_channels(x_denorm, mndwi_channel)
    x_s3_snow = select_channels(x_denorm, s3_snow_channel)
    x_nbsi = select_channels(x_denorm, nbsi_channel)
    x_nbai = select_channels(x_denorm, nbai_channel)
    x_ndsi_soil = select_channels(x_denorm, ndsi_soil_channel)
    x_rgb = select_channels(x_denorm, rgb_channels)
    
    os.makedirs('./visualizations', exist_ok=True)

    for i in range(n_samples):
        fig, axs = plt.subplots(5, 5, figsize=(25, 25))
        
        rgb_image = x_rgb[i].permute(1, 2, 0).cpu().numpy()
        rgb_min, rgb_max = rgb_image.min(), rgb_image.max()
        if rgb_max - rgb_min > 0:
            rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb_image = np.zeros_like(rgb_image)
        
        axs[0, 0].imshow(rgb_image)
        axs[0, 0].set_title('RGB (B04, B03, B02)')
        axs[0, 0].axis('off')

        # SAR Image
        sar_image = x_sar[i].squeeze().cpu().numpy()
        sar_min, sar_max = sar_image.min(), sar_image.max()
        sar_image = (sar_image - sar_min) / (sar_max - sar_min) if sar_max > sar_min else np.zeros_like(sar_image)
        axs[0, 1].imshow(sar_image, cmap='gray')
        axs[0, 1].set_title('SAR Image')
        axs[0, 1].axis('off')

        # Ground Truth
        axs[0, 2].imshow(y[i].cpu(), cmap='jet', vmin=0, vmax=9)
        axs[0, 2].set_title('Ground Truth')
        axs[0, 2].axis('off')
        
        # Prediction
        axs[0, 3].imshow(preds[i].cpu(), cmap='jet', vmin=0, vmax=9)
        axs[0, 3].set_title('Prediction')
        axs[0, 3].axis('off')

        # Water Index
        water_index = x_water_index[i].squeeze().cpu().numpy()
        wi_min, wi_max = water_index.min(), water_index.max()
        water_index = (water_index - wi_min) / (wi_max - wi_min) if wi_max > wi_min else np.zeros_like(water_index)
        axs[0, 4].imshow(water_index, cmap='viridis')
        axs[0, 4].set_title('Water Index')
        axs[0, 4].axis('off')

        # NDSI (Snow)
        ndsi = x_ndsi[i].squeeze().cpu().numpy()
        ndsi_min, ndsi_max = ndsi.min(), ndsi.max()
        ndsi = (ndsi - ndsi_min) / (ndsi_max - ndsi_min) if ndsi_max > ndsi_min else np.zeros_like(ndsi)
        axs[1, 0].imshow(ndsi, cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 0].set_title('NDSI (Snow)')
        axs[1, 0].axis('off')

        # NDMI
        ndmi = x_ndmi[i].squeeze().cpu().numpy()
        ndmi_min, ndmi_max = ndmi.min(), ndmi.max()
        ndmi = (ndmi - ndmi_min) / (ndmi_max - ndmi_min) if ndmi_max > ndmi_min else np.zeros_like(ndmi)
        axs[1, 1].imshow(ndmi, cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 1].set_title('NDMI')
        axs[1, 1].axis('off')

        # NDVI
        ndvi = x_ndvi[i].squeeze().cpu().numpy()
        ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
        ndvi = (ndvi - ndvi_min) / (ndvi_max - ndvi_min) if ndvi_max > ndvi_min else np.zeros_like(ndvi)
        axs[1, 2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axs[1, 2].set_title('NDVI')
        axs[1, 2].axis('off')

        # MNDWI
        mndwi = x_mndwi[i].squeeze().cpu().numpy()
        mndwi_min, mndwi_max = mndwi.min(), mndwi.max()
        mndwi = (mndwi - mndwi_min) / (mndwi_max - mndwi_min) if mndwi_max > mndwi_min else np.zeros_like(mndwi)
        axs[1, 3].imshow(mndwi, cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 3].set_title('MNDWI')
        axs[1, 3].axis('off')

        # S3 Snow Index
        s3_snow = x_s3_snow[i].squeeze().cpu().numpy()
        s3_snow_min, s3_snow_max = s3_snow.min(), s3_snow.max()
        s3_snow = (s3_snow - s3_snow_min) / (s3_snow_max - s3_snow_min) if s3_snow_max > s3_snow_min else np.zeros_like(s3_snow)
        axs[1, 4].imshow(s3_snow, cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 4].set_title('S3 Snow Index')
        axs[1, 4].axis('off')

        # NBSI (Normalized Bare Soil Index)
        nbsi = x_nbsi[i].squeeze().cpu().numpy()
        nbsi_min, nbsi_max = nbsi.min(), nbsi.max()
        nbsi = (nbsi - nbsi_min) / (nbsi_max - nbsi_min) if nbsi_max > nbsi_min else np.zeros_like(nbsi)
        axs[2, 0].imshow(nbsi, cmap='YlOrBr', vmin=-1, vmax=1)
        axs[2, 0].set_title('NBSI')
        axs[2, 0].axis('off')

        # NBAI (Normalized Built-up Area Index)
        nbai = x_nbai[i].squeeze().cpu().numpy()
        nbai_min, nbai_max = nbai.min(), nbai.max()
        nbai = (nbai - nbai_min) / (nbai_max - nbai_min) if nbai_max > nbai_min else np.zeros_like(nbai)
        axs[2, 1].imshow(nbai, cmap='RdYlBu', vmin=-1, vmax=1)
        axs[2, 1].set_title('NBAI')
        axs[2, 1].axis('off')

        # NDSI (Soil)
        ndsi_soil = x_ndsi_soil[i].squeeze().cpu().numpy()
        ndsi_soil_min, ndsi_soil_max = ndsi_soil.min(), ndsi_soil.max()
        ndsi_soil = (ndsi_soil - ndsi_soil_min) / (ndsi_soil_max - ndsi_soil_min) if ndsi_soil_max > ndsi_soil_min else np.zeros_like(ndsi_soil)
        axs[2, 2].imshow(ndsi_soil, cmap='RdYlBu', vmin=-1, vmax=1)
        axs[2, 2].set_title('NDSI (Soil)')
        axs[2, 2].axis('off')

        # Individual channel previews
        for j in range(10):  # 10 MSI bands
            channel_image = x_msi[i, j].cpu().numpy()
            ch_min, ch_max = channel_image.min(), channel_image.max()
            channel_image = (channel_image - ch_min) / (ch_max - ch_min) if ch_max > ch_min else np.zeros_like(channel_image)
            
            row = (j+15) // 5
            col = (j) % 5
            axs[row, col].imshow(channel_image, cmap='gray')
            axs[row, col].set_title(f'MSI Channel {j+1}')
            axs[row, col].axis('off')

        plt.suptitle(f'Epoch {epoch+1}, Batch {batch_idx}')
        plt.tight_layout()
    
        save_path = f'./visualizations/epoch_{epoch+1}_batch_{batch_idx}_sample_{i}.png'
        plt.savefig(save_path)
        plt.close()