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


def visualize_predictions(model, dataloader, device, mean, std, n_samples=3, step=None):
    """
    Visualize and save predictions.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for data.
        device (torch.device): Device to perform computation on.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        n_samples (int): Number of samples to visualize.
        step (int): Training step number for saving the plot.
    """
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            break 

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    preds = preds.cpu().numpy()

    os.makedirs('./visualizations', exist_ok=True)

    for i in range(n_samples):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        def denorm(img, mean, std):
            mean = np.array(mean).reshape(-1, 1, 1)
            std = np.array(std).reshape(-1, 1, 1)
            return img * std + mean

        input_img = denorm(x[i], mean, std)
        input_img_rgb = input_img[[3, 2, 1], :, :]  # B04, B03, B02 channels for RGB (Red, Green, Blue)

        input_img_rgb = np.clip(input_img_rgb / np.max(input_img_rgb), 0, 1)

        axs[0].imshow(input_img_rgb.transpose(1, 2, 0))
        axs[0].set_title('Input (B04, B03, B02)')
        axs[0].axis('off')

        axs[1].imshow(y[i], cmap='jet', vmin=0, vmax=8)
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')
        
        axs[2].imshow(preds[i], cmap='jet', vmin=0, vmax=8)
        axs[2].set_title('Prediction')
        axs[2].axis('off')

        if step is not None:
            plt.suptitle(f'Step {step}')
        plt.tight_layout()

        save_path = f'./visualizations/step_{step}_sample_{i}.png'
        plt.savefig(save_path)
        plt.close()

