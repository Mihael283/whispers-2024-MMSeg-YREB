import numpy as np
import os
import argparse
import torch
from torch import nn
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
import matplotlib.pyplot as plt

from losses import FocalLoss, mIoULoss
from models.unet import UNet
from dataloader import WhisperDataLoader, get_dataloaders
from plots import plot_losses, visualize_predictions
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

data_path = 'MMSeg-YREB'
num_epochs = 100
batch_size = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loss
focal_loss = FocalLoss(gamma=3/4).to(device)
iou_loss = mIoULoss(n_classes=14).to(device)

def combined_loss(pred, target):
    return focal_loss(pred, target) + iou_loss(pred, target)

criterion = combined_loss

mean = [0.485] * 13 + [0.5]
std = [0.229] * 13 + [0.1]

print("\nInitializing DataLoaders with normalization...")
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    data_dir=data_path,
    batch_size=batch_size,
    num_workers=4,
    mean=mean,
    std=std,    
    normalize=True 
)

print('Number of training data:', len(train_dataloader.dataset))
print('Number of validation data:', len(val_dataloader.dataset))
print('Number of test data:', len(test_dataloader.dataset))

model = UNet(n_channels=14, n_classes=9, bilinear=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

T_max = num_epochs
eta_min = 1e-6
lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

os.makedirs('./saved_models', exist_ok=True)
min_loss = float('inf')
plot_losses_data = []

# Set visualization interval
visualize_interval = 1
global_step = 0	

for epoch in range(num_epochs):
    model.train()
    train_loss_list = []
    train_acc_list = []
    train_loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for x, y in train_loop:
        x, y = x.to(device), y.to(device)
        pred_mask = model(x)  
        loss = criterion(pred_mask, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.cpu().detach().numpy())

        global_step += 1

        seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
        train_acc_list.append(seg_acc.numpy())

        train_loop.set_postfix(loss=np.mean(train_loss_list), acc=np.mean(train_acc_list))
        
        # Check if it's time to visualize
        #if global_step % visualize_interval == 0:
            #visualize_predictions(model, val_dataloader, device, mean, std, n_samples=3, step=global_step)
    
    model.eval()
    val_loss_list = []
    val_acc_list = []
    val_loop = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
    with torch.no_grad():
        for x, y in val_loop:
            x, y = x.to(device), y.to(device)
            pred_mask = model(x)  
            val_loss = criterion(pred_mask, y)
            val_loss_list.append(val_loss.cpu().detach().numpy())
            
            seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
            val_acc_list.append(seg_acc.numpy())

            val_loop.set_postfix(loss=np.mean(val_loss_list), acc=np.mean(val_acc_list))

    train_loss = np.mean(train_loss_list)
    train_acc = np.mean(train_acc_list)
    val_loss = np.mean(val_loss_list)
    val_acc = np.mean(val_acc_list)
    
    print(f'Epoch {epoch+1} - loss : {train_loss:.5f} - acc : {train_acc:.2f} - val loss : {val_loss:.5f} - val acc : {val_acc:.2f}')
    plot_losses_data.append([epoch, train_loss, val_loss])

    is_best = val_loss < min_loss
    if is_best:
        min_loss = val_loss
        torch.save(model.state_dict(), f'./saved_models/unet_epoch_{epoch+1}_{val_loss:.5f}.pt')

    lr_scheduler.step()
    print(f"Epoch {epoch+1}: Learning rate set to {optimizer.param_groups[0]['lr']}")

    #if (epoch + 1) % visualize_interval == 0 or epoch == 0:
        #visualize_predictions(model, val_dataloader, device, mean, std, n_samples=3, epoch=epoch)

np.save('plot_losses.npy', np.array(plot_losses_data))
plot_losses()