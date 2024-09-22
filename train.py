import numpy as np
import os
import argparse
import torch
from torch import nn
from tqdm import tqdm

from losses import FocalLoss, mIoULoss
from models.unet import UNet
from dataloader import WhisperDataLoader, get_dataloaders
from plots import plot_losses

# Set parameters directly
data_path = 'MMSeg-YREB'
num_epochs = 100
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loss function
focal_loss = FocalLoss(gamma=3/4).to(device)
iou_loss = mIoULoss(n_classes=9).to(device)

def combined_loss(pred, target):
    return focal_loss(pred, target) + iou_loss(pred, target)

criterion = combined_loss

# Data loaders
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=batch_size)

print('Number of training data:', len(train_dataloader.dataset))
print('Number of validation data:', len(val_dataloader.dataset))
print('Number of test data:', len(test_dataloader.dataset))

# Model and optimizer
model = UNet(n_channels=14, n_classes=9, bilinear=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# Training setup
os.makedirs('./saved_models', exist_ok=True)
min_loss = float('inf')
plot_losses_data = []
scheduler_counter = 0

# Training loop
for epoch in range(num_epochs):
    # Training
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
        
        seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
        train_acc_list.append(seg_acc.numpy())

        train_loop.set_postfix(loss=np.mean(train_loss_list), acc=np.mean(train_acc_list))
    
    # Validation
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
        scheduler_counter = 0
        min_loss = val_loss
        torch.save(model.state_dict(), f'./saved_models/unet_epoch_{epoch+1}_{val_loss:.5f}.pt')
    
    if scheduler_counter > 5:
        lr_scheduler.step()
        print(f"Lowering learning rate to {optimizer.param_groups[0]['lr']}")
        scheduler_counter = 0
    
    scheduler_counter += 1

# Save plot data
np.save('plot_losses.npy', np.array(plot_losses_data))

# Generate and save the plot
plot_losses()