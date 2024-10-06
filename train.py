import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
import matplotlib.pyplot as plt

from losses import FocalLoss, mIoULoss
from models.deeplabv3 import DeepLabV3Plus
from dataloader import WhisperDataLoader, get_dataloaders
from plots import plot_losses, visualize_batch
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DEBUG = False

data_path = 'MMSeg-YREB'
num_epochs = 100
batch_size = 32
num_classes = 10  # Updated to 10 classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alpha = [
    0.55,  # Tree
    0.04,  # Grassland
    0.07,  # Cropland
    0.19,  # Low Vegetation
    0.01,  # Wetland
    0.04,  # Water
    0.04,  # Built-up
    0.04,  # Bareground
    0.01   # Snow
]

alpha = [a / sum(alpha) for a in alpha]

focal_loss = FocalLoss(gamma=3/4, alpha=alpha).to(device)
iou_loss = mIoULoss(n_classes=num_classes).to(device)

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

model = DeepLabV3Plus(num_classes=num_classes,num_channels=14).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

T_max = num_epochs
eta_min = 1e-6
lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

os.makedirs('./saved_models', exist_ok=True)
min_loss = float('inf')
plot_losses_data = []

# Set visualization interval
visualize_interval = 1
global_step = 0	

from sklearn.metrics import confusion_matrix

def calculate_miou(y_pred, y_true, num_classes):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = np.zeros_like(intersection, dtype=np.float32)
    valid = union != 0
    IoU[valid] = intersection[valid].astype(np.float32) / union[valid].astype(np.float32)
    return np.mean(IoU)

for epoch in range(num_epochs):
    model.train()
    train_loss_list = []
    train_acc_list = []
    train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch_idx, (x, y) in train_loop:
        x, y = x.to(device), y.to(device)

        if DEBUG:
            print(f"Batch {batch_idx}")
            print(f"Input shape: {x.shape}, Range: ({x.min().item():.2f}, {x.max().item():.2f})")
            print(f"Unique values in ground truth: {torch.unique(y).cpu().numpy()}")

        pred_mask = model(x)  

        if DEBUG:
            with torch.no_grad():
                print(f"Output shape: {pred_mask.shape}")
                print(f"Output range: ({pred_mask.min().item():.2f}, {pred_mask.max().item():.2f})")
                print(f"Unique predicted classes: {torch.unique(pred_mask.argmax(dim=1)).cpu().numpy()}")

        loss = criterion(pred_mask, y)

        optimizer.zero_grad()
        loss.backward()

        if DEBUG:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} - grad range: ({param.grad.min().item():.2f}, {param.grad.max().item():.2f})")
        
        optimizer.step()
        train_loss_list.append(loss.cpu().detach().numpy())

        seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
        train_acc_list.append(seg_acc.numpy())

        train_loop.set_postfix(loss=np.mean(train_loss_list), acc=np.mean(train_acc_list))
        
        # Visualize every 10th batch
        if batch_idx % 10 == 0:  # Visualize every 10th batch
            with torch.no_grad():
                preds = torch.argmax(pred_mask, dim=1)
                visualize_batch(model, x, y, preds, batch_idx, epoch, mean=mean, std=std)
        
        # Check if it's time to visualize
        #if global_step % visualize_interval == 0:
            #visualize_predictions(model, val_dataloader, device, mean, std, n_samples=3, step=global_step)
    
    model.eval()
    val_loss_list = []
    val_acc_list = []
    val_miou_list = []
    val_loop = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
    with torch.no_grad():
        for x, y in val_loop:
            x, y = x.to(device), y.to(device)
            pred_mask = model(x)
            val_loss = criterion(pred_mask, y)
            val_loss_list.append(val_loss.cpu().detach().numpy())

            preds = torch.argmax(pred_mask, dim=1)
            seg_acc = (y == preds).float().mean()
            val_acc_list.append(seg_acc.cpu().numpy())

            miou = calculate_miou(preds.cpu().numpy(), y.cpu().numpy(), num_classes=9)
            val_miou_list.append(miou)

            val_loop.set_postfix(loss=np.mean(val_loss_list), acc=np.mean(val_acc_list))

    val_miou = np.mean(val_miou_list)
    train_loss = np.mean(train_loss_list)
    train_acc = np.mean(train_acc_list)
    val_loss = np.mean(val_loss_list)
    val_acc = np.mean(val_acc_list)
    
    print(f'Epoch {epoch+1} - loss : {train_loss:.5f} - acc : {train_acc:.2f} - val loss : {val_loss:.5f} - val acc : {val_acc:.2f} - val mIoU : {val_miou:.2f}')
    plot_losses_data.append([epoch, train_loss, val_loss])

    is_best = val_loss < min_loss
    if is_best:
        min_loss = val_loss
        torch.save(model.state_dict(), f'./saved_models/unet_epoch_{epoch+1}_{val_loss:.5f}.pt')

    lr_scheduler.step()
    print(f"Epoch {epoch+1}: Learning rate set to {optimizer.param_groups[0]['lr']}")

    #if global_step % visualize_interval == 0:
           #visualize_predictions(model, val_dataloader, device, mean, std, n_samples=5, step=global_step)

np.save('plot_losses.npy', np.array(plot_losses_data))
plot_losses()
