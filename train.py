import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
from losses import CombinedLoss
from models.deeplabv3 import DeepLabV3Plus
from models.unet import UNet
from dataloader import WhisperDataLoader, get_dataloaders
from plots import plot_losses, visualize_batch
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DEBUG = False

data_path = 'MMSeg-YREB'
num_epochs = 150
batch_size = 32
num_classes = 10

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Options: 'miou' or 'lovasz'
secondary_loss_type = 'miou'

criterion = CombinedLoss(weight_ce=None, weight_secondary=1.0, secondary_loss=secondary_loss_type).to(device)

mean = [0.485] * 10 + [0.5] * 11
std = [0.229] * 10 + [0.1] * 11 

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

model = UNet(n_channels=21,n_classes=num_classes,bilinear=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

T_max = num_epochs
eta_min = 1e-6
lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

os.makedirs('./saved_models', exist_ok=True)
min_loss = float('inf')
plot_losses_data = []

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

        optimizer.zero_grad()
        pred_mask = model(x)  

        loss = criterion(pred_mask, y)
        
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.cpu().detach().numpy())

        seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
        train_acc_list.append(seg_acc.item())

        train_loop.set_postfix(loss=np.mean(train_loss_list), acc=np.mean(train_acc_list))
        
        if batch_idx % 200 == 0:
            with torch.no_grad():
                preds = torch.argmax(pred_mask, dim=1)
                visualize_batch(model, x, y, preds, batch_idx, epoch, mean=mean, std=std)
        
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

            miou = calculate_miou(preds.cpu().numpy(), y.cpu().numpy(), num_classes=num_classes)
            val_miou_list.append(miou)

            val_loop.set_postfix(loss=np.mean(val_loss_list), acc=np.mean(val_acc_list))
    
    val_miou = np.mean(val_miou_list)
    train_loss = np.mean(train_loss_list)
    train_acc = np.mean(train_acc_list)
    val_loss = np.mean(val_loss_list)
    val_acc = np.mean(val_acc_list)
    
    print(f'Epoch {epoch+1} - loss : {train_loss:.5f} - acc : {train_acc:.2f} - val loss : {val_loss:.5f} - val acc : {val_acc:.2f} - val mIoU : {val_miou:.4f}')
    plot_losses_data.append([epoch, train_loss, val_loss])

    is_best = val_loss < min_loss
    if is_best:
        min_loss = val_loss
        torch.save(model.state_dict(), f'./saved_models/UNET_21_mod_new_no_w_epoch_{epoch+1}_{val_loss:.5f}.pt')

    lr_scheduler.step()
    print(f"Epoch {epoch+1}: Learning rate set to {optimizer.param_groups[0]['lr']}")

np.save('plot_losses.npy', np.array(plot_losses_data))
plot_losses()