import numpy as np
from dataloader import get_dataloaders

# Load a batch of labels
train_dataloader, val_dataloader, test_dataloader = get_dataloaders("MMSeg-YREB", batch_size=32)
all_labels = []
for _, y in train_dataloader:
    all_labels.append(y.numpy())
all_labels = np.concatenate(all_labels)

# Check the unique class indices
unique_classes = np.unique(all_labels)
print("Unique class indices in the dataset:", unique_classes)
