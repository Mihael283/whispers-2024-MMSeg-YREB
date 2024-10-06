import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import ResNet101_Weights

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, num_channels=14):
        super(DeepLabV3Plus, self).__init__()
        
        # Load pretrained DeepLabV3 with ResNet101 backbone
        self.deeplab = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
        
        # Modify the first convolutional layer to accept the desired number of input channels
        self.deeplab.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the classifier to output the desired number of classes
        self.deeplab.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.deeplab(x)['out']
        return output