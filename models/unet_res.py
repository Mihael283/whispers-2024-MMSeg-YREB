import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Check size differences between g1 and x1
        if g1.size() != x1.size():
            diffY = x1.size()[2] - g1.size()[2]
            diffX = x1.size()[3] - g1.size()[3]
            g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Add a Conv2d layer to adjust the input channels of x1 to match x2
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Adjust channels here
        self.conv = DoubleConv(out_channels * 2, out_channels)  # DoubleConv after concatenation
        self.att = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.se = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust x1 channels to match x2
        if x1.size(1) != x2.size(1):
            x1 = self.adjust_channels(x1)

        x2 = self.att(g=x1, x=x2)

        # Padding if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class ImprovedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ImprovedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.backbone = resnet34(pretrained=True)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = self.backbone.layer1  # 64 channels
        self.down2 = self.backbone.layer2  # 128 channels
        self.down3 = self.backbone.layer3  # 256 channels
        self.down4 = self.backbone.layer4  # 512 channels

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Deep supervision
        self.deep3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.deep2 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.deep1 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        d3 = self.deep3(x)
        x = self.up2(x, x3)
        d2 = self.deep2(x)
        x = self.up3(x, x2)
        d1 = self.deep1(x)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
