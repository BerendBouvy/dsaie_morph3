# 

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with 3D Convolution"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, drop_channels=True, p_drop=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3D convolution layer added here with a smaller temporal kernel size
        self.conv3d = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size//2, kernel_size//2), bias=False)
        
        if drop_channels:
            self.double_conv.add_module('dropout', nn.Dropout2d(p=p_drop))

    def forward(self, x):
        x = self.double_conv(x)  # Applies 2D convolution
        # Reshape x for 3D convolution (batch_size, mid_channels, 1, height, width)
        x = x.unsqueeze(2)
        x = self.conv3d(x)  # Applies 3D convolution
        x = x.squeeze(2)  # Remove the extra dimension
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None):
        super().__init__()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(2)
        self.pool_conv = nn.Sequential(
            self.pooling,
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, init_hid_dim=8, kernel_size=3, pooling='max', bilinear=False, drop_channels=False, p_drop=None):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_hid_dim = init_hid_dim 
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.drop_channels = drop_channels
        self.p_drop = p_drop

        hid_dims = [init_hid_dim * (2**i) for i in range(5)]
        self.hid_dims = hid_dims

        # Initial 2D Convolution
        self.inc = DoubleConv(n_channels, hid_dims[0], kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

        # Downscaling with 2D Convolution followed by MaxPooling
        self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop)
        self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop)
        self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop)

        # Temporal Convolution with 3D Convolution
        self.temporal_conv = nn.Conv3d(hid_dims[3], hid_dims[3], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Downscaling with 2D Convolution followed by MaxPooling
        factor = 2 if bilinear else 1
        self.down4 = Down(hid_dims[3], hid_dims[4] // factor, kernel_size, pooling, drop_channels, p_drop)

        # Upscaling with 2D Convolution followed by Double Convolution
        self.up1 = Up(hid_dims[4], hid_dims[3] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up2 = Up(hid_dims[3], hid_dims[2] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up3 = Up(hid_dims[2], hid_dims[1] // factor, kernel_size, bilinear, drop_channels, p_drop)

        # Final 2D Convolution for output
        self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop)
        self.outc = OutConv(hid_dims[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Temporal Convolution
        x4_temporal = x4.unsqueeze(2)  # Add temporal dimension
        x4_temporal = self.temporal_conv(x4_temporal)
        x4 = x4_temporal.squeeze(2)  # Remove temporal dimension

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.sigmoid(x)
        return x