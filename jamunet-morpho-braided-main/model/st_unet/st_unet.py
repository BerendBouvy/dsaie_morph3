import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DoubleConv(nn.Module):
    """
    Double Conv Block with optional 3D convolution for temporal modeling.
    (Conv2D => BatchNorm => ReLU) * 2 + optional Dropout and 3D Conv

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after 3D convolution.
        mid_channels (int, optional): Number of intermediate channels, defaults to out_channels.
        kernel_size (int): Kernel size for Conv2D and Conv3D.
        drop_channels (bool): Whether to apply Dropout2D after the first convolution.
        p_drop (float): Dropout2D probability (relevant if drop_channels=True).
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, drop_channels=True, p_drop=None, seed=None):
        super().__init__()
        if seed is not None:
            self._set_seed(seed)
        
        if not mid_channels:
            mid_channels = out_channels  # Default intermediate channels to out_channels

        # Define the double Conv2D block
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add optional Dropout after the first Conv2D
        if drop_channels:
            self.double_conv.add_module('dropout', nn.Dropout2d(p=p_drop))

        # Add a Conv3D for temporal modeling
        self.conv3d = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), 
                                padding=(0, kernel_size // 2, kernel_size // 2), bias=False)

    def forward(self, x):
        x = self.double_conv(x)  # Apply double Conv2D block
        # Convert to 3D convolutions: Add temporal dimension
        x = x.unsqueeze(2)  # Add depth (temporal) channel (B, C, 1, H, W)
        x = self.conv3d(x)  # Apply Conv3D
        x = x.squeeze(2)  # Remove added temporal dimension
        return x

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for convolutional layers
        torch.backends.cudnn.benchmark = False    # Disable auto-tuning for performance
    
class Down(nn.Module):
    """
    Downsampling Block with Pooling and DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after DoubleConv.
        kernel_size (int): Kernel size for Conv2D and Conv3D layers.
        pooling (str): Pooling method ('max' for MaxPooling, 'avg' for AvgPooling).
        drop_channels (bool): Whether to apply Dropout2D.
        p_drop (float): Dropout2D probability.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None, seed=None):
        super().__init__()
        if seed is not None:
            self._set_seed(seed)
        
        # Define pooling type
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(2)
        
        # Combine pooling and DoubleConv
        self.pool_conv = nn.Sequential(
            self.pooling,
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, seed=seed)
        )

    def forward(self, x):
        return self.pool_conv(x)  # Apply pooling followed by DoubleConv

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
class Up(nn.Module):
    """
    Upsampling Block with Bilinear or Transposed Convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after DoubleConv.
        kernel_size (int): Kernel size for Conv2D and Conv3D layers.
        bilinear (bool): Whether to use bilinear interpolation or transposed convolution for upsampling.
        drop_channels (bool): Whether to apply Dropout2D.
        p_drop (float): Dropout2D probability.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None, seed=None):
        super().__init__()
        if seed is not None:
            self._set_seed(seed)
        
        if bilinear:
            # Upsampling with bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, seed=seed)
        else:
            # Upsampling with transposed convolution
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, seed=seed)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample x1
        # Handle spatial size mismatches due to pooling or cropping
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)  # Concatenate skip connection
        return self.conv(x)  # Apply DoubleConv

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class OutConv(nn.Module):
    """
    Output Convolution Block to reduce feature maps to the desired number of output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the final prediction.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, in_channels, out_channels, seed=None):
        super(OutConv, self).__init__()
        if seed is not None:
            self._set_seed(seed)
        
        # Define a 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  # Apply 1x1 convolution

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class UNet3D(nn.Module):  # UNet3D Model
    def __init__(self, n_channels, n_classes, init_hid_dim=8, kernel_size=3, pooling='max', bilinear=False, drop_channels=False, p_drop=None, seed=None):
        super(UNet3D, self).__init__()
        # """
        # UNet-based architecture extended with temporal 3D convolutions.

        # Args:
        #     n_channels (int): Number of input channels.
        #     n_classes (int): Number of output channels/classes.
        #     init_hid_dim (int): Initial number of hidden dimensions (doubles at each layer).
        #     kernel_size (int): Kernel size for convolution layers.
        #     pooling (str): Pooling method ('max' or alternative).
        #     bilinear (bool): Whether to use bilinear interpolation for upscaling.
        #     drop_channels (bool): Whether to apply channel dropout.
        #     p_drop (float): Dropout probability (if drop_channels is True).
        #     seed (int, optional): Optional random seed for reproducibility.
        # """
        
        if seed is not None:
            self._set_seed(seed)  # Set seed for reproducibility

        # Initialize model parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_hid_dim = init_hid_dim
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.drop_channels = drop_channels
        self.p_drop = p_drop

        # Calculate hidden dimensions for each UNet layer
        hid_dims = [init_hid_dim * (2 ** i) for i in range(5)]  # E.g., [8, 16, 32, 64, 128]
        self.hid_dims = hid_dims

        # UNet Encoder: Initial 2D convolution
        self.inc = DoubleConv(n_channels, hid_dims[0], kernel_size=kernel_size, 
                              drop_channels=drop_channels, p_drop=p_drop, seed=seed)

        # Downscaling layers with convolution and pooling
        self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop, seed=seed)
        self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop, seed=seed)
        self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop, seed=seed)

        # Temporal Convolution: 3D convolution over temporal dimension
        self.temporal_conv = nn.Conv3d(hid_dims[3], hid_dims[3], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Further downscaling (optional bilinear upscaling adjustment)
        factor = 2 if bilinear else 1  # Adjust hidden dimensions if bilinear upscaling is enabled
        self.down4 = Down(hid_dims[3], hid_dims[4] // factor, kernel_size, pooling, drop_channels, p_drop, seed=seed)

        # UNet Decoder: Upscaling layers with convolution
        self.up1 = Up(hid_dims[4], hid_dims[3] // factor, kernel_size, bilinear, drop_channels, p_drop, seed=seed)
        self.up2 = Up(hid_dims[3], hid_dims[2] // factor, kernel_size, bilinear, drop_channels, p_drop, seed=seed)
        self.up3 = Up(hid_dims[2], hid_dims[1] // factor, kernel_size, bilinear, drop_channels, p_drop, seed=seed)
        self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop, seed=seed)

        # Final convolution: Produces the output segmentation map
        self.outc = OutConv(hid_dims[0], n_classes, seed=seed)
        self.sigmoid = nn.Sigmoid()  # Apply sigmoid for binary segmentation

    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes, height, width).
        """
        # Encoder: Series of downscaling convolutions
        x1 = self.inc(x)       # Initial 2D convolution
        x2 = self.down1(x1)    # Downsampling 1
        x3 = self.down2(x2)    # Downsampling 2
        x4 = self.down3(x3)    # Downsampling 3

        # Temporal Convolution: Apply 3D convolution along an added temporal dimension
        x4_temporal = x4.unsqueeze(2)  # Temporarily add temporal (depth) dimension
        x4_temporal = self.temporal_conv(x4_temporal)  # 3D convolution
        x4 = x4_temporal.squeeze(2)  # Remove temporal dimension post-convolution

        # Further downsampling
        x5 = self.down4(x4)

        # Decoder: Series of upscaling convolutions with skip connections
        x = self.up1(x5, x4)   # Upsampling 1 + skip connection
        x = self.up2(x, x3)    # Upsampling 2 + skip connection
        x = self.up3(x, x2)    # Upsampling 3 + skip connection
        x = self.up4(x, x1)    # Upsampling 4 + skip connection

        # Final output
        x = self.outc(x)       # Final convolution for output segmentation map
        x = self.sigmoid(x)    # Sigmoid activation for binary segmentation
        return x  # Output tensor

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance
            
class UNet3D_full(nn.Module):  #2.
    def __init__(self, in_channels=1, out_channels=1, init_features=8, temporal=3, seed=None):
        """
        Full UNet architecture with 3D convolutions, including temporal modeling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels/classes.
            init_features (int): Initial number of hidden dimensions (doubles at each layer).
            temporal (int): Temporal kernel size for 3D convolutions (time dimension).
            seed (int, optional): Optional random seed for reproducibility.
        """
        super(UNet3D_full, self).__init__()
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)

        # Initialize feature map size
        features = init_features

        # ENCODER BLOCKS
        # First encoder block
        self.encoder1 = self._block(in_channels, features, temporal_kernel=temporal)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Reduce spatial dimensions

        # Second encoder block
        self.encoder2 = self._block(features, features * 2, temporal_kernel=min(temporal, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Third encoder block
        self.encoder3 = self._block(features * 2, features * 4, temporal_kernel=min(temporal, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Fourth encoder block
        self.encoder4 = self._block(features * 4, features * 8, temporal_kernel=min(temporal, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # BOTTLENECK
        self.bottleneck = self._block(features * 8, features * 16, temporal_kernel=min(temporal, 3))

        # DECODER BLOCKS
        # Upsample and decode the compressed representation
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder4 = self._block(features * 16, features * 8, temporal_kernel=min(temporal, 3))
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder3 = self._block(features * 8, features * 4, temporal_kernel=min(temporal, 3))
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder2 = self._block(features * 4, features * 2, temporal_kernel=min(temporal, 3))
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder1 = self._block(features * 2, features, temporal_kernel=min(temporal, 3))

        # FINAL CONVOLUTION
        self.conv = nn.Conv3d(features, out_channels, kernel_size=(4, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input tensor with shape (B, in_channels, T, H, W).
        
        Returns:
            torch.Tensor: Output tensor with shape (B, out_channels, H, W).
        """
        # Add channel dimension if input tensor is 4D (e.g., B x T x H x W)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        # ENCODER
        enc1 = self.encoder1(x)               # First encoder block
        enc2 = self.encoder2(self.pool1(enc1))  # Second encoder block + pooling
        enc3 = self.encoder3(self.pool2(enc2))  # Third encoder block + pooling
        enc4 = self.encoder4(self.pool3(enc3))  # Fourth encoder block + pooling

        # BOTTLENECK
        bottleneck = self.bottleneck(self.pool4(enc4))  # Bottleneck layer

        # DECODER
        # Decoder 4: Upsample and concatenate with encoder 4 output
        dec4 = self.upconv4(bottleneck)
        dec4 = self._crop_and_concat(enc4, dec4)
        dec4 = self.decoder4(dec4)

        # Decoder 3: Upsample and concatenate with encoder 3 output
        dec3 = self.upconv3(dec4)
        dec3 = self._crop_and_concat(enc3, dec3)
        dec3 = self.decoder3(dec3)

        # Decoder 2: Upsample and concatenate with encoder 2 output
        dec2 = self.upconv2(dec3)
        dec2 = self._crop_and_concat(enc2, dec2)
        dec2 = self.decoder2(dec2)

        # Decoder 1: Upsample and concatenate with encoder 1 output
        dec1 = self.upconv1(dec2)
        dec1 = self._crop_and_concat(enc1, dec1)
        dec1 = self.decoder1(dec1)

        # FINAL OUTPUT
        output = torch.sigmoid(self.conv(dec1))  # Final sigmoid activation for probabilities
        output = output.squeeze(2)  # Remove temporal dimension (T=1)
        return output.squeeze(1)  # Remove channel dimension (C=1)

    def _block(self, in_channels, features, temporal_kernel):
        """
        Create a convolutional block with BatchNorm and ReLU.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.
            temporal_kernel (int): Temporal kernel size for 3D convolutions.

        Returns:
            nn.Sequential: Convolutional block.
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=(temporal_kernel, 3, 3), 
                      padding=(temporal_kernel // 2, 1, 1), bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=(temporal_kernel, 3, 3), 
                      padding=(temporal_kernel // 2, 1, 1), bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        )

    def _crop_and_concat(self, enc, dec):
        """
        Crop and concatenate encoder output with decoder input.

        Args:
            enc (torch.Tensor): Encoder output.
            dec (torch.Tensor): Decoder input to be concatenated.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        diffZ = enc.size(2) - dec.size(2)
        diffY = enc.size(3) - dec.size(3)
        diffX = enc.size(4) - dec.size(4)
        dec = F.pad(dec, [diffX // 2, diffX - diffX // 2,  # Pad width
                          diffY // 2, diffY - diffY // 2,  # Pad height
                          diffZ // 2, diffZ - diffZ // 2]) # Pad depth
        return torch.cat((enc, dec), dim=1)

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance
        
class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=8, seed=None):
        """
        UNet 2D architecture for image segmentation tasks.

        Args:
            in_channels (int): Number of input channels (e.g., RGB=3 or grayscale=1).
            out_channels (int): Number of output channels/classes (e.g., 1 for binary segmentation).
            init_features (int): Number of initial feature maps (doubles in each layer).
            seed (int, optional): Random seed for reproducibility.
        """
        super(UNet2D, self).__init__()
        
        if seed is not None:
            self._set_seed(seed)  # Set seed for reproducibility

        # ENCODER: Downsampling path
        self.encoder1 = self._block(in_channels, init_features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(init_features, init_features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(init_features * 2, init_features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(init_features * 4, init_features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK
        self.bottleneck = self._block(init_features * 8, init_features * 16, name="bottleneck")

        # DECODER
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(init_features * 16, init_features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(init_features * 8, init_features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(init_features * 4, init_features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder1 = self._block(init_features * 2, init_features, name="dec1")

        # FINAL CONVOLUTION
        self.conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # BOTTLENECK
        bottleneck = self.bottleneck(self.pool4(enc4))

        # DECODER
        dec4 = self.upconv4(bottleneck)
        diffY = enc4.size(2) - dec4.size(2)
        diffX = enc4.size(3) - dec4.size(3)
        dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        diffY = enc3.size(2) - dec3.size(2)
        diffX = enc3.size(3) - dec3.size(3)
        dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        diffY = enc2.size(2) - dec2.size(2)
        diffX = enc2.size(3) - dec2.size(3)
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        diffY = enc1.size(2) - dec1.size(2)
        diffX = enc1.size(3) - dec1.size(3)
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class UNet_semi3D(nn.Module): #4 & #7
    def __init__(self, in_channels, out_channels=1, init_features=8, seed=None):
        """
        Semi-3D UNet architecture for image segmentation tasks.

        Args:
            in_channels (int): Number of input channels (temporal slices or image frames).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            init_features (int): Number of initial feature maps (doubles with each layer).
            seed (int, optional): Random seed for reproducibility.
        """
        super(UNet_semi3D, self).__init__()

        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)

        # ENCODER: Temporal convolution with 3D Convolutions (captures temporal features)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=(4, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True)
        )

        # Temporal dimension transition: Collapse temporal dimension with a 1x1x1 convolution
        self.squeeze_temporal = nn.Conv3d(init_features, init_features, kernel_size=(1, 1, 1), bias=False)

        # ENCODER: Spatial feature extraction with 2D Convolutions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions
        self.encoder2 = self._block_2d(init_features, init_features * 2)  # 8 -> 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block_2d(init_features * 2, init_features * 4)  # 16 -> 32
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block_2d(init_features * 4, init_features * 8)  # 32 -> 64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK
        self.bottleneck = self._block_2d(init_features * 8, init_features * 16)  # Bottleneck (64 -> 128)

        # DECODER: Upsampling path with skip connections
        self.upconv4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (128 -> 64)
        self.decoder4 = self._block_2d(init_features * (16 + 8), init_features * 8)  # 128+64 -> 64

        self.upconv3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (64 -> 32)
        self.decoder3 = self._block_2d(init_features * (8 + 4), init_features * 4)  # 64+32 -> 32

        self.upconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (32 -> 16)
        self.decoder2 = self._block_2d(init_features * (4 + 2), init_features * 2)  # 32+16 -> 16

        self.upconv1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (16 -> 8)
        self.decoder1 = self._block_2d(init_features * (2 + 1), init_features)  # 16+8 -> 8

        # FINAL OUTPUT
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)  # Final output layer (8 -> out_channels)

    @staticmethod
    def _block_2d(in_channels, features):
        """
        Creates a 2D convolutional block with BatchNorm and ReLU activations.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block consisting of two Conv2D layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),  # First Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),  # Second Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward propagation through the semi-3D UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, temporal_depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # ENCODER
        if len(x.shape) == 4:  # Add an extra channel dimension if input lacks temporal dimension
            x = x.unsqueeze(1)
        
        # Temporal dimension is retained in the initial 3D convolution step
        enc1 = self.encoder1(x).squeeze(2)  # Apply 3D encoder and collapse temporal dimension
        enc2 = self.encoder2(self.pool1(enc1))  # Downsample + Encode (8 -> 16)
        enc3 = self.encoder3(self.pool2(enc2))  # Downsample + Encode (16 -> 32)
        enc4 = self.encoder4(self.pool3(enc3))  # Downsample + Encode (32 -> 64)

        # BOTTLENECK
        bottleneck = self.bottleneck(self.pool4(enc4))  # Bottleneck (64 -> 128)

        # DECODER (with skip connections)
        # Decoder layer 4
        dec4 = self.upconv4(bottleneck)  # Upsample (128 -> 64)
        dec4 = F.pad(dec4, [enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2), 0])  # Pad spatial dimensions to match
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate with corresponding encoder output
        dec4 = self.decoder4(dec4)  # Decode (128 -> 64)

        # Decoder layer 3
        dec3 = self.upconv3(dec4)  # Upsample (64 -> 32)
        dec3 = F.pad(dec3, [enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2), 0])
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection 3
        dec3 = self.decoder3(dec3)

        # Decoder layer 2
        dec2 = self.upconv2(dec3)  # Upsample (32 -> 16)
        dec2 = F.pad(dec2, [enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2), 0])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        # Decoder layer 1
        dec1 = self.upconv1(dec2)  # Upsample (16 -> 8)
        dec1 = F.pad(dec1, [enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2), 0])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # FINAL OUTPUT
        return torch.sigmoid(self.final_conv(dec1))  # Final activation using sigmoid for probabilities

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance


class UNet_semi3D_drop(nn.Module):  #5 BCE, #17 Dice  
    def __init__(self, in_channels, out_channels=1, init_features=8, seed=None, dropout_rate=0.1):
        """
        Semi-3D UNet architecture with Dropout for segmentation tasks.

        Args:
            in_channels (int): Number of input channels (temporal slices or image frames).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            init_features (int): Number of initial feature maps (doubles with each layer).
            seed (int, optional): Random seed for reproducibility.
            dropout_rate (float): Dropout probability for 2D layers.
        """
        super(UNet_semi3D_drop, self).__init__()

        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)

        self.dropout_rate = dropout_rate  # Dropout probability for regularization

        # ENCODER: Temporal convolution with 3D Convolutions (captures temporal features)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=(4, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True)
        )

        # Temporal dimension transition: Collapse temporal dimension with a 1x1x1 convolution
        self.squeeze_temporal = nn.Conv3d(init_features, init_features, kernel_size=(1, 1, 1), bias=False)

        # ENCODER: Spatial feature extraction with 2D Layers (with Dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions
        self.encoder2 = self._block_2d(init_features, init_features * 2)  # 8 -> 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block_2d(init_features * 2, init_features * 4)  # 16 -> 32
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block_2d(init_features * 4, init_features * 8)  # 32 -> 64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK
        self.bottleneck = self._block_2d(init_features * 8, init_features * 16)  # Bottleneck (64 -> 128)

        # DECODER: Upsampling path with skip connections and Dropout
        self.upconv4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (128 -> 64)
        self.decoder4 = self._block_2d(init_features * (16 + 8), init_features * 8)  # Input: 128+64 -> 64

        self.upconv3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (64 -> 32)
        self.decoder3 = self._block_2d(init_features * (8 + 4), init_features * 4)  # Input: 64+32 -> 32

        self.upconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (32 -> 16)
        self.decoder2 = self._block_2d(init_features * (4 + 2), init_features * 2)  # Input: 32+16 -> 16

        self.upconv1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (16 -> 8)
        self.decoder1 = self._block_2d(init_features * (2 + 1), init_features)  # Input: 16+8 -> 8

        # FINAL OUTPUT
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)  # Final output layer (8 -> out_channels)

    def _block_2d(self, in_channels, features):
        """
        2D Convolutional block with Dropout, BatchNorm, and ReLU.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block consisting of two Conv2D layers and Dropout.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),  # Dropout layer for regularization

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward propagation through the semi-3D UNet with dropouts.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, temporal_slices, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # ENCODER: Capture features
        if len(x.shape) == 4:  # Add channel dimension if temporal dimension is missing
            x = x.unsqueeze(1)

        enc1 = self.encoder1(x).squeeze(2)  # Apply 3D convolution and collapse temporal dimension
        enc2 = self.encoder2(self.pool1(enc1))  # Downsample + Encode
        enc3 = self.encoder3(self.pool2(enc2))  # Downsample + Encode
        enc4 = self.encoder4(self.pool3(enc3))  # Downsample + Encode

        # BOTTLENECK
        bottleneck = self.bottleneck(self.pool4(enc4))  # Compress features in bottleneck

        # DECODER: Reconstruct output with skip connections
        # Decoder layer 4
        dec4 = self.upconv4(bottleneck)  # Upsample
        dec4 = F.pad(dec4, [enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2), 0])  # Pad spatial dimensions
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate with corresponding encoder features
        dec4 = self.decoder4(dec4)

        # Decoder layer 3
        dec3 = self.upconv3(dec4)  # Upsample
        dec3 = F.pad(dec3, [enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2), 0])
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate
        dec3 = self.decoder3(dec3)

        # Decoder layer 2
        dec2 = self.upconv2(dec3)  # Upsample
        dec2 = F.pad(dec2, [enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2), 0])
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate
        dec2 = self.decoder2(dec2)

        # Decoder layer 1
        dec1 = self.upconv1(dec2)  # Upsample
        dec1 = F.pad(dec1, [enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2), 0])
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenate
        dec1 = self.decoder1(dec1)

        # FINAL OUTPUT
        return torch.sigmoid(self.final_conv(dec1))  # Output probabilities using Sigmoid activation

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance


class UNet_semi3D_shallow8(nn.Module):  #6, #15 BCE, #16 Dice
    def __init__(self, in_channels, out_channels=1, init_features=8, seed=None):
        """
        A semi-3D shallow UNet with reduced depth, using a bottleneck of 8 feature maps.

        Args:
            in_channels (int): Number of input channels (e.g., temporal slices or image frames).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            init_features (int): Number of initial feature maps (default=8).
            seed (int, optional): Random seed for reproducibility.
        """
        super(UNet_semi3D_shallow8, self).__init__()

        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)

        # ENCODER: Initial 3D convolution block to process spatio-temporal input
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=(4, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
        )

        # Transition to 2D: Squeeze temporal dimension with a 1x1x1 convolution
        self.squeeze_temporal = nn.Conv3d(init_features, init_features, kernel_size=(1, 1, 1), bias=False)

        # ENCODER: Further downsampling with 2D convolutions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions by a factor of 2
        self.encoder2 = self._block_2d(init_features, init_features * 2)  # 8 -> 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block_2d(init_features * 2, init_features * 4)  # 16 -> 32
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK: Intermediate narrow layer for most compressed features
        self.bottleneck = self._block_2d(init_features * 4, init_features * 8)  # 32 -> 64

        # DECODER: Upsampling path with skip connections
        self.upconv3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 64 -> 32
        self.decoder3 = self._block_2d(init_features * (8 + 4), init_features * 4)  # 64+32 -> 32

        self.upconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 32 -> 16
        self.decoder2 = self._block_2d(init_features * (4 + 2), init_features * 2)  # 32+16 -> 16

        self.upconv1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 16 -> 8
        self.decoder1 = self._block_2d(init_features * (2 + 1), init_features)  # 16+8 -> 8

        # FINAL OUTPUT: Single-channel output for binary segmentation
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)  # Final output layer

    @staticmethod
    def _block_2d(in_channels, features):
        """
        2D convolutional block with BatchNorm and ReLU.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.

        Returns:
            nn.Sequential: Block of two Conv2D layers with BatchNorm and ReLU activations.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward propagation through the semi-3D shallow UNet.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, temporal_depth, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        """
        # ENCODER: Process input through the encoder
        if len(x.shape) == 4:  # Add temporal channel if the input lacks one
            x = x.unsqueeze(1)

        enc1 = self.encoder1(x).squeeze(2)  # Initial 3D convolution + Collapse temporal dimension
        enc2 = self.encoder2(self.pool1(enc1))  # Downsample + Encode at level 2
        enc3 = self.encoder3(self.pool2(enc2))  # Downsample + Encode at level 3

        # BOTTLENECK: Compress to narrow feature representation
        bottleneck = self.bottleneck(self.pool3(enc3))  # 32 -> 64

        # DECODER: Reconstruct features with skip connections
        # Decoder level 3
        dec3 = self.upconv3(bottleneck)  # Upsample bottleneck 64 -> 32
        dec3 = F.pad(dec3, [enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2), 0])  # Pad to match encoder size
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate skip connection
        dec3 = self.decoder3(dec3)

        # Decoder level 2
        dec2 = self.upconv2(dec3)  # Upsample 32 -> 16
        dec2 = F.pad(dec2, [enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2), 0])  # Pad to match encoder size
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate skip connection
        dec2 = self.decoder2(dec2)

        # Decoder level 1
        dec1 = self.upconv1(dec2)  # Upsample 16 -> 8
        dec1 = F.pad(dec1, [enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2), 0])  # Pad to match encoder size
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenate skip connection
        dec1 = self.decoder1(dec1)

        # FINAL OUTPUT: Sigmoid activation for probabilities
        return torch.sigmoid(self.final_conv(dec1))

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance

class UNet_semi3D_4_8_resBN_Dice_drop(nn.Module):  #10 & #11 & 12  experimental UNet, both Dice and BCE
    def __init__(self, in_channels, out_channels=1, init_features=8, seed=None, dropout_rate=0.1):
        """
        Experimental Semi-3D UNet with Residual Connections, Dropout, and BatchNorm
        
            This network integrates 3D convolution for temporal handling,
            residual blocks in the bottleneck, 2D convolutions for spatial features, 
            and dropout for regularization. Supports both BCE and Dice loss.
        
        Args:
            in_channels (int): Number of input channels (e.g., temporal slices or image frames).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            init_features (int): Number of initial feature maps (default=8).
            seed (int, optional): Random seed for reproducibility.
            dropout_rate (float): Dropout probability for dropout layers.
        """
        super(UNet_semi3D_4_8_resBN_Dice_drop, self).__init__()

        if seed is not None:
            self._set_seed(seed)  # Set seed for reproducibility

        self.dropout_rate = dropout_rate  # Dropout probability for 2D convolutional layers

        # INITIAL ENCODER: Temporal Modeling with 3D Convolutions
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=(4, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True)
        )

        # Temporal Feature Transition: Reduce temporal dimension to 1
        self.squeeze_temporal = nn.Conv3d(init_features, init_features, kernel_size=(1, 1, 1), bias=False)

        # ENCODER: Further downsampling of spatial dimensions with 2D convolutions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        self.encoder2 = self._block_2d(init_features, init_features * 2)  # 8 -> 16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block_2d(init_features * 2, init_features * 4)  # 16 -> 32
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block_2d(init_features * 4, init_features * 8)  # 32 -> 64

        # BOTTLENECK: Residual Block Integration
        self.bottleneck = self._residual_block(init_features * 8, init_features * 16)  # Residual connection at deepest layer

        # DECODER: Upsampling with skip connections and dropout regularization
        self.upconv4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 128 -> 64
        self.decoder4 = self._block_2d(init_features * (16 + 8), init_features * 8)  # Input: 128+64 -> 64

        self.upconv3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 64 -> 32
        self.decoder3 = self._block_2d(init_features * (8 + 4), init_features * 4)  # Input: 64+32 -> 32

        self.upconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 32 -> 16
        self.decoder2 = self._block_2d(init_features * (4 + 2), init_features * 2)  # Input: 32+16 -> 16

        self.upconv1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample 16 -> 8
        self.decoder1 = self._block_2d(init_features * (2 + 1), init_features)  # Input: 16+8 -> 8

        # FINAL OUTPUT: Single-channel output for binary segmentation
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)  # Final Conv2D

    def _block_2d(self, in_channels, features):
        """
        2D Convolutional Block with Dropout, BatchNorm, and ReLU

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),  # Conv2D
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate)  # Regularization with Dropout
        )

    def _residual_block(self, in_channels, features):
        """
        Residual Block with Dropout for regularization and BatchNorm.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels/features.
        """
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, features, dropout_rate):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(features)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(features)
                self.dropout = nn.Dropout2d(p=dropout_rate)
                self.downsample = None  # For aligning dimensions when needed
                if in_channels != features:
                    self.downsample = nn.Conv2d(in_channels, features, kernel_size=1, bias=False)

            def forward(self, x):
                residual = x
                if self.downsample is not None:
                    residual = self.downsample(x)  # Adjust the residual if dimensions differ
                x = self.relu(self.bn1(self.conv1(x)))  # Conv1
                x = self.dropout(x)  # Apply dropout
                x = self.bn2(self.conv2(x))  # Conv2
                x = self.dropout(x)  # Apply dropout
                return self.relu(x + residual)  # Residual connection

        return ResidualBlock(in_channels, features, self.dropout_rate)

    def forward(self, x):
        """
        Forward propagation through the network.
        """
        if len(x.shape) == 4:  # Add channel dimension if temporal dimension is missing
            x = x.unsqueeze(1)

        # ENCODER
        enc1 = self.encoder1(x).squeeze(2)  # Initial 3D convolution + collapse temporal dimension
        enc2 = self.encoder2(self.pool1(enc1))  # Downsampling + Encode
        enc3 = self.encoder3(self.pool2(enc2))  # Downsampling + Encode
        enc4 = self.encoder4(self.pool3(enc3))  # Downsampling + Encode

        # BOTTLENECK
        bottleneck = self.bottleneck(enc4)  # Residual bottleneck layer

        # DECODER
        dec4 = self.upconv4(bottleneck)  # Upsample
        dec4 = F.pad(dec4, [enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2), 0])
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)  # Upsample
        dec3 = F.pad(dec3, [enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2), 0])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)  # Upsample
        dec2 = F.pad(dec2, [enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2), 0])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)  # Upsample
        dec1 = F.pad(dec1, [enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2), 0])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))  # Final Sigmoid Activation

    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance

class UNet_semi3D_resBN_3_8(nn.Module):  #8, if init_features set on 16 => #9 & #13
    def __init__(self, in_channels, out_channels=1, init_features=8, seed=None):
        """
        Semi-3D UNet with BatchNorm and Residual Blocks

            This model uses 3D convolution for initial temporal modeling, 2D convolution for spatial processing,
            and residual blocks in the bottleneck for improved gradient flow. If `init_features` is increased to 16,
            the model becomes a deeper architecture suitable for higher complexity tasks.
        
        Args:
            in_channels (int): Number of input channels (e.g., temporal slices or image frames).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
            init_features (int): Number of feature maps in the first layer (doubles at each layer).
            seed (int, optional): Random seed for reproducibility.
        """
        super(UNet_semi3D_resBN_3_8, self).__init__()

        if seed is not None:
            self._set_seed(seed)  # Set seed for deterministic results

        # ENCODER: Initial temporal modeling using 3D convolution
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=(4, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),  # BatchNorm for 3D features
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),  # 3D Conv
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
        )

        # Transition to 2D: Collapse temporal information with a 1x1x1 convolution
        self.squeeze_temporal = nn.Conv3d(init_features, init_features, kernel_size=(1, 1, 1), bias=False)

        # ENCODER: Spatial feature extraction using 2D convolutions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions (H, W) by 2
        self.encoder2 = self._block_2d(init_features, init_features * 2)  # 8 -> 16 feature maps
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Further downsample spatial dimensions
        self.encoder3 = self._block_2d(init_features * 2, init_features * 4)  # 16 -> 32 feature maps

        # BOTTLENECK: Residual-based convolutional representation
        self.bottleneck = self._residual_block(init_features * 4, init_features * 8)  # 32 -> 64 feature maps

        # DECODER: Upsampling path with skip connections
        self.upconv3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (64 -> 32)
        self.decoder3 = self._block_2d(init_features * (8 + 4), init_features * 4)  # Input: (64 + 32)

        self.upconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (32 -> 16)
        self.decoder2 = self._block_2d(init_features * (4 + 2), init_features * 2)  # Input: (32 + 16)

        self.upconv1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # Upsample (16 -> 8)
        self.decoder1 = self._block_2d(init_features * (2 + 1), init_features)  # Input: (16 + 8)

        # FINAL OUTPUT: Generate binary segmentation map
        self.final_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)  # Output layer

    @staticmethod
    def _block_2d(in_channels, features):
        """
        2D Convolutional Block with BatchNorm and ReLU Activations.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output feature maps.

        Returns:
            nn.Sequential: A convolutional block with two Conv2D layers, BatchNorm, and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _residual_block(in_channels, features):
        """
        Residual Block with two Conv2D layers for the bottleneck.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output feature maps.

        Returns:
            ResidualBlock: A custom residual block.
        """
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, features):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(features)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(features)
                self.downsample = None
                if in_channels != features:  # Adjust dimensions using a 1x1 convolution if necessary
                    self.downsample = nn.Conv2d(in_channels, features, kernel_size=1, bias=False)

            def forward(self, x):
                residual = x  # Save the input as the residual connection
                if self.downsample is not None:
                    residual = self.downsample(x)  # Adjust residual dimensions if they don't match
                x = self.relu(self.bn1(self.conv1(x)))  # First Conv2D layer
                x = self.bn2(self.conv2(x))  # Second Conv2D layer
                return self.relu(x + residual)  # Add the residual connection

        return ResidualBlock(in_channels, features)

    def forward(self, x):
        """
        Forward propagation through the UNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, temporal_slices, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # ENCODER
        if len(x.shape) == 4:  # Add temporal dimension if missing
            x = x.unsqueeze(1)

        enc1 = self.encoder1(x).squeeze(2)  # 3D convolution and collapse temporal dimension
        enc2 = self.encoder2(self.pool1(enc1))  # Downsample + Encode (level 2)
        enc3 = self.encoder3(self.pool2(enc2))  # Downsample + Encode (level 3)

        # BOTTLENECK
        bottleneck = self.bottleneck(enc3)  # Residual block in bottleneck

        # DECODER
        # Decoder level 3
        dec3 = self.upconv3(bottleneck)  # Upsample
        dec3 = F.pad(dec3, [enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2), 0])
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate with encoder output
        dec3 = self.decoder3(dec3)

        # Decoder level 2
        dec2 = self.upconv2(dec3)  # Upsample
        dec2 = F.pad(dec2, [enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2), 0])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        # Decoder level 1
        dec1 = self.upconv1(dec2)  # Upsample
        dec1 = F.pad(dec1, [enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2), 0])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # FINAL OUTPUT
        return torch.sigmoid(self.final_conv(dec1))  # Return the binary segmentation map
    
    def _set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False    # Disables auto-tuning for performance
