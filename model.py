# coding:utf-8
"""
model.py - MFNet Architecture for RGB + Thermal Semantic Segmentation.
Paper: MFNet: Towards Real-Time Semantic Segmentation for Autonomous Vehicles
       with Multi-Spectral Scenes (IROS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Global Parameters / Configuration
# ==========================================
IN_CHANNELS  = 4       # 3 RGB + 1 Thermal
NUM_CLASSES  = 9       # MFNet has 9 classes (0-8)
DROPOUT_RATE = 0.3     # Spatial Dropout rate to prevent overfitting


class ConvBnLeakyRelu2d(nn.Module):
    """
    Convolution -> BatchNorm -> LeakyReLU block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    """
    Mini Inception block with parallel dilated convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels // 2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels // 2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels // 2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels // 2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels // 2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels // 2, padding=2, dilation=2)

    def forward(self, x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


class MFNet(nn.Module):
    """
    Multi-spectral Fusion Network (MFNet).
    Input:  [B, 4, H, W] where channels 0:3 are RGB and channel 3 is Thermal.
    Output: [B, n_class, H, W] class segmentation logits.
    """

    def __init__(self, n_class=NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        super(MFNet, self).__init__()
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        rgb_ch = [16, 48, 48, 96, 96]
        inf_ch = [16, 16, 16, 36, 36]

        # RGB Encoder
        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])

        # Thermal (Infrared) Encoder
        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])

        # Dropout regularization (Spatial Dropout2d zeroes out full channels to prevent overfitting)
        if dropout_rate > 0.0:
            enc_p = dropout_rate * 0.7
            self.drop_enc_rgb = nn.Dropout2d(p=enc_p)
            self.drop_enc_inf = nn.Dropout2d(p=enc_p)
            self.drop_fuse    = nn.Dropout2d(p=dropout_rate)
            self.drop_dec4    = nn.Dropout2d(p=dropout_rate)
            self.drop_dec3    = nn.Dropout2d(p=dropout_rate)
            self.drop_dec2    = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop_enc_rgb = nn.Identity()
            self.drop_enc_inf = nn.Identity()
            self.drop_fuse    = nn.Identity()
            self.drop_dec4    = nn.Identity()
            self.drop_dec3    = nn.Identity()
            self.drop_dec2    = nn.Identity()

        # Fused Decoder
        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3] + inf_ch[3], rgb_ch[2] + inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2] + inf_ch[2], rgb_ch[1] + inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1] + inf_ch[1], rgb_ch[0] + inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0] + inf_ch[0], n_class)

    def forward(self, x):
        # Split data into RGB and Thermal (INF)
        x_rgb = x[:, :3]
        x_inf = x[:, 3:]

        # Encode RGB
        x_rgb    = self.conv1_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2)   # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)
        x_rgb    = self.drop_enc_rgb(x_rgb)

        # Encode Thermal (INF)
        x_inf    = self.conv1_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2)   # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)
        x_inf    = self.drop_enc_inf(x_inf)

        # Fusion: concatenate RGB and Thermal feature maps
        x = torch.cat((x_rgb, x_inf), dim=1)
        x = self.drop_fuse(x)

        # Decode with skip connections (using F.interpolate for PyTorch compatibility)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = self.drop_dec4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # unpool3
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = self.drop_dec3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = self.drop_dec2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # unpool1
        x = self.decode1(x)

        return x


def build_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    **kwargs,
) -> nn.Module:
    """
    Build and return the MFNet segmentation model with optional dropout regularization.
    """
    return MFNet(n_class=num_classes, dropout_rate=dropout_rate)


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2, 4, 480, 640).astype(np.float32))
    model = MFNet(n_class=NUM_CLASSES)
    y = model(x)
    print('MFNet output shape:', y.shape)
    assert y.shape == (2, NUM_CLASSES, 480, 640), f'output shape (2,{NUM_CLASSES},480,640) is expected!'
    print('MFNet unit test passed!')
    
    print("model")
    print(model)


if __name__ == '__main__':
    unit_test()
