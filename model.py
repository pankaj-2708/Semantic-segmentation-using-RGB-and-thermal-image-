"""
model.py - Neural Network Architectures for Semantic Segmentation
Defines UNet with checkpointed DoubleConv blocks.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# ==========================================
# Global Parameters / Configuration
# ==========================================
IN_CHANNELS = 3
NUM_CLASSES = 7
BASE_FEATURES = 32
USE_GRADIENT_CHECKPOINTING = True


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """
    5-stage UNet Architecture with optional Gradient Checkpointing for memory efficiency.
    """
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        base_features: int = BASE_FEATURES,
        use_checkpoint: bool = USE_GRADIENT_CHECKPOINTING,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        f = base_features

        # Encoder stages
        self.enc_l1 = DoubleConv(in_channels, f)        # 3 -> 32
        self.enc_l2 = DoubleConv(f, f * 2)             # 32 -> 64
        self.enc_l3 = DoubleConv(f * 2, f * 4)         # 64 -> 128
        self.enc_l4 = DoubleConv(f * 4, f * 8)         # 128 -> 256
        self.enc_l5 = DoubleConv(f * 8, f * 16)        # 256 -> 512
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bot = DoubleConv(f * 16, f * 32)          # 512 -> 1024

        # Decoder stages
        self.up_1 = nn.ConvTranspose2d(f * 32, f * 16, kernel_size=2, stride=2)
        self.dec_l1 = DoubleConv(f * 32, f * 16)       # (512 + 512) -> 512

        self.up_2 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec_l2 = DoubleConv(f * 16, f * 8)        # (256 + 256) -> 256

        self.up_3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec_l3 = DoubleConv(f * 8, f * 4)         # (128 + 128) -> 128

        self.up_4 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec_l4 = DoubleConv(f * 4, f * 2)         # (64 + 64) -> 64

        self.up_5 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec_l5 = DoubleConv(f * 2, f)             # (32 + 32) -> 32

        # Final 1x1 conv classifier
        self.out_layer = nn.Conv2d(f, num_classes, kernel_size=1)

    def _apply_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self._apply_block(self.enc_l1, x)
        enc2 = self._apply_block(self.enc_l2, self.pool(enc1))
        enc3 = self._apply_block(self.enc_l3, self.pool(enc2))
        enc4 = self._apply_block(self.enc_l4, self.pool(enc3))
        enc5 = self._apply_block(self.enc_l5, self.pool(enc4))

        # Bottleneck
        bot = self._apply_block(self.bot, self.pool(enc5))

        # Decoder with skip connections
        dec1 = self.up_1(bot)
        dec1 = torch.cat([dec1, enc5], dim=1)
        dec1 = self._apply_block(self.dec_l1, dec1)

        dec2 = self.up_2(dec1)
        dec2 = torch.cat([dec2, enc4], dim=1)
        dec2 = self._apply_block(self.dec_l2, dec2)

        dec3 = self.up_3(dec2)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self._apply_block(self.dec_l3, dec3)

        dec4 = self.up_4(dec3)
        dec4 = torch.cat([dec4, enc2], dim=1)
        dec4 = self._apply_block(self.dec_l4, dec4)

        dec5 = self.up_5(dec4)
        dec5 = torch.cat([dec5, enc1], dim=1)
        dec5 = self._apply_block(self.dec_l5, dec5)

        # Output logits: [B, num_classes, H, W]
        out = self.out_layer(dec5)
        return out


def build_model(
    in_channels: int = IN_CHANNELS,
    num_classes: int = NUM_CLASSES,
    use_checkpoint: bool = USE_GRADIENT_CHECKPOINTING,
) -> nn.Module:
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
    )
