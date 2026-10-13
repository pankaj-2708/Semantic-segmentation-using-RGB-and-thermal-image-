"""Custom dual-stream U-Net with multi-scale cross-modal attention for MFNet.

Input : float tensor [B, 4, H, W] in [0, 1], where channels 0:3 are RGB
        and channel 3 is thermal.
Output: segmentation logits [B, num_classes, H, W].

This is a compact, train-from-scratch architecture intended to fit into the
existing MFNet training pipeline. It does not load a friend's MFNet weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 9
IN_CHANNELS = 4
BASE_CHANNELS = 16


def _groups(channels: int) -> int:
    """Choose a GroupNorm group count that divides the channel count."""
    for n in (8, 4, 2):
        if channels % n == 0:
            return n
    return 1


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModalityEncoder(nn.Module):
    """Encoder for either RGB or thermal input; both streams have the same layout."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        self.stage1 = ConvBlock(in_channels, c1)
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c1, c2))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c2, c3))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c3, c4))
        self.stage5 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c4, c5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5


class CrossModalAttentionFusion(nn.Module):
    """Fuse aligned RGB/thermal features using channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )
        # Each spatial gate sees both modalities, allowing one modality to guide
        # which locations are emphasized in the other.
        self.rgb_spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.thermal_spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((rgb, thermal), dim=1)
        channel_weights = self.channel_attention(joined)
        rgb_channel, thermal_channel = channel_weights.chunk(2, dim=1)

        rgb_mean = rgb.mean(dim=1, keepdim=True)
        thermal_mean = thermal.mean(dim=1, keepdim=True)
        rgb_spatial = torch.sigmoid(
            self.rgb_spatial_attention(torch.cat((rgb_mean, thermal_mean), dim=1))
        )
        thermal_spatial = torch.sigmoid(
            self.thermal_spatial_attention(torch.cat((thermal_mean, rgb_mean), dim=1))
        )

        rgb_attended = rgb * rgb_channel * rgb_spatial
        thermal_attended = thermal * thermal_channel * thermal_spatial
        fused = self.fuse(torch.cat((rgb_attended, thermal_attended), dim=1))

        # Residual keeps useful information available if an attention gate starts
        # by suppressing a modality too strongly early in training.
        return fused + 0.5 * (rgb + thermal)


class DecoderBlock(nn.Module):
    """Upsample, concatenate an encoder skip, then refine with convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        return self.refine(torch.cat((x, skip), dim=1))


class CustomDualStreamUNet(nn.Module):
    """Dual-stream U-Net with attention fusion at five encoder scales."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        base_channels: int = BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("MFNet input must have 4 channels: RGB + thermal.")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8 for this model.")

        channels = [base_channels, base_channels * 2, base_channels * 4,
                    base_channels * 8, base_channels * 12]
        self.rgb_encoder = ModalityEncoder(in_channels=3, base_channels=base_channels)
        self.thermal_encoder = ModalityEncoder(in_channels=1, base_channels=base_channels)
        self.fusion = nn.ModuleList(
            [CrossModalAttentionFusion(c) for c in channels]
        )

        self.decoder4 = DecoderBlock(channels[4], channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])
        self.segmentation_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(
                f"Expected input shape [B, 4, H, W] (RGB + thermal), got {tuple(x.shape)}"
            )

        rgb_features = self.rgb_encoder(x[:, :3, :, :])
        thermal_features = self.thermal_encoder(x[:, 3:4, :, :])
        fused = [
            fusion(rgb_feature, thermal_feature)
            for fusion, rgb_feature, thermal_feature
            in zip(self.fusion, rgb_features, thermal_features)
        ]

        y = self.decoder4(fused[4], fused[3])
        y = self.decoder3(y, fused[2])
        y = self.decoder2(y, fused[1])
        y = self.decoder1(y, fused[0])
        logits = self.segmentation_head(y)

        # Defensively preserve input resolution for odd or nonstandard image sizes.
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def build_model(
    num_classes: int = NUM_CLASSES,
    in_channels: int = IN_CHANNELS,
    base_channels: int = BASE_CHANNELS,
) -> CustomDualStreamUNet:
    """Factory used by train.py, keeping the trainer/model interface simple."""
    return CustomDualStreamUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
    )
