"""Second-stage document super-resolution / refinement network.

The NexaDoc ``upsample`` task is 256x256 -> 768x1024, i.e. an anisotropic 3x4
upscale.  Bicubic is the baseline to beat (MAE 0.0377, PSNR 20.0, SSIM 0.835
on index 40000).  This is a residual U-Net over the bicubic image, not a
Laplacian mask refiner.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(8, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.reduce(torch.cat([x, skip], dim=1))


class SRUNet(nn.Module):
    """Residual U-Net for 256x256 -> 768x1024 refinement."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        clamp_output: bool = True,
    ):
        super().__init__()
        c = base_channels
        self.clamp_output = bool(clamp_output)

        self.stem = ConvBlock(in_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.down4 = DownBlock(c * 8, c * 8)

        self.up1 = UpBlock(c * 8, c * 8, c * 4)
        self.up2 = UpBlock(c * 4, c * 4, c * 2)
        self.up3 = UpBlock(c * 2, c * 2, c)
        self.up4 = UpBlock(c, c, c)

        self.head = nn.Conv2d(c, out_channels, 3, 1, 1)
        # Start as an identity over the bicubic input.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        image: torch.Tensor,
        out_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Args:
            image: (B, 3, 256, 256) canonical shadow-free page.
            out_size: (H, W), defaults to (4*H, 3*W) = 1024x768.
        """
        if out_size is None:
            out_size = (image.shape[-2] * 4, image.shape[-1] * 3)

        base = F.interpolate(
            image, size=out_size, mode="bicubic", align_corners=False
        )

        s1 = self.stem(base)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)

        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        output = base + self.head(x)
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        return output


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
