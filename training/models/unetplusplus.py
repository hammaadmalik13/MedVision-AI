"""UNet++ 3D with nested skip connections."""

import torch
import torch.nn as nn

from training.models.unet import ConvBlock3D


class UNetPlusPlus3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: list[int] | None = None,
        dropout: float = 0.1,
        deep_supervision: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        features = features or [32, 64, 128, 256]
        self.deep_supervision = deep_supervision
        self.features = features

        self.pool = nn.MaxPool3d(2, 2)
        self.conv0_0 = ConvBlock3D(in_channels, features[0], dropout=dropout)
        self.conv1_0 = ConvBlock3D(features[0], features[1], dropout=dropout)
        self.conv2_0 = ConvBlock3D(features[1], features[2], dropout=dropout)
        self.conv3_0 = ConvBlock3D(features[2], features[3], dropout=dropout)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv0_1 = ConvBlock3D(features[0] + features[1], features[0], dropout=dropout)
        self.conv1_1 = ConvBlock3D(features[1] + features[2], features[1], dropout=dropout)
        self.conv2_1 = ConvBlock3D(features[2] + features[3], features[2], dropout=dropout)

        self.conv0_2 = ConvBlock3D(features[0] * 2 + features[1], features[0], dropout=dropout)
        self.conv1_2 = ConvBlock3D(features[1] * 2 + features[2], features[1], dropout=dropout)

        self.conv0_3 = ConvBlock3D(features[0] * 3 + features[1], features[0], dropout=dropout)

        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)
        if deep_supervision:
            self.ds1 = nn.Conv3d(features[0], out_channels, 1)
            self.ds2 = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        out = self.final(x0_3)

        if self.deep_supervision and self.training:
            return [out, self.ds1(x0_2), self.ds2(x0_1)]
        return out
