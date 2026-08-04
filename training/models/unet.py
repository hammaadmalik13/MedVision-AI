"""3D U-Net for brain tumor segmentation."""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "instance",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        act_layer = nn.ReLU(inplace=True) if activation == "relu" else nn.LeakyReLU(0.1, inplace=True)

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer,
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: list[int] | None = None,
        dropout: float = 0.1,
        norm: str = "instance",
        activation: str = "relu",
        deep_supervision: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        features = features or [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        in_ch = in_channels
        for feat in features:
            self.encoder.append(ConvBlock3D(in_ch, feat, norm, activation, dropout))
            in_ch = feat

        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2, norm, activation, dropout)

        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        rev_features = features[::-1]
        in_ch = features[-1] * 2
        for feat in rev_features:
            self.upconvs.append(nn.ConvTranspose3d(in_ch, feat, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock3D(feat * 2, feat, norm, activation, dropout))
            in_ch = feat

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder, strict=True)):
            x = upconv(x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final_conv(x)
