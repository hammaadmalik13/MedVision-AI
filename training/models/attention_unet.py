"""Attention U-Net 3D."""

import torch
import torch.nn as nn

from training.models.unet import ConvBlock3D, UNet3D


class AttentionGate3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_g(gate)
        x = self.W_x(skip)
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)
        psi = self.psi(self.relu(g + x))
        return skip * psi


class AttentionUNet3D(UNet3D):
    def __init__(self, *args, attention_type: str = "additive", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        features = kwargs.get("features", [32, 64, 128, 256, 512])
        self.attention_gates = nn.ModuleList()
        rev_features = features[::-1]
        gate_ch = features[-1] * 2
        for feat in rev_features:
            self.attention_gates.append(AttentionGate3D(gate_ch, feat, feat // 2))
            gate_ch = feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx, (upconv, dec, attn) in enumerate(
            zip(self.upconvs, self.decoder, self.attention_gates, strict=True)
        ):
            x = upconv(x)
            skip = skip_connections[idx]
            skip = attn(x, skip)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final_conv(x)
