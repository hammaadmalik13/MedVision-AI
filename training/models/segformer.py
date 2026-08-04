"""SegFormer 3D - simplified MiT-based segmentation."""

import torch
import torch.nn as nn


class MixFFN3D(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv3d(dim, hidden, 1)
        self.dwconv = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Conv3d(hidden, dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TransformerBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN3D(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + attn_out
        x = x_flat.transpose(1, 2).reshape(b, c, d, h, w)
        x = x + self.mlp(self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, d, h, w))
        return x


class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        depths: list[int] | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        embed_dims = embed_dims or [32, 64, 160, 256]
        num_heads = num_heads or [1, 2, 5, 8]
        depths = depths or [2, 2, 2, 2]

        self.stages = nn.ModuleList()
        in_ch = in_channels
        for dim, heads, depth in zip(embed_dims, num_heads, depths, strict=True):
            stage = nn.Sequential(
                nn.Conv3d(in_ch, dim, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(dim),
                nn.ReLU(inplace=True),
                *[TransformerBlock3D(dim, heads, dropout=dropout) for _ in range(depth)],
            )
            self.stages.append(stage)
            in_ch = dim

        self.fuse = nn.Conv3d(sum(embed_dims), embed_dims[0], 1)
        self.head = nn.Conv3d(embed_dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(
                nn.functional.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
            )
        x = self.fuse(torch.cat(features, dim=1))
        return self.head(x)
