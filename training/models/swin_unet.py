"""Swin U-Net 3D - simplified transformer-based segmentation."""

import torch
import torch.nn as nn
from einops import rearrange


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwinUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.stem = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.encoder_blocks = nn.ModuleList()
        dim = embed_dim
        for depth, heads in zip(depths, num_heads, strict=True):
            blocks = nn.Sequential(*[SwinBlock3D(dim, heads, dropout=dropout) for _ in range(depth)])
            self.encoder_blocks.append(blocks)
            dim *= 2

        self.bottleneck_dim = embed_dim * (2 ** (len(depths) - 1))
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.bottleneck_dim, embed_dim * 4, 2, stride=2),
            nn.InstanceNorm3d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim * 4, embed_dim * 2, 2, stride=2),
            nn.InstanceNorm3d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim * 2, embed_dim, 2, stride=2),
            nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x = self.stem(x)
        _, ec, ed, eh, ew = x.shape
        x_flat = rearrange(x, "b c d h w -> b (d h w) c")

        for blocks in self.encoder_blocks:
            x_flat = blocks(x_flat)
            x_flat = nn.functional.pad(x_flat, (0, 0, 0, x_flat.shape[1]))

        x = rearrange(x_flat[:, : ed * eh * ew], "b (d h w) c -> b c d h w", d=ed, h=eh, w=ew)
        x = self.decoder(x)
        if x.shape[2:] != (d, h, w):
            x = nn.functional.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)
        return self.head(x)
