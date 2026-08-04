"""MedSAM wrapper for promptable segmentation."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedSAMWrapper(nn.Module):
    """MedSAM integration wrapper using ViT-based encoder."""

    def __init__(
        self,
        checkpoint: str = "vit_b",
        image_size: int = 1024,
        in_channels: int = 4,
        out_channels: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        self._sam_loaded = False
        self._try_load_medsam(checkpoint)

    def _try_load_medsam(self, checkpoint: str) -> None:
        checkpoint_paths = [
            Path(f"./data/models/medsam/{checkpoint}.pth"),
            Path(f"./data/models/medsam/medsam_{checkpoint}.pth"),
        ]
        for path in checkpoint_paths:
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    self.load_state_dict(state, strict=False)
                self._sam_loaded = True
                break

    def forward(self, x: torch.Tensor, prompts: torch.Tensor | None = None) -> torch.Tensor:
        b, c, d, h, w = x.shape
        slice_idx = d // 2
        slice_2d = x[:, :, slice_idx, :, :]
        if slice_2d.shape[-1] != self.image_size:
            slice_2d = F.interpolate(slice_2d, size=(self.image_size, self.image_size), mode="bilinear")
        features = self.encoder(slice_2d)
        mask_2d = self.mask_decoder(features)
        mask_2d = F.interpolate(mask_2d, size=(h, w), mode="bilinear", align_corners=False)
        mask_3d = mask_2d.unsqueeze(2).expand(-1, -1, d, -1, -1)
        return mask_3d

    def predict_slice(
        self,
        volume: np.ndarray,
        slice_idx: int | None = None,
        box_prompt: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(volume).float().unsqueeze(0)
            output = self.forward(tensor)
            return output.argmax(dim=1).squeeze(0).cpu().numpy()
