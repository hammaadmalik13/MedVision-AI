"""nnUNet wrapper for integration with MedVision AI."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class NNUNetWrapper(nn.Module):
    """Wrapper around nnUNetv2 trained checkpoints."""

    def __init__(
        self,
        configuration: str = "3d_fullres",
        fold: int = 0,
        plans_identifier: str = "nnUNetPlans",
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.configuration = configuration
        self.fold = fold
        self.plans_identifier = plans_identifier
        self._network: nn.Module | None = None
        self._checkpoint_path = checkpoint_path

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(Path(checkpoint_path))

    def _build_fallback(self, in_channels: int = 4, out_channels: int = 4) -> nn.Module:
        from training.models.unet import UNet3D

        return UNet3D(in_channels=in_channels, out_channels=out_channels)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            if self._network is None:
                self._network = self._build_fallback()
            self._network.load_state_dict(checkpoint["network_weights"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if self._network is None:
                self._network = self._build_fallback()
            self._network.load_state_dict(checkpoint["state_dict"])
        else:
            if self._network is None:
                self._network = self._build_fallback()
            self._network.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._network is None:
            self._network = self._build_fallback(x.shape[1], 4)
            self._network = self._network.to(x.device)
        return self._network(x)

    def predict(self, volume: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(volume).float().unsqueeze(0)
            if next(self.parameters()).is_cuda:
                tensor = tensor.cuda()
            output = self.forward(tensor)
            return output.argmax(dim=1).squeeze(0).cpu().numpy()

    def get_nnunet_predictor(self) -> Any:
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            return predictor
        except ImportError:
            return None
