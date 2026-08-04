"""Segmentation model interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class SegmentationModel(ABC):
    """Protocol for all segmentation models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def predict(self, volume: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None: ...

    @abstractmethod
    def save_checkpoint(self, path: Path, metadata: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def export_onnx(self, path: Path, input_shape: tuple[int, ...]) -> None: ...

    @property
    @abstractmethod
    def model(self) -> nn.Module: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
