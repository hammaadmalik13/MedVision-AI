"""Model registry and factory."""

from typing import Any

import torch.nn as nn

from training.models.attention_unet import AttentionUNet3D
from training.models.medsam_wrapper import MedSAMWrapper
from training.models.nnunet_wrapper import NNUNetWrapper
from training.models.segformer import SegFormer3D
from training.models.swin_unet import SwinUNet3D
from training.models.unet import UNet3D
from training.models.unetplusplus import UNetPlusPlus3D


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "unet": UNet3D,
    "attention_unet": AttentionUNet3D,
    "unetplusplus": UNetPlusPlus3D,
    "swin_unet": SwinUNet3D,
    "segformer": SegFormer3D,
    "nnunet": NNUNetWrapper,
    "medsam": MedSAMWrapper,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
