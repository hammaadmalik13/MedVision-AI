"""Explainable AI: GradCAM, EigenCAM, attention maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, list):
            output = output[0]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self) -> None:
        for hook in self._hooks:
            hook.remove()


class EigenCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("Activations not captured")

        act = self.activations[0].cpu().numpy()
        act_flat = act.reshape(act.shape[0], -1)
        cov = np.cov(act_flat)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        cam = eigenvectors[:, -1].reshape(act.shape[1:])
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self) -> None:
        self._hook.remove()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    output_path: Path | None = None,
) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] <= 4:
        display = image[image.shape[0] // 2] if image.shape[0] > 1 else image[0]
    else:
        display = image

    if heatmap.shape != display.shape:
        from scipy.ndimage import zoom

        factors = [d / h for d, h in zip(display.shape, heatmap.shape, strict=True)]
        heatmap = zoom(heatmap, factors, order=1)

    display = (display - display.min()) / (display.max() - display.min() + 1e-8)
    cmap = plt.cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * np.stack([display] * 3, axis=-1) + alpha * cmap

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(output_path), overlay)

    return overlay
