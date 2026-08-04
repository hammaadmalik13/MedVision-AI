"""ML inference engine."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from medvision.config import get_settings
from training.models.registry import build_model


class InferenceEngine:
    def __init__(
        self,
        model_name: str | None = None,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.default_model
        self.device = torch.device(
            device or settings.inference_device if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = settings.use_mixed_precision and self.device.type == "cuda"
        self.model: nn.Module | None = None
        self.checkpoint_path = checkpoint_path

    def load_model(self, checkpoint_path: Path | None = None) -> None:
        path = checkpoint_path or self.checkpoint_path
        self.model = build_model(self.model_name, in_channels=4, out_channels=4)
        if path and path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        if self.use_amp and hasattr(torch, "compile") and get_settings().use_torch_compile:
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def predict(self, image: np.ndarray | torch.Tensor) -> tuple[np.ndarray, float]:
        if self.model is None:
            self.load_model()

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = image.float()

        tensor = tensor.to(self.device)
        start = time.perf_counter()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(tensor)
        else:
            output = self.model(tensor)

        if isinstance(output, list):
            output = output[0]

        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        inference_time_ms = (time.perf_counter() - start) * 1000
        return pred, inference_time_ms

    def export_onnx(self, output_path: Path, input_shape: tuple = (1, 4, 128, 128, 128)) -> None:
        if self.model is None:
            self.load_model()
        dummy_input = torch.randn(*input_shape).to(self.device)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )


class ONNXInferenceEngine:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        if image.ndim == 4:
            image = image[np.newaxis, ...]
        output = self.session.run(None, {self.input_name: image.astype(np.float32)})[0]
        pred = output.argmax(axis=1).squeeze(0)
        inference_time_ms = (time.perf_counter() - start) * 1000
        return pred, inference_time_ms
