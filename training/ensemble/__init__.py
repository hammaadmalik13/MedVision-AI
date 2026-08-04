"""Ensemble prediction strategies."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class EnsembleStrategy(str, Enum):
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"


class EnsemblePredictor:
    def __init__(
        self,
        models: list[nn.Module],
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        weights: list[float] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.models = models
        self.strategy = strategy
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models:
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        image = image.to(self.device)
        outputs = []
        for model in self.models:
            out = model(image)
            if isinstance(out, list):
                out = out[0]
            outputs.append(torch.softmax(out, dim=1))

        if self.strategy == EnsembleStrategy.AVERAGE:
            combined = torch.stack(outputs).mean(dim=0)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            weighted = sum(w * o for w, o in zip(self.weights, outputs, strict=True))
            combined = weighted / sum(self.weights)
        elif self.strategy == EnsembleStrategy.VOTING:
            votes = torch.stack([o.argmax(dim=1) for o in outputs])
            combined = torch.mode(votes, dim=0).values.float()
            return combined.squeeze(0).cpu().numpy().astype(np.int64)
        else:
            combined = torch.stack(outputs).mean(dim=0)

        return combined.argmax(dim=1).squeeze(0).cpu().numpy()
