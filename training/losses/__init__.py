"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, include_background: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

        start = 0 if self.include_background else 1
        dice_scores = []
        for c in range(start, num_classes):
            p = pred[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum(dim=(1, 2, 3))
            union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1.0 - torch.stack(dice_scores, dim=1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        return focal.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

        tversky_scores = []
        for c in range(1, num_classes):
            p = pred[:, c]
            t = target_one_hot[:, c]
            tp = (p * t).sum(dim=(1, 2, 3))
            fp = (p * (1 - t)).sum(dim=(1, 2, 3))
            fn = ((1 - p) * t).sum(dim=(1, 2, 3))
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_scores.append(tversky)

        return 1.0 - torch.stack(tversky_scores, dim=1).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target.long())


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target.long())


class ComboLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.4,
        ce_weight: float = 0.3,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.ce_weight * self.ce(pred, target.long())
            + self.focal_weight * self.focal(pred, target)
        )


LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "dice": DiceLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "dice_bce": DiceBCELoss,
    "dice_ce": DiceCELoss,
    "tversky": TverskyLoss,
    "combo": ComboLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
