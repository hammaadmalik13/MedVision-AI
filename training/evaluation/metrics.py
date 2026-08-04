"""Evaluation metrics for segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve


def dice_score(pred: np.ndarray, target: np.ndarray, num_classes: int = 4, include_background: bool = False) -> dict[int, float]:
    scores = {}
    start = 0 if include_background else 1
    for c in range(start, num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        scores[c] = float((2 * intersection + 1e-6) / (union + 1e-6))
    return scores


def iou_score(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> dict[int, float]:
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).astype(bool)
        t = (target == c).astype(bool)
        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        scores[c] = float((intersection + 1e-6) / (union + 1e-6))
    return scores


def precision_recall(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> dict[str, dict[int, float]]:
    precision, recall = {}, {}
    for c in range(1, num_classes):
        p = (pred == c).astype(bool)
        t = (target == c).astype(bool)
        tp = np.logical_and(p, t).sum()
        precision[c] = float(tp / (p.sum() + 1e-6))
        recall[c] = float(tp / (t.sum() + 1e-6))
    return {"precision": precision, "recall": recall}


def sensitivity_specificity(pred: np.ndarray, target: np.ndarray, tumor_class: int = 3) -> dict[str, float]:
    p = (pred == tumor_class).astype(bool)
    t = (target == tumor_class).astype(bool)
    tp = np.logical_and(p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    return {
        "sensitivity": float(tp / (tp + fn + 1e-6)),
        "specificity": float(tn / (tn + fp + 1e-6)),
    }


def hausdorff_distance(pred: np.ndarray, target: np.ndarray, tumor_class: int = 3, spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    try:
        from medpy.metric.binary import hd95

        p = (pred == tumor_class).astype(np.uint8)
        t = (target == tumor_class).astype(np.uint8)
        if p.sum() == 0 or t.sum() == 0:
            return float("inf")
        return float(hd95(p, t, voxelspacing=spacing))
    except Exception:
        return float("nan")


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    dice = dice_score(pred, target, num_classes)
    iou = iou_score(pred, target, num_classes)
    pr = precision_recall(pred, target, num_classes)
    ss = sensitivity_specificity(pred, target)
    hd = hausdorff_distance(pred, target, spacing=spacing)

    tumor_mask = pred > 0
    total_voxels = pred.size
    tumor_voxels = tumor_mask.sum()
    voxel_volume_mm3 = np.prod(spacing)

    return {
        "dice": dice,
        "mean_dice": float(np.mean(list(dice.values()))),
        "iou": iou,
        "mean_iou": float(np.mean(list(iou.values()))),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "sensitivity": ss["sensitivity"],
        "specificity": ss["specificity"],
        "hausdorff_distance": hd,
        "tumor_volume_cm3": float(tumor_voxels * voxel_volume_mm3 / 1000),
        "tumor_percentage": float(tumor_voxels / total_voxels * 100),
    }


def save_confusion_matrix(pred: np.ndarray, target: np.ndarray, output_path: Path, num_classes: int = 4) -> None:
    cm = confusion_matrix(target.flatten(), pred.flatten(), labels=list(range(num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(pred_probs: np.ndarray, target: np.ndarray, output_path: Path, tumor_class: int = 3) -> float:
    y_true = (target == tumor_class).astype(int).flatten()
    y_score = pred_probs[tumor_class].flatten()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return roc_auc


def evaluate_model(model, dataloader, device: torch.device, output_dir: Path | None = None) -> dict[str, Any]:
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            for pred, target in zip(preds, targets, strict=True):
                all_metrics.append(compute_all_metrics(pred, target))

    aggregated = {
        "mean_dice": float(np.mean([m["mean_dice"] for m in all_metrics])),
        "mean_iou": float(np.mean([m["mean_iou"] for m in all_metrics])),
        "mean_sensitivity": float(np.mean([m["sensitivity"] for m in all_metrics])),
        "mean_specificity": float(np.mean([m["specificity"] for m in all_metrics])),
        "mean_hausdorff": float(np.nanmean([m["hausdorff_distance"] for m in all_metrics])),
        "num_samples": len(all_metrics),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(aggregated, indent=2))

    return aggregated
