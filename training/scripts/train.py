"""Training script with MLflow integration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.evaluation.metrics import compute_all_metrics, evaluate_model
from training.losses import get_loss
from training.models.registry import build_model
from training.optimizers import get_optimizer, get_scheduler
from training.pipelines.dataloader import BraTSDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                if isinstance(outputs, list):
                    loss = sum(criterion(o, labels) for o in outputs) / len(outputs)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, list):
                loss = sum(criterion(o, labels) for o in outputs) / len(outputs)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            for pred, target in zip(preds, targets, strict=True):
                all_metrics.append(compute_all_metrics(pred, target))

    avg_metrics = {
        "mean_dice": float(np.mean([m["mean_dice"] for m in all_metrics])) if all_metrics else 0.0,
        "mean_iou": float(np.mean([m["mean_iou"] for m in all_metrics])) if all_metrics else 0.0,
    }
    return total_loss / max(len(loader), 1), avg_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MedVision AI Training")
    parser.add_argument("--config", type=str, default="configs/training/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    if args.model:
        training_cfg["model"] = args.model
    if args.epochs:
        training_cfg["epochs"] = args.epochs
    if args.batch_size:
        training_cfg["batch_size"] = args.batch_size

    set_seed(training_cfg.get("seed", 42))
    device = torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    use_amp = training_cfg.get("mixed_precision", True) and device.type == "cuda"

    cache_path = Path(config.get("dataset", {}).get("cache_path", "./data/cache"))
    split_path = Path(config.get("dataset", {}).get("root_path", "./datasets/brats")) / "split.json"
    train_ids, val_ids = None, None
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_ids, val_ids = split.get("train"), split.get("val")

    train_dataset = BraTSDataset(cache_path, train_ids)
    val_dataset = BraTSDataset(cache_path, val_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(
        training_cfg["model"],
        in_channels=4,
        out_channels=4,
    ).to(device)

    if training_cfg.get("torch_compile") and hasattr(torch, "compile"):
        model = torch.compile(model)

    loss_cfg = config.get("loss", {"name": "dice_ce"})
    criterion = get_loss(loss_cfg["name"], **loss_cfg.get("params", {}))

    opt_cfg = config.get("optimizer", {"name": "adamw"})
    optimizer = get_optimizer(opt_cfg["name"], model.parameters(), **opt_cfg.get("params", {}))

    sched_cfg = config.get("scheduler", {"name": "cosine"})
    scheduler = get_scheduler(sched_cfg["name"], optimizer, **sched_cfg.get("params", {}))

    scaler = GradScaler() if use_amp else None
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "./data/models/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "medvision-brain-tumor"))

    best_dice = 0.0
    patience_counter = 0
    patience = training_cfg.get("early_stopping_patience", 15)

    with mlflow.start_run():
        mlflow.log_params({
            "model": training_cfg["model"],
            "epochs": training_cfg["epochs"],
            "batch_size": training_cfg["batch_size"],
            "loss": loss_cfg["name"],
            "optimizer": opt_cfg["name"],
        })

        for epoch in range(training_cfg["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            if sched_cfg["name"] == "reduce_on_plateau":
                scheduler.step(val_loss)
            elif sched_cfg["name"] != "onecycle":
                scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_metrics["mean_dice"],
                "val_iou": val_metrics["mean_iou"],
            }, step=epoch)

            print(f"Epoch {epoch + 1}/{training_cfg['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Dice: {val_metrics['mean_dice']:.4f}")

            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "config": config,
                }, checkpoint_dir / f"{training_cfg['model']}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        metrics_dir = checkpoint_dir / "evaluation"
        evaluate_model(model, val_loader, device, metrics_dir)
        if mlflow_cfg.get("log_model"):
            mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
