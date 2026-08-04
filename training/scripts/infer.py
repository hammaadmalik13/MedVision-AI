"""Inference script."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from training.evaluation.metrics import compute_all_metrics
from training.models.registry import build_model
from training.pipelines.preprocessing import PreprocessingPipeline


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def run_inference(
    model_name: str,
    checkpoint_path: Path,
    input_paths: dict[str, Path],
    output_path: Path,
    device: str = "cuda",
) -> dict:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, in_channels=4, out_channels=4)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    pipeline = PreprocessingPipeline({"preprocessing": {"target_size": [128, 128, 128]}})
    cache = pipeline.preprocess_subject(input_paths, output_path.parent)

    data = torch.load(cache, weights_only=False)
    image = data["image"].unsqueeze(0).to(device_obj)

    start = time.perf_counter()
    with torch.no_grad():
        if device_obj.type == "cuda":
            with torch.cuda.amp.autocast():
                output = model(image)
        else:
            output = model(image)
        if isinstance(output, list):
            output = output[0]
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    inference_time_ms = (time.perf_counter() - start) * 1000

    ref_img = nib.load(str(input_paths["flair"]))
    pred_nii = nib.Nifti1Image(pred.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(pred_nii, str(output_path))

    metrics = {}
    if "seg" in input_paths:
        from training.pipelines.dataset import remap_labels

        gt = remap_labels(nib.load(str(input_paths["seg"])).get_fdata())
        if gt.shape != pred.shape:
            from scipy.ndimage import zoom

            factors = [p / g for p, g in zip(pred.shape, gt.shape, strict=True)]
            gt = zoom(gt, factors, order=0)
        metrics = compute_all_metrics(pred, gt.astype(np.int64))

    return {"inference_time_ms": inference_time_ms, "output_path": str(output_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="MedVision AI Inference")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = {}
    for modality in ["t1", "t1ce", "t2", "flair"]:
        matches = list(input_dir.glob(f"*{modality}*"))
        if matches:
            paths[modality] = matches[0]
    seg_matches = list(input_dir.glob("*seg*"))
    if seg_matches:
        paths["seg"] = seg_matches[0]

    result = run_inference(
        args.model,
        Path(args.checkpoint),
        paths,
        Path(args.output),
        args.device,
    )
    print(f"Inference complete in {result['inference_time_ms']:.1f}ms")
    if result["metrics"]:
        print(f"Mean Dice: {result['metrics'].get('mean_dice', 0):.4f}")


if __name__ == "__main__":
    main()
