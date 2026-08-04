"""Preprocessing pipeline using MONAI and OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)

from training.pipelines.dataset import BraTSOrganizer, remap_labels

logger = logging.getLogger(__name__)


def zscore_normalize(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        region = data[mask > 0]
        if region.size == 0:
            return data
        mean, std = region.mean(), region.std()
    else:
        mean, std = data.mean(), data.std()
    if std < 1e-8:
        return data - mean
    return (data - mean) / std


def histogram_normalize(data: np.ndarray, bins: int = 256) -> np.ndarray:
    flat = data.flatten()
    hist, bin_edges = np.histogram(flat, bins=bins)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    return np.interp(flat, bin_edges[:-1], cdf).reshape(data.shape)


def denoise_volume(data: np.ndarray, method: str = "gaussian") -> np.ndarray:
    import cv2

    result = np.zeros_like(data)
    for i in range(data.shape[2]):
        slice_2d = data[:, :, i].astype(np.float32)
        if method == "gaussian":
            result[:, :, i] = cv2.GaussianBlur(slice_2d, (3, 3), 0)
        elif method == "bilateral":
            normalized = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(
                np.uint8
            )
            result[:, :, i] = cv2.bilateralFilter(normalized, 9, 75, 75).astype(np.float32)
        else:
            result[:, :, i] = slice_2d
    return result


class PreprocessingPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.preproc_config = config.get("preprocessing", {})
        self.aug_config = config.get("augmentation", {})

    def get_load_transforms(self) -> Compose:
        keys = ["t1", "t1ce", "t2", "flair", "seg"]
        return Compose([
            LoadImaged(keys=keys, ensure_channel_first=True),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
        ])

    def get_preprocess_transforms(self) -> Compose:
        keys = ["t1", "t1ce", "t2", "flair"]
        transforms = [
            Spacingd(keys=keys + ["seg"], pixdim=self.preproc_config.get("target_spacing", [1.0, 1.0, 1.0])),
        ]

        norm_method = self.preproc_config.get("intensity_normalization", "zscore")
        if norm_method == "zscore":
            transforms.append(NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True))
        elif norm_method == "minmax":
            transforms.append(ScaleIntensityRanged(keys=keys, a_min=0, a_max=255, b_min=0, b_max=1, clip=True))

        target_size = self.preproc_config.get("target_size")
        if target_size:
            transforms.append(Resized(keys=keys + ["seg"], spatial_size=target_size, mode=("trilinear",) * 4 + ("nearest",)))

        if self.preproc_config.get("crop_foreground", True):
            transforms.append(CropForegroundd(keys=keys + ["seg"], source_key="flair"))

        return Compose(transforms)

    def get_augmentation_transforms(self) -> Compose | None:
        if not self.aug_config.get("enabled", True):
            return None

        keys = ["image", "label"]
        transforms = []
        if self.aug_config.get("random_rotation"):
            transforms.append(RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 2)))
        if self.aug_config.get("horizontal_flip"):
            transforms.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=0))
        if self.aug_config.get("vertical_flip"):
            transforms.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=1))
        if self.aug_config.get("elastic_transform"):
            transforms.append(RandAffined(keys=keys, prob=0.3, rotate_range=0.1, scale_range=0.1))
        if self.aug_config.get("gaussian_noise"):
            transforms.append(RandGaussianNoised(keys=["image"], prob=0.2, std=self.aug_config["gaussian_noise"]))
        if self.aug_config.get("brightness"):
            transforms.append(RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3))
            transforms.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3))

        return Compose(transforms) if transforms else None

    def preprocess_subject(self, subject_paths: dict[str, Path], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_image = []
        for modality in ["t1", "t1ce", "t2", "flair"]:
            img = nib.load(str(subject_paths[modality]))
            data = img.get_fdata().astype(np.float32)
            if self.preproc_config.get("denoising"):
                data = denoise_volume(data)
            data = zscore_normalize(data)
            combined_image.append(data)

        image = np.stack(combined_image, axis=0)
        seg_path = subject_paths.get("seg")
        if seg_path:
            seg = remap_labels(nib.load(str(seg_path)).get_fdata())
        else:
            seg = np.zeros(image.shape[1:], dtype=np.int64)

        target_size = self.preproc_config.get("target_size")
        if target_size:
            from scipy.ndimage import zoom

            factors = [1.0] + [t / s for t, s in zip(target_size, image.shape[1:], strict=True)]
            image = zoom(image, factors, order=1)
            seg = zoom(seg, factors[1:], order=0)

        subject_id = subject_paths["t1"].stem.split("_")[0]
        cache_path = output_dir / f"{subject_id}_preprocessed.pt"
        torch.save({"image": torch.from_numpy(image), "label": torch.from_numpy(seg)}, cache_path)
        return cache_path

    def process_dataset(self, root_path: Path, cache_path: Path) -> list[Path]:
        organizer = BraTSOrganizer(root_path)
        subjects = organizer.discover_subjects()
        cache_path.mkdir(parents=True, exist_ok=True)
        cached = []
        for subject in subjects:
            paths = {**subject.modalities, "seg": subject.segmentation}
            try:
                cached.append(self.preprocess_subject(paths, cache_path))
            except Exception as e:
                logger.error("Failed to preprocess %s: %s", subject.subject_id, e)
        return cached
