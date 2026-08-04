"""BraTS dataset pipeline: download, verify, preprocess, augment."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODALITY_SUFFIXES = {
    "t1": ["_t1.nii", "_t1.nii.gz", "-t1n.nii.gz"],
    "t1ce": ["_t1ce.nii", "_t1ce.nii.gz", "-t1c.nii.gz"],
    "t2": ["_t2.nii", "_t2.nii.gz", "-t2w.nii.gz"],
    "flair": ["_flair.nii", "_flair.nii.gz", "-t2f.nii.gz"],
}
SEG_SUFFIXES = ["_seg.nii", "_seg.nii.gz", "-seg.nii.gz"]
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}


@dataclass
class BraTSSubject:
    subject_id: str
    root_path: Path
    modalities: dict[str, Path] = field(default_factory=dict)
    segmentation: Path | None = None

    def is_complete(self) -> bool:
        return len(self.modalities) == 4 and self.segmentation is not None


class BraTSOrganizer:
    """Organize and verify BraTS dataset structure."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def discover_subjects(self) -> list[BraTSSubject]:
        subjects: dict[str, BraTSSubject] = {}
        for nii_file in sorted(self.root_path.rglob("*.nii*")):
            name = nii_file.name.lower()
            subject_id = self._extract_subject_id(nii_file)
            if subject_id not in subjects:
                subjects[subject_id] = BraTSSubject(subject_id=subject_id, root_path=nii_file.parent)

            for modality, suffixes in MODALITY_SUFFIXES.items():
                if any(s in name for s in suffixes):
                    subjects[subject_id].modalities[modality] = nii_file
                    break
            if any(s in name for s in SEG_SUFFIXES):
                subjects[subject_id].segmentation = nii_file

        return [s for s in subjects.values() if s.is_complete()]

    def _extract_subject_id(self, path: Path) -> str:
        name = path.stem.replace(".nii", "")
        for suffixes in list(MODALITY_SUFFIXES.values()) + [SEG_SUFFIXES]:
            for s in suffixes:
                if name.lower().endswith(s.replace(".nii.gz", "").replace(".nii", "")):
                    return name[: -len(s.replace(".nii.gz", "").replace(".nii", ""))].rstrip("-_")
        return name.split("_")[0]

    def verify_subject(self, subject: BraTSSubject) -> dict[str, Any]:
        report: dict[str, Any] = {"subject_id": subject.subject_id, "valid": True, "errors": []}
        shapes = []
        affines = []

        for modality, path in subject.modalities.items():
            img = nib.load(str(path))
            shapes.append(img.shape)
            affines.append(img.affine.tolist())
            data = img.get_fdata()
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                report["errors"].append(f"{modality}: contains NaN/Inf")
                report["valid"] = False

        if len(set(shapes)) > 1:
            report["errors"].append(f"Shape mismatch: {shapes}")
            report["valid"] = False

        if subject.segmentation:
            seg = nib.load(str(subject.segmentation)).get_fdata().astype(np.int32)
            unique_labels = set(np.unique(seg).astype(int).tolist())
            invalid = unique_labels - {0, 1, 2, 4}
            if invalid:
                report["errors"].append(f"Invalid labels: {invalid}")
                report["valid"] = False

        report["shape"] = shapes[0] if shapes else None
        return report

    def verify_all(self) -> list[dict[str, Any]]:
        return [self.verify_subject(s) for s in self.discover_subjects()]

    def create_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[str]]:
        subjects = self.discover_subjects()
        ids = [s.subject_id for s in subjects]
        train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
        val_size = val_ratio / (1 - train_ratio)
        val_ids, test_ids = train_test_split(temp_ids, train_size=val_size, random_state=seed)
        split = {"train": train_ids, "val": val_ids, "test": test_ids}
        split_path = self.root_path / "split.json"
        split_path.write_text(json.dumps(split, indent=2))
        logger.info("Split saved: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))
        return split

    def get_manifest(self) -> dict[str, Any]:
        subjects = self.discover_subjects()
        return {
            "total_subjects": len(subjects),
            "root_path": str(self.root_path),
            "subjects": [
                {
                    "id": s.subject_id,
                    "modalities": {k: str(v) for k, v in s.modalities.items()},
                    "segmentation": str(s.segmentation) if s.segmentation else None,
                }
                for s in subjects
            ],
        }


def remap_labels(segmentation: np.ndarray) -> np.ndarray:
    result = np.zeros_like(segmentation, dtype=np.int64)
    for old, new in LABEL_REMAP.items():
        result[segmentation == old] = new
    return result


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
