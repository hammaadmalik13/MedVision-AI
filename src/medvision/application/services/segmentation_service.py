"""Segmentation and prediction service."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import nibabel as nib
import numpy as np

from medvision.domain.entities.model_version import ModelVersion
from medvision.domain.entities.prediction import Prediction, PredictionMetrics
from medvision.infrastructure.ml.inference import InferenceEngine
from medvision.infrastructure.ml.report_generator import compute_tumor_metrics


class SegmentationService:
    def __init__(self, inference_engine: InferenceEngine | None = None) -> None:
        self.engine = inference_engine or InferenceEngine()

    async def segment_study(
        self,
        study_id: UUID,
        modality_paths: dict[str, str],
        model_version: ModelVersion,
        user_id: UUID | None = None,
    ) -> Prediction:
        if not self.engine.model:
            self.engine.load_model(Path(model_version.checkpoint_uri))

        image = self._load_multimodal(modality_paths)
        pred_mask, inference_time_ms = self.engine.predict(image)

        metrics_dict = compute_tumor_metrics(pred_mask)
        mask_path = Path(f"./data/storage/masks/{study_id}_{uuid4()}.nii.gz")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        ref = nib.load(modality_paths.get("flair", list(modality_paths.values())[0]))
        nib.save(nib.Nifti1Image(pred_mask.astype(np.float32), ref.affine), str(mask_path))

        return Prediction(
            study_id=study_id,
            model_version_id=model_version.id,
            user_id=user_id,
            mask_uri=str(mask_path),
            metrics=PredictionMetrics(
                tumor_volume_cm3=metrics_dict["tumor_volume_cm3"],
                tumor_area_mm2=metrics_dict["tumor_area_mm2"],
                tumor_percentage=metrics_dict["tumor_percentage"],
                confidence_score=metrics_dict["confidence_score"],
            ),
            inference_time_ms=inference_time_ms,
        )

    def _load_multimodal(self, modality_paths: dict[str, str]) -> np.ndarray:
        modalities = []
        for mod in ["t1", "t1ce", "t2", "flair"]:
            path = modality_paths.get(mod)
            if path:
                data = nib.load(path).get_fdata().astype(np.float32)
                mean, std = data.mean(), data.std()
                data = (data - mean) / (std + 1e-8)
                modalities.append(data)
        if not modalities:
            raise ValueError("No modality files found")
        while len(modalities) < 4:
            modalities.append(modalities[-1])
        return np.stack(modalities[:4], axis=0)
