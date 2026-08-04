"""Explainability router."""

from pathlib import Path
from uuid import UUID

import nibabel as nib
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLPredictionRepository
from medvision.infrastructure.ml.inference import InferenceEngine
from medvision.presentation.api.dependencies import get_current_user, get_inference_engine, get_prediction_repo
from medvision.presentation.api.schemas import ExplainRequest
from training.explainability import EigenCAM, GradCAM, overlay_heatmap
from training.models.unet import UNet3D

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("/heatmap")
async def generate_heatmap(
    request: ExplainRequest,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    engine: InferenceEngine = Depends(get_inference_engine),
    _user: User = Depends(get_current_user),
):
    prediction = await prediction_repo.get_by_id(request.prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if engine.model is None:
        engine.load_model()

    model = engine.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    target_layer = model.encoder[-1] if hasattr(model, "encoder") else model
    if request.method == "eigencam":
        cam_generator = EigenCAM(model, target_layer)
    else:
        cam_generator = GradCAM(model, target_layer)

    dummy_input = torch.randn(1, 4, 128, 128, 128).to(engine.device)
    if request.method == "eigencam":
        heatmap = cam_generator.generate(dummy_input)
    else:
        heatmap = cam_generator.generate(dummy_input)

    output_path = Path(f"./data/storage/heatmaps/{request.prediction_id}_{request.method}.png")
    overlay = overlay_heatmap(np.random.randn(128, 128).astype(np.float32), heatmap, output_path=output_path)
    cam_generator.cleanup()

    return FileResponse(str(output_path), media_type="image/png")


@router.get("/attention/{prediction_id}")
async def get_attention_maps(
    prediction_id: UUID,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    _user: User = Depends(get_current_user),
):
    prediction = await prediction_repo.get_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    heatmap_dir = Path("./data/storage/heatmaps")
    maps = list(heatmap_dir.glob(f"{prediction_id}_*.png")) if heatmap_dir.exists() else []
    return {"prediction_id": str(prediction_id), "attention_maps": [str(m) for m in maps]}
