"""Segmentation and prediction router."""

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException

from medvision.application.services.segmentation_service import SegmentationService
from medvision.domain.entities.model_version import ModelVersion
from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import (
    SQLModelVersionRepository,
    SQLMRIStudyRepository,
    SQLPredictionRepository,
)
from medvision.infrastructure.ml.inference import InferenceEngine
from medvision.presentation.api.dependencies import (
    get_current_user,
    get_inference_engine,
    get_model_repo,
    get_prediction_repo,
    get_study_repo,
)
from medvision.presentation.api.schemas import PredictionResponse, SegmentRequest

router = APIRouter(tags=["Segmentation"])


@router.post("/segment", response_model=PredictionResponse)
async def segment_mri(
    request: SegmentRequest,
    study_repo: SQLMRIStudyRepository = Depends(get_study_repo),
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    model_repo: SQLModelVersionRepository = Depends(get_model_repo),
    engine: InferenceEngine = Depends(get_inference_engine),
    user: User = Depends(get_current_user),
):
    study = await study_repo.get_by_id(request.study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    model_version = await model_repo.get_production()
    if not model_version:
        checkpoint = Path(f"./data/models/checkpoints/{request.model_name}_best.pt")
        model_version = ModelVersion(
            id=uuid4(),
            name=request.model_name,
            version="1.0.0",
            architecture=request.model_name,
            checkpoint_uri=str(checkpoint),
        )

    service = SegmentationService(engine)
    prediction = await service.segment_study(
        study_id=study.id,
        modality_paths=study.modality_paths,
        model_version=model_version,
        user_id=user.id,
    )
    saved = await prediction_repo.create(prediction)
    return PredictionResponse(
        id=saved.id,
        study_id=saved.study_id,
        model_version_id=saved.model_version_id,
        mask_uri=saved.mask_uri,
        metrics=saved.metrics.model_dump(),
        inference_time_ms=saved.inference_time_ms,
        created_at=saved.created_at,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: SegmentRequest, **kwargs):
    return await segment_mri(request, **kwargs)


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: UUID,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    _user: User = Depends(get_current_user),
):
    prediction = await prediction_repo.get_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionResponse(
        id=prediction.id,
        study_id=prediction.study_id,
        model_version_id=prediction.model_version_id,
        mask_uri=prediction.mask_uri,
        metrics=prediction.metrics.model_dump(),
        inference_time_ms=prediction.inference_time_ms,
        created_at=prediction.created_at,
    )
