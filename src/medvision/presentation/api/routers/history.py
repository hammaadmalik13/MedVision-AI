"""History and experiments router."""

from uuid import UUID

from fastapi import APIRouter, Depends

from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLExperimentRepository, SQLPredictionRepository
from medvision.presentation.api.dependencies import get_current_user, get_experiment_repo, get_prediction_repo
from medvision.presentation.api.schemas import ExperimentResponse, PredictionResponse

router = APIRouter(tags=["History"])


@router.get("/history", response_model=list[PredictionResponse])
async def get_history(
    skip: int = 0,
    limit: int = 50,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    user: User = Depends(get_current_user),
):
    predictions = await prediction_repo.list_by_user(user.id, skip=skip, limit=limit)
    return [
        PredictionResponse(
            id=p.id,
            study_id=p.study_id,
            model_version_id=p.model_version_id,
            mask_uri=p.mask_uri,
            metrics=p.metrics.model_dump() if hasattr(p.metrics, "model_dump") else {},
            inference_time_ms=p.inference_time_ms,
            created_at=p.created_at,
        )
        for p in predictions
    ]


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_experiments(
    skip: int = 0,
    limit: int = 50,
    experiment_repo: SQLExperimentRepository = Depends(get_experiment_repo),
    _user: User = Depends(get_current_user),
):
    experiments = await experiment_repo.list_all(skip=skip, limit=limit)
    return [ExperimentResponse.model_validate(e) for e in experiments]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    experiment_repo: SQLExperimentRepository = Depends(get_experiment_repo),
    _user: User = Depends(get_current_user),
):
    experiment = await experiment_repo.get_by_id(experiment_id)
    if not experiment:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse.model_validate(experiment)
