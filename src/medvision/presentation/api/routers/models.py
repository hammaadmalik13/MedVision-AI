"""Model management router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from medvision.domain.entities.user import User, UserRole
from medvision.infrastructure.db.repositories import SQLModelVersionRepository
from medvision.presentation.api.dependencies import get_current_user, get_model_repo, require_role
from medvision.presentation.api.schemas import ModelVersionResponse
from training.models.registry import list_models

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", response_model=list[ModelVersionResponse])
async def list_registered_models(
    model_repo: SQLModelVersionRepository = Depends(get_model_repo),
    _user: User = Depends(get_current_user),
):
    models = await model_repo.list_all()
    return [ModelVersionResponse.model_validate(m) for m in models]


@router.get("/architectures")
async def list_architectures(_user: User = Depends(get_current_user)):
    return {"models": list_models()}


@router.get("/production", response_model=ModelVersionResponse | None)
async def get_production_model(
    model_repo: SQLModelVersionRepository = Depends(get_model_repo),
    _user: User = Depends(get_current_user),
):
    model = await model_repo.get_production()
    return ModelVersionResponse.model_validate(model) if model else None


@router.post("/{model_id}/promote", response_model=ModelVersionResponse)
async def promote_model(
    model_id: UUID,
    model_repo: SQLModelVersionRepository = Depends(get_model_repo),
    _user: User = Depends(require_role(UserRole.ADMIN)),
):
    try:
        model = await model_repo.promote_to_production(model_id)
        return ModelVersionResponse.model_validate(model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
