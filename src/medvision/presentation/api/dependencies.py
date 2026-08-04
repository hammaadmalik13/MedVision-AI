"""FastAPI dependency injection."""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from medvision.application.services.auth_service import get_current_user_from_token
from medvision.domain.entities.user import UserRole
from medvision.infrastructure.db.repositories import (
    SQLExperimentRepository,
    SQLModelVersionRepository,
    SQLMRIStudyRepository,
    SQLPatientRepository,
    SQLPredictionRepository,
    SQLReportRepository,
    SQLUserRepository,
)
from medvision.infrastructure.db.session import get_db_session
from medvision.infrastructure.ml.inference import InferenceEngine
from medvision.infrastructure.storage import get_storage_service

security = HTTPBearer(auto_error=False)


async def get_session() -> AsyncSession:
    async for session in get_db_session():
        yield session


async def get_user_repo(session: AsyncSession = Depends(get_session)) -> SQLUserRepository:
    return SQLUserRepository(session)


async def get_patient_repo(session: AsyncSession = Depends(get_session)) -> SQLPatientRepository:
    return SQLPatientRepository(session)


async def get_study_repo(session: AsyncSession = Depends(get_session)) -> SQLMRIStudyRepository:
    return SQLMRIStudyRepository(session)


async def get_prediction_repo(session: AsyncSession = Depends(get_session)) -> SQLPredictionRepository:
    return SQLPredictionRepository(session)


async def get_report_repo(session: AsyncSession = Depends(get_session)) -> SQLReportRepository:
    return SQLReportRepository(session)


async def get_experiment_repo(session: AsyncSession = Depends(get_session)) -> SQLExperimentRepository:
    return SQLExperimentRepository(session)


async def get_model_repo(session: AsyncSession = Depends(get_session)) -> SQLModelVersionRepository:
    return SQLModelVersionRepository(session)


def get_inference_engine() -> InferenceEngine:
    return InferenceEngine()


def get_storage():
    return get_storage_service()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    user_repo: SQLUserRepository = Depends(get_user_repo),
):
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token_data = get_current_user_from_token(credentials.credentials)
    if not token_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id, _role = token_data
    user = await user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_role(*roles: UserRole):
    async def checker(current_user=Depends(get_current_user)):
        if current_user.role not in roles and current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return checker
