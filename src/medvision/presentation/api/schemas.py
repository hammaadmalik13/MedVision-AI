"""Pydantic schemas for API."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from medvision.domain.entities.user import UserRole


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8)
    role: UserRole = UserRole.RESEARCHER


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    username: str
    password: str


class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    medical_record_number: str | None = None


class PatientResponse(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    date_of_birth: date | None
    gender: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MRIStudyResponse(BaseModel):
    id: UUID
    patient_id: UUID
    modality_paths: dict[str, str]
    storage_uri: str | None
    shape: list[int] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    id: UUID
    study_id: UUID
    model_version_id: UUID
    mask_uri: str | None
    metrics: dict
    inference_time_ms: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentRequest(BaseModel):
    study_id: UUID
    model_name: str = "unet"


class ReportResponse(BaseModel):
    id: UUID
    prediction_id: UUID
    tumor_volume_cm3: float
    tumor_area_mm2: float
    tumor_percentage: float
    predicted_grade: str | None
    confidence_score: float
    clinical_summary: str | None
    pdf_uri: str | None
    is_experimental: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ExplainRequest(BaseModel):
    prediction_id: UUID
    method: str = "gradcam"
    slice_idx: int | None = None


class ModelVersionResponse(BaseModel):
    id: UUID
    name: str
    version: str
    architecture: str
    metrics: dict
    is_production: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class CompareRequest(BaseModel):
    prediction_ids: list[UUID]


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict] = []


class ExperimentResponse(BaseModel):
    id: UUID
    name: str
    status: str
    metrics: dict
    mlflow_run_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    mlflow: str
