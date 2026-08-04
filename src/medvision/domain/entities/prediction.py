"""Prediction domain entity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PredictionMetrics(BaseModel):
    dice: float | None = None
    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    hausdorff_distance: float | None = None
    tumor_volume_cm3: float | None = None
    tumor_area_mm2: float | None = None
    tumor_percentage: float | None = None
    confidence_score: float | None = None


class Prediction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    study_id: UUID
    model_version_id: UUID
    user_id: UUID | None = None
    mask_uri: str | None = None
    metrics: PredictionMetrics = Field(default_factory=PredictionMetrics)
    inference_time_ms: float | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}
