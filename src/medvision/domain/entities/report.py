"""Clinical report domain entity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TumorLocation(BaseModel):
    region: str
    centroid: tuple[float, float, float] | None = None
    bounding_box: tuple[int, int, int, int, int, int] | None = None


class ClinicalReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    prediction_id: UUID
    study_id: UUID
    patient_id: UUID
    tumor_volume_cm3: float
    tumor_area_mm2: float
    tumor_percentage: float
    tumor_location: TumorLocation | None = None
    predicted_grade: str | None = None
    confidence_score: float
    clinical_summary: str | None = None
    patient_friendly_summary: str | None = None
    pdf_uri: str | None = None
    generated_by: UUID | None = None
    is_experimental: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}
