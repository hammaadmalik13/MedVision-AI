"""MRI Study domain entity."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Modality(str, Enum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"


class MRIStudy(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    study_date: datetime | None = None
    modality_paths: dict[str, str] = Field(default_factory=dict)
    storage_uri: str | None = None
    original_filename: str | None = None
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None
    metadata: dict = Field(default_factory=dict)
    uploaded_by: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def has_all_modalities(self) -> bool:
        required = {m.value for m in Modality}
        return required.issubset(set(self.modality_paths.keys()))

    model_config = {"from_attributes": True}
