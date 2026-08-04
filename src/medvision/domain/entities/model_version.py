"""Model version domain entity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelVersion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    version: str
    architecture: str
    checkpoint_uri: str
    onnx_uri: str | None = None
    mlflow_model_uri: str | None = None
    metrics: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    is_production: bool = False
    is_staging: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    promoted_at: datetime | None = None

    model_config = {"from_attributes": True}
