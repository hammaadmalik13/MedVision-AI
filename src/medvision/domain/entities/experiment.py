"""Experiment domain entity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Experiment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    mlflow_run_id: str | None = None
    config_hash: str | None = None
    git_sha: str | None = None
    status: str = "running"
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    created_by: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}
