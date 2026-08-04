"""Patient domain entity."""

from datetime import date, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Patient(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None
    first_name: str
    last_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    medical_record_number: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    model_config = {"from_attributes": True}
