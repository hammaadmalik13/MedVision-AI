"""User domain entity."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    PATIENT = "patient"


class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    username: str
    hashed_password: str
    role: UserRole = UserRole.RESEARCHER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}
