"""SQLAlchemy database models."""

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="researcher")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions: Mapped[list["PredictionModel"]] = relationship(back_populates="user")
    audit_logs: Mapped[list["AuditLogModel"]] = relationship(back_populates="user")


class PatientModel(Base):
    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    medical_record_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    studies: Mapped[list["MRIStudyModel"]] = relationship(back_populates="patient")


class MRIStudyModel(Base):
    __tablename__ = "mri_studies"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"))
    study_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    modality_paths: Mapped[dict] = mapped_column(JSON, default=dict)
    storage_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    shape: Mapped[list | None] = mapped_column(JSON, nullable=True)
    spacing: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    uploaded_by: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped["PatientModel"] = relationship(back_populates="studies")
    predictions: Mapped[list["PredictionModel"]] = relationship(back_populates="study")


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(50))
    architecture: Mapped[str] = mapped_column(String(100))
    checkpoint_uri: Mapped[str] = mapped_column(String(500))
    onnx_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mlflow_model_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    is_staging: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    predictions: Mapped[list["PredictionModel"]] = relationship(back_populates="model_version")


class PredictionModel(Base):
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    study_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mri_studies.id"))
    model_version_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    user_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    mask_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    study: Mapped["MRIStudyModel"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersionModel"] = relationship(back_populates="predictions")
    user: Mapped["UserModel | None"] = relationship(back_populates="predictions")
    report: Mapped["ReportModel | None"] = relationship(back_populates="prediction", uselist=False)


class ReportModel(Base):
    __tablename__ = "reports"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id"))
    study_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mri_studies.id"))
    patient_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"))
    tumor_volume_cm3: Mapped[float] = mapped_column(Float)
    tumor_area_mm2: Mapped[float] = mapped_column(Float)
    tumor_percentage: Mapped[float] = mapped_column(Float)
    tumor_location: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    predicted_grade: Mapped[str | None] = mapped_column(String(50), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    clinical_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    patient_friendly_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    generated_by: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    is_experimental: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["PredictionModel"] = relationship(back_populates="report")


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="running")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifacts: Mapped[list] = mapped_column(JSON, default=list)
    created_by: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["UserModel | None"] = relationship(back_populates="audit_logs")
