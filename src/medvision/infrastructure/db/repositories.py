"""Repository implementations."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from medvision.domain.entities.experiment import Experiment
from medvision.domain.entities.model_version import ModelVersion
from medvision.domain.entities.mri_study import MRIStudy
from medvision.domain.entities.patient import Patient
from medvision.domain.entities.prediction import Prediction, PredictionMetrics
from medvision.domain.entities.report import ClinicalReport, TumorLocation
from medvision.domain.entities.user import User, UserRole
from medvision.infrastructure.db.models import (
    ExperimentModel,
    ModelVersionModel,
    MRIStudyModel,
    PatientModel,
    PredictionModel,
    ReportModel,
    UserModel,
)


class SQLUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user: User) -> User:
        db_user = UserModel(
            id=user.id,
            email=user.email,
            username=user.username,
            hashed_password=user.hashed_password,
            role=user.role.value,
            is_active=user.is_active,
        )
        self.session.add(db_user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        result = await self.session.execute(select(UserModel).where(UserModel.id == user_id))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(UserModel).where(UserModel.email == email))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def get_by_username(self, username: str) -> User | None:
        result = await self.session.execute(select(UserModel).where(UserModel.username == username))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def update(self, user: User) -> User:
        await self.session.execute(
            update(UserModel)
            .where(UserModel.id == user.id)
            .values(
                email=user.email,
                username=user.username,
                role=user.role.value,
                is_active=user.is_active,
            )
        )
        return user

    async def delete(self, user_id: UUID) -> bool:
        result = await self.session.execute(select(UserModel).where(UserModel.id == user_id))
        db_user = result.scalar_one_or_none()
        if db_user:
            await self.session.delete(db_user)
            return True
        return False

    @staticmethod
    def _to_entity(db_user: UserModel) -> User:
        return User(
            id=db_user.id,
            email=db_user.email,
            username=db_user.username,
            hashed_password=db_user.hashed_password,
            role=UserRole(db_user.role),
            is_active=db_user.is_active,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
        )


class SQLPatientRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, patient: Patient) -> Patient:
        db_patient = PatientModel(
            id=patient.id,
            external_id=patient.external_id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            date_of_birth=patient.date_of_birth,
            gender=patient.gender,
            medical_record_number=patient.medical_record_number,
        )
        self.session.add(db_patient)
        await self.session.flush()
        return patient

    async def get_by_id(self, patient_id: UUID) -> Patient | None:
        result = await self.session.execute(select(PatientModel).where(PatientModel.id == patient_id))
        db = result.scalar_one_or_none()
        return Patient.model_validate(db) if db else None

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Patient]:
        result = await self.session.execute(select(PatientModel).offset(skip).limit(limit))
        return [Patient.model_validate(p) for p in result.scalars().all()]

    async def update(self, patient: Patient) -> Patient:
        await self.session.execute(
            update(PatientModel)
            .where(PatientModel.id == patient.id)
            .values(
                first_name=patient.first_name,
                last_name=patient.last_name,
                date_of_birth=patient.date_of_birth,
                gender=patient.gender,
            )
        )
        return patient


class SQLMRIStudyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, study: MRIStudy) -> MRIStudy:
        db_study = MRIStudyModel(
            id=study.id,
            patient_id=study.patient_id,
            study_date=study.study_date,
            modality_paths=study.modality_paths,
            storage_uri=study.storage_uri,
            original_filename=study.original_filename,
            shape=list(study.shape) if study.shape else None,
            spacing=list(study.spacing) if study.spacing else None,
            metadata_=study.metadata,
            uploaded_by=study.uploaded_by,
        )
        self.session.add(db_study)
        await self.session.flush()
        return study

    async def get_by_id(self, study_id: UUID) -> MRIStudy | None:
        result = await self.session.execute(select(MRIStudyModel).where(MRIStudyModel.id == study_id))
        db = result.scalar_one_or_none()
        if not db:
            return None
        return MRIStudy(
            id=db.id,
            patient_id=db.patient_id,
            study_date=db.study_date,
            modality_paths=db.modality_paths,
            storage_uri=db.storage_uri,
            original_filename=db.original_filename,
            shape=tuple(db.shape) if db.shape else None,
            spacing=tuple(db.spacing) if db.spacing else None,
            metadata=db.metadata_,
            uploaded_by=db.uploaded_by,
            created_at=db.created_at,
            updated_at=db.updated_at,
        )

    async def list_by_patient(self, patient_id: UUID) -> list[MRIStudy]:
        result = await self.session.execute(
            select(MRIStudyModel).where(MRIStudyModel.patient_id == patient_id)
        )
        return [
            MRIStudy(
                id=db.id,
                patient_id=db.patient_id,
                modality_paths=db.modality_paths,
                storage_uri=db.storage_uri,
                created_at=db.created_at,
            )
            for db in result.scalars().all()
        ]

    async def update(self, study: MRIStudy) -> MRIStudy:
        await self.session.execute(
            update(MRIStudyModel)
            .where(MRIStudyModel.id == study.id)
            .values(modality_paths=study.modality_paths, storage_uri=study.storage_uri)
        )
        return study


class SQLPredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, prediction: Prediction) -> Prediction:
        db_pred = PredictionModel(
            id=prediction.id,
            study_id=prediction.study_id,
            model_version_id=prediction.model_version_id,
            user_id=prediction.user_id,
            mask_uri=prediction.mask_uri,
            metrics=prediction.metrics.model_dump(),
            inference_time_ms=prediction.inference_time_ms,
            metadata_=prediction.metadata,
        )
        self.session.add(db_pred)
        await self.session.flush()
        return prediction

    async def get_by_id(self, prediction_id: UUID) -> Prediction | None:
        result = await self.session.execute(
            select(PredictionModel).where(PredictionModel.id == prediction_id)
        )
        db = result.scalar_one_or_none()
        if not db:
            return None
        return Prediction(
            id=db.id,
            study_id=db.study_id,
            model_version_id=db.model_version_id,
            user_id=db.user_id,
            mask_uri=db.mask_uri,
            metrics=PredictionMetrics(**db.metrics),
            inference_time_ms=db.inference_time_ms,
            metadata=db.metadata_,
            created_at=db.created_at,
        )

    async def list_by_study(self, study_id: UUID) -> list[Prediction]:
        result = await self.session.execute(
            select(PredictionModel).where(PredictionModel.study_id == study_id)
        )
        return [
            Prediction(
                id=db.id,
                study_id=db.study_id,
                model_version_id=db.model_version_id,
                mask_uri=db.mask_uri,
                metrics=PredictionMetrics(**db.metrics),
                created_at=db.created_at,
            )
            for db in result.scalars().all()
        ]

    async def list_by_user(self, user_id: UUID, skip: int = 0, limit: int = 100) -> list[Prediction]:
        result = await self.session.execute(
            select(PredictionModel)
            .where(PredictionModel.user_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        return [
            Prediction(id=db.id, study_id=db.study_id, model_version_id=db.model_version_id, created_at=db.created_at)
            for db in result.scalars().all()
        ]


class SQLReportRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, report: ClinicalReport) -> ClinicalReport:
        db_report = ReportModel(
            id=report.id,
            prediction_id=report.prediction_id,
            study_id=report.study_id,
            patient_id=report.patient_id,
            tumor_volume_cm3=report.tumor_volume_cm3,
            tumor_area_mm2=report.tumor_area_mm2,
            tumor_percentage=report.tumor_percentage,
            tumor_location=report.tumor_location.model_dump() if report.tumor_location else None,
            predicted_grade=report.predicted_grade,
            confidence_score=report.confidence_score,
            clinical_summary=report.clinical_summary,
            patient_friendly_summary=report.patient_friendly_summary,
            pdf_uri=report.pdf_uri,
            generated_by=report.generated_by,
            is_experimental=report.is_experimental,
        )
        self.session.add(db_report)
        await self.session.flush()
        return report

    async def get_by_id(self, report_id: UUID) -> ClinicalReport | None:
        result = await self.session.execute(select(ReportModel).where(ReportModel.id == report_id))
        db = result.scalar_one_or_none()
        if not db:
            return None
        return ClinicalReport(
            id=db.id,
            prediction_id=db.prediction_id,
            study_id=db.study_id,
            patient_id=db.patient_id,
            tumor_volume_cm3=db.tumor_volume_cm3,
            tumor_area_mm2=db.tumor_area_mm2,
            tumor_percentage=db.tumor_percentage,
            tumor_location=TumorLocation(**db.tumor_location) if db.tumor_location else None,
            predicted_grade=db.predicted_grade,
            confidence_score=db.confidence_score,
            clinical_summary=db.clinical_summary,
            patient_friendly_summary=db.patient_friendly_summary,
            pdf_uri=db.pdf_uri,
            created_at=db.created_at,
        )

    async def get_by_prediction(self, prediction_id: UUID) -> ClinicalReport | None:
        result = await self.session.execute(
            select(ReportModel).where(ReportModel.prediction_id == prediction_id)
        )
        db = result.scalar_one_or_none()
        if not db:
            return None
        return await self.get_by_id(db.id)


class SQLExperimentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, experiment: Experiment) -> Experiment:
        db_exp = ExperimentModel(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            mlflow_run_id=experiment.mlflow_run_id,
            config_hash=experiment.config_hash,
            git_sha=experiment.git_sha,
            status=experiment.status,
            parameters=experiment.parameters,
            metrics=experiment.metrics,
            artifacts=experiment.artifacts,
            created_by=experiment.created_by,
        )
        self.session.add(db_exp)
        await self.session.flush()
        return experiment

    async def get_by_id(self, experiment_id: UUID) -> Experiment | None:
        result = await self.session.execute(
            select(ExperimentModel).where(ExperimentModel.id == experiment_id)
        )
        db = result.scalar_one_or_none()
        return Experiment.model_validate(db) if db else None

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Experiment]:
        result = await self.session.execute(select(ExperimentModel).offset(skip).limit(limit))
        return [Experiment.model_validate(e) for e in result.scalars().all()]

    async def update(self, experiment: Experiment) -> Experiment:
        await self.session.execute(
            update(ExperimentModel)
            .where(ExperimentModel.id == experiment.id)
            .values(status=experiment.status, metrics=experiment.metrics, completed_at=experiment.completed_at)
        )
        return experiment


class SQLModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, model: ModelVersion) -> ModelVersion:
        db_model = ModelVersionModel(
            id=model.id,
            name=model.name,
            version=model.version,
            architecture=model.architecture,
            checkpoint_uri=model.checkpoint_uri,
            onnx_uri=model.onnx_uri,
            mlflow_model_uri=model.mlflow_model_uri,
            metrics=model.metrics,
            config=model.config,
            is_production=model.is_production,
            is_staging=model.is_staging,
        )
        self.session.add(db_model)
        await self.session.flush()
        return model

    async def get_by_id(self, model_id: UUID) -> ModelVersion | None:
        result = await self.session.execute(
            select(ModelVersionModel).where(ModelVersionModel.id == model_id)
        )
        db = result.scalar_one_or_none()
        return ModelVersion.model_validate(db) if db else None

    async def get_production(self) -> ModelVersion | None:
        result = await self.session.execute(
            select(ModelVersionModel).where(ModelVersionModel.is_production.is_(True))
        )
        db = result.scalar_one_or_none()
        return ModelVersion.model_validate(db) if db else None

    async def list_all(self) -> list[ModelVersion]:
        result = await self.session.execute(select(ModelVersionModel))
        return [ModelVersion.model_validate(m) for m in result.scalars().all()]

    async def promote_to_production(self, model_id: UUID) -> ModelVersion:
        await self.session.execute(
            update(ModelVersionModel).values(is_production=False)
        )
        await self.session.execute(
            update(ModelVersionModel)
            .where(ModelVersionModel.id == model_id)
            .values(is_production=True, is_staging=False)
        )
        result = await self.get_by_id(model_id)
        if not result:
            raise ValueError(f"Model {model_id} not found")
        return result
