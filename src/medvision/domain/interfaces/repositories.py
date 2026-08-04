"""Domain repository interfaces."""

from abc import ABC, abstractmethod
from uuid import UUID

from medvision.domain.entities.experiment import Experiment
from medvision.domain.entities.model_version import ModelVersion
from medvision.domain.entities.mri_study import MRIStudy
from medvision.domain.entities.patient import Patient
from medvision.domain.entities.prediction import Prediction
from medvision.domain.entities.report import ClinicalReport
from medvision.domain.entities.user import User


class UserRepository(ABC):
    @abstractmethod
    async def create(self, user: User) -> User: ...

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> User | None: ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None: ...

    @abstractmethod
    async def update(self, user: User) -> User: ...

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool: ...


class PatientRepository(ABC):
    @abstractmethod
    async def create(self, patient: Patient) -> Patient: ...

    @abstractmethod
    async def get_by_id(self, patient_id: UUID) -> Patient | None: ...

    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Patient]: ...

    @abstractmethod
    async def update(self, patient: Patient) -> Patient: ...


class MRIStudyRepository(ABC):
    @abstractmethod
    async def create(self, study: MRIStudy) -> MRIStudy: ...

    @abstractmethod
    async def get_by_id(self, study_id: UUID) -> MRIStudy | None: ...

    @abstractmethod
    async def list_by_patient(self, patient_id: UUID) -> list[MRIStudy]: ...

    @abstractmethod
    async def update(self, study: MRIStudy) -> MRIStudy: ...


class PredictionRepository(ABC):
    @abstractmethod
    async def create(self, prediction: Prediction) -> Prediction: ...

    @abstractmethod
    async def get_by_id(self, prediction_id: UUID) -> Prediction | None: ...

    @abstractmethod
    async def list_by_study(self, study_id: UUID) -> list[Prediction]: ...

    @abstractmethod
    async def list_by_user(self, user_id: UUID, skip: int = 0, limit: int = 100) -> list[Prediction]: ...


class ReportRepository(ABC):
    @abstractmethod
    async def create(self, report: ClinicalReport) -> ClinicalReport: ...

    @abstractmethod
    async def get_by_id(self, report_id: UUID) -> ClinicalReport | None: ...

    @abstractmethod
    async def get_by_prediction(self, prediction_id: UUID) -> ClinicalReport | None: ...


class ExperimentRepository(ABC):
    @abstractmethod
    async def create(self, experiment: Experiment) -> Experiment: ...

    @abstractmethod
    async def get_by_id(self, experiment_id: UUID) -> Experiment | None: ...

    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Experiment]: ...

    @abstractmethod
    async def update(self, experiment: Experiment) -> Experiment: ...


class ModelVersionRepository(ABC):
    @abstractmethod
    async def create(self, model: ModelVersion) -> ModelVersion: ...

    @abstractmethod
    async def get_by_id(self, model_id: UUID) -> ModelVersion | None: ...

    @abstractmethod
    async def get_production(self) -> ModelVersion | None: ...

    @abstractmethod
    async def list_all(self) -> list[ModelVersion]: ...

    @abstractmethod
    async def promote_to_production(self, model_id: UUID) -> ModelVersion: ...
