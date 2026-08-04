"""Upload and MRI study router."""

import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import nibabel as nib
from fastapi import APIRouter, Depends, File, Form, UploadFile

from medvision.domain.entities.mri_study import MRIStudy
from medvision.domain.entities.patient import Patient
from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLMRIStudyRepository, SQLPatientRepository
from medvision.infrastructure.storage import StorageService
from medvision.presentation.api.dependencies import get_current_user, get_patient_repo, get_storage, get_study_repo
from medvision.presentation.api.schemas import MRIStudyResponse, PatientCreate, PatientResponse

router = APIRouter(tags=["Upload"])


@router.post("/patients", response_model=PatientResponse)
async def create_patient(
    data: PatientCreate,
    patient_repo: SQLPatientRepository = Depends(get_patient_repo),
    _user: User = Depends(get_current_user),
):
    patient = Patient(**data.model_dump())
    created = await patient_repo.create(patient)
    return PatientResponse.model_validate(created)


@router.post("/upload", response_model=MRIStudyResponse)
async def upload_mri(
    patient_id: UUID = Form(...),
    t1: UploadFile = File(None),
    t1ce: UploadFile = File(None),
    t2: UploadFile = File(None),
    flair: UploadFile = File(None),
    study_repo: SQLMRIStudyRepository = Depends(get_study_repo),
    storage: StorageService = Depends(get_storage),
    user: User = Depends(get_current_user),
):
    study_id = uuid4()
    modality_paths: dict[str, str] = {}
    shape = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for modality, file in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
            if file and file.filename:
                local_file = tmp_path / f"{study_id}_{modality}.nii.gz"
                with open(local_file, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                remote_key = f"studies/{study_id}/{modality}.nii.gz"
                uri = await storage.upload(local_file, remote_key)
                modality_paths[modality] = uri
                if shape is None:
                    img = nib.load(str(local_file))
                    shape = list(img.shape)

    study = MRIStudy(
        id=study_id,
        patient_id=patient_id,
        modality_paths=modality_paths,
        storage_uri=f"studies/{study_id}",
        shape=tuple(shape) if shape else None,
        uploaded_by=user.id,
    )
    created = await study_repo.create(study)
    return MRIStudyResponse(
        id=created.id,
        patient_id=created.patient_id,
        modality_paths=created.modality_paths,
        storage_uri=created.storage_uri,
        shape=list(created.shape) if created.shape else None,
        created_at=created.created_at,
    )


@router.get("/studies/{study_id}", response_model=MRIStudyResponse)
async def get_study(
    study_id: UUID,
    study_repo: SQLMRIStudyRepository = Depends(get_study_repo),
    _user: User = Depends(get_current_user),
):
    study = await study_repo.get_by_id(study_id)
    if not study:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Study not found")
    return MRIStudyResponse(
        id=study.id,
        patient_id=study.patient_id,
        modality_paths=study.modality_paths,
        storage_uri=study.storage_uri,
        shape=list(study.shape) if study.shape else None,
        created_at=study.created_at,
    )
