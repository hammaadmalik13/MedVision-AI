"""Clinical report router."""

from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLPredictionRepository, SQLReportRepository
from medvision.infrastructure.llm import LLMService
from medvision.infrastructure.ml.report_generator import generate_clinical_report, generate_pdf_report
from medvision.presentation.api.dependencies import get_current_user, get_prediction_repo, get_report_repo
from medvision.presentation.api.schemas import ReportResponse

router = APIRouter(prefix="/report", tags=["Reports"])
llm_service = LLMService()


@router.post("/generate/{prediction_id}", response_model=ReportResponse)
async def generate_report(
    prediction_id: str,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    report_repo: SQLReportRepository = Depends(get_report_repo),
    user: User = Depends(get_current_user),
):
    from uuid import UUID

    pred_id = UUID(prediction_id)
    prediction = await prediction_repo.get_by_id(pred_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    mask = np.zeros((128, 128, 128), dtype=np.int64)
    if prediction.mask_uri and Path(prediction.mask_uri).exists():
        mask = nib.load(prediction.mask_uri).get_fdata().astype(np.int64)

    report = generate_clinical_report(
        mask=mask,
        prediction_id=prediction.id,
        study_id=prediction.study_id,
        patient_id=uuid4(),
    )

    report_data = {
        "tumor_volume_cm3": report.tumor_volume_cm3,
        "tumor_percentage": report.tumor_percentage,
        "predicted_grade": report.predicted_grade,
    }
    report.clinical_summary = llm_service.generate_clinical_summary(report_data)
    report.patient_friendly_summary = llm_service.generate_patient_report(report_data)
    report.generated_by = user.id

    pdf_path = Path(f"./data/storage/reports/{report.id}.pdf")
    generate_pdf_report(report, pdf_path)
    report.pdf_uri = str(pdf_path)

    saved = await report_repo.create(report)
    return ReportResponse.model_validate(saved)


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    report_repo: SQLReportRepository = Depends(get_report_repo),
    _user: User = Depends(get_current_user),
):
    from uuid import UUID

    report = await report_repo.get_by_id(UUID(report_id))
    if not report or not report.pdf_uri:
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report.pdf_uri, filename=f"report_{report_id}.pdf", media_type="application/pdf")
