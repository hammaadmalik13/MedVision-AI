"""Clinical report generation and PDF export."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from medvision.domain.entities.report import ClinicalReport, TumorLocation


def compute_tumor_metrics(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    tumor_mask = mask > 0
    total_voxels = mask.size
    tumor_voxels = tumor_mask.sum()
    voxel_volume_mm3 = np.prod(spacing)

    coords = np.argwhere(tumor_mask)
    if len(coords) > 0:
        centroid = coords.mean(axis=0)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        bbox = (*min_coords, *max_coords)
    else:
        centroid = np.array([0, 0, 0])
        bbox = (0, 0, 0, 0, 0, 0)

    region = _estimate_brain_region(centroid, mask.shape)
    mid_slice_area = tumor_mask[:, :, mask.shape[2] // 2].sum() * spacing[0] * spacing[1]

    return {
        "tumor_volume_cm3": float(tumor_voxels * voxel_volume_mm3 / 1000),
        "tumor_area_mm2": float(mid_slice_area),
        "tumor_percentage": float(tumor_voxels / total_voxels * 100) if total_voxels > 0 else 0.0,
        "tumor_location": TumorLocation(
            region=region,
            centroid=tuple(centroid.tolist()),
            bounding_box=tuple(int(x) for x in bbox),
        ),
        "confidence_score": min(0.95, 0.5 + tumor_voxels / total_voxels * 2) if tumor_voxels > 0 else 0.0,
    }


def _estimate_brain_region(centroid: np.ndarray, shape: tuple) -> str:
    z, y, x = centroid
    d, h, w = shape
    lr = "Right" if x > w / 2 else "Left"
    ap = "Anterior" if y > h / 2 else "Posterior"
    si = "Superior" if z > d / 2 else "Inferior"
    return f"{lr} {ap} {si}"


def predict_grade_experimental(metrics: dict) -> str:
    volume = metrics.get("tumor_volume_cm3", 0)
    if volume < 5:
        return "Low (experimental)"
    if volume < 20:
        return "Moderate (experimental)"
    return "High (experimental)"


def generate_clinical_report(
    mask: np.ndarray,
    prediction_id: UUID,
    study_id: UUID,
    patient_id: UUID,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ClinicalReport:
    metrics = compute_tumor_metrics(mask, spacing)
    return ClinicalReport(
        prediction_id=prediction_id,
        study_id=study_id,
        patient_id=patient_id,
        tumor_volume_cm3=metrics["tumor_volume_cm3"],
        tumor_area_mm2=metrics["tumor_area_mm2"],
        tumor_percentage=metrics["tumor_percentage"],
        tumor_location=metrics["tumor_location"],
        predicted_grade=predict_grade_experimental(metrics),
        confidence_score=metrics["confidence_score"],
        is_experimental=True,
    )


def generate_pdf_report(report: ClinicalReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("MedVision AI - Clinical Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is AI-generated and experimental. "
        "It is NOT a substitute for professional medical diagnosis.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Tumor Volume", f"{report.tumor_volume_cm3:.2f} cm³"],
        ["Tumor Area (mid-slice)", f"{report.tumor_area_mm2:.2f} mm²"],
        ["Tumor Percentage", f"{report.tumor_percentage:.2f}%"],
        ["Location", report.tumor_location.region if report.tumor_location else "N/A"],
        ["Predicted Grade", report.predicted_grade or "N/A"],
        ["Confidence Score", f"{report.confidence_score:.2%}"],
    ]
    table = Table(data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    if report.clinical_summary:
        story.append(Paragraph("<b>Clinical Summary</b>", styles["Heading2"]))
        story.append(Paragraph(report.clinical_summary, styles["Normal"]))

    if report.patient_friendly_summary:
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Patient Summary</b>", styles["Heading2"]))
        story.append(Paragraph(report.patient_friendly_summary, styles["Normal"]))

    doc.build(story)
    return output_path


def generate_pdf_bytes(report: ClinicalReport) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MedVision AI - Clinical Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Tumor Volume: {report.tumor_volume_cm3:.2f} cm³", styles["Normal"]),
        Paragraph(f"Confidence: {report.confidence_score:.2%}", styles["Normal"]),
    ]
    doc.build(story)
    return buffer.getvalue()
