"""Model comparison router."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLPredictionRepository
from medvision.presentation.api.dependencies import get_current_user, get_prediction_repo
from medvision.presentation.api.schemas import CompareRequest

router = APIRouter(prefix="/compare", tags=["Comparison"])


@router.post("/")
async def compare_predictions(
    request: CompareRequest,
    prediction_repo: SQLPredictionRepository = Depends(get_prediction_repo),
    _user: User = Depends(get_current_user),
):
    if len(request.prediction_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 predictions required")

    comparisons = []
    for pred_id in request.prediction_ids:
        pred = await prediction_repo.get_by_id(pred_id)
        if pred:
            comparisons.append({
                "prediction_id": str(pred.id),
                "model_version_id": str(pred.model_version_id),
                "metrics": pred.metrics.model_dump(),
                "inference_time_ms": pred.inference_time_ms,
            })

    if len(comparisons) < 2:
        raise HTTPException(status_code=404, detail="Insufficient predictions found")

    best_dice = max(comparisons, key=lambda x: x["metrics"].get("confidence_score", 0))
    return {
        "comparisons": comparisons,
        "best_prediction_id": best_dice["prediction_id"],
        "summary": {
            "num_models": len(comparisons),
            "avg_inference_ms": sum(c["inference_time_ms"] or 0 for c in comparisons) / len(comparisons),
        },
    }
