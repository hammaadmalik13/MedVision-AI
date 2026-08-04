"""Prometheus metrics endpoint."""

from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

router = APIRouter(tags=["Metrics"])

INFERENCE_COUNT = Counter("medvision_inference_total", "Total inference requests")
INFERENCE_LATENCY = Histogram("medvision_inference_latency_seconds", "Inference latency")
API_REQUESTS = Counter("medvision_api_requests_total", "Total API requests", ["method", "endpoint"])


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
