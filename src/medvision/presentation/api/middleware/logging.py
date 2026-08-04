"""FastAPI middleware."""

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from medvision.logging_config import get_logger
from medvision.presentation.api.routers.metrics import API_REQUESTS

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        API_REQUESTS.labels(method=request.method, endpoint=request.url.path).inc()
        return response


class CORSMiddleware:
    pass  # Use FastAPI built-in CORSMiddleware in main.py
