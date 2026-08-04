"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medvision import __version__
from medvision.config import get_settings
from medvision.infrastructure.db.session import init_db
from medvision.logging_config import setup_logging
from medvision.presentation.api.middleware.logging import LoggingMiddleware
from medvision.presentation.api.routers import auth, chat, compare, explain, history, metrics, models, report, segment, upload
from medvision.presentation.api.schemas import HealthResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    yield


app = FastAPI(
    title="MedVision AI",
    description="Production-Grade Brain Tumor MRI Segmentation Platform",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")
app.include_router(segment.router, prefix="/api/v1")
app.include_router(report.router, prefix="/api/v1")
app.include_router(explain.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(metrics.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    db_status = "healthy"
    mlflow_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.mlflow_tracking_uri}/health")
            mlflow_status = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception:
        mlflow_status = "unavailable"

    return HealthResponse(
        status="healthy",
        version=__version__,
        database=db_status,
        mlflow=mlflow_status,
    )


def run_server():
    import uvicorn

    uvicorn.run(
        "medvision.presentation.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
