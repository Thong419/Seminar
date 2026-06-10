"""Health and version endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse, VersionResponse
from src.config.settings import get_settings
from src.config.settings import AppSettings


router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    description="Returns a simple health indicator for deployment probes.",
)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Get model version",
    description="Returns the deployed model name and version string.",
)
def version(settings: AppSettings = Depends(get_settings)) -> VersionResponse:
    return VersionResponse(model_name=settings.model_name, model_version="1.0.0")
