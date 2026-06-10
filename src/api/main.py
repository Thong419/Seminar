"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.health import router as health_router
from src.api.routes import router as api_router
from src.api.schemas import APIErrorResponse
from src.config.settings import get_settings
from src.utils.logging import configure_logging


logger = logging.getLogger("src.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        start_time = perf_counter()
        response = await call_next(request)
        latency_ms = (perf_counter() - start_time) * 1000.0
        timestamp = datetime.now(UTC).isoformat()
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s timestamp=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            request_id,
            timestamp,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Production FastAPI backend for fake news and misinformation detection.",
        openapi_tags=[
            {"name": "system", "description": "Health and version endpoints."},
            {"name": "prediction", "description": "Prediction and analysis endpoints."},
        ],
    )

    app.add_middleware(RequestContextMiddleware)
    app.include_router(health_router)
    app.include_router(api_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        payload = APIErrorResponse(
            error={
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": {"errors": exc.errors()},
            },
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        payload = APIErrorResponse(
            error={
                "code": str(detail.get("code", "http_error")),
                "message": str(detail.get("message", "An HTTP error occurred.")),
                "details": detail.get("details", {}),
            },
            request_id=request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        logger.exception("request_id=%s unexpected_error=%s", request_id, exc)
        payload = APIErrorResponse(
            error={
                "code": "unexpected_error",
                "message": "An unexpected internal error occurred.",
                "details": {},
            },
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    return app


app = create_app()