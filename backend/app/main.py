"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.database import engine
from app.api.router import api_router
from app.utils.logging import (
    setup_logging,
    get_logger,
    generate_request_id,
    request_id_var,
    request_path_var,
    log_duration,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    # Startup
    setup_logging(level=logging.DEBUG if settings.debug else logging.INFO)
    logger.info("Application starting", extra={"debug": settings.debug})

    if settings.scheduler_enabled:
        from app.scheduler.scheduler import start_scheduler
        start_scheduler()
        logger.info("Scheduler started",
                     extra={"interval_min": settings.collection_interval_minutes})

    # Status must describe the published bundle, not the newest candidate.
    try:
        from app.services.model_release_service import read_active_release
        release = read_active_release()
        settings.active_model_version = release['version'] if release else ''
        logger.info('Published model release detected',
                    extra={'active_model_version':settings.active_model_version})
    except Exception as e:
        logger.warning("Could not detect model version: %s", e)

    yield

    # Shutdown
    if settings.scheduler_enabled:
        from app.scheduler.scheduler import stop_scheduler
        stop_scheduler()
        logger.info("Scheduler stopped")

    await engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ──────────────────────────────────


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Inject a unique request ID into every request for log correlation."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    request_id_var.set(request_id)
    request_path_var.set(f"{request.method} {request.url.path}")
    request.state.request_id = request_id

    t0 = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - t0) * 1000, 1)

    # Add request ID to response headers for client-side debugging
    response.headers["X-Request-ID"] = request_id

    # Structured access log
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response


# ── API Routes ─────────────────────────────────────────────

app.include_router(api_router, prefix="/api/v1")


# ── Global Exception Handlers ──────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with a consistent format."""
    details = []
    for error in exc.errors():
        details.append({
            "field": " → ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "code": error.get("type", "validation_error"),
        })

    logger.warning(
        "Validation error",
        extra={
            "method": request.method,
            "path": request.url.path,
            "error_type": "validation_error",
            "error_count": len(details),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "Request validation failed",
            "details": details,
            "status_code": 422,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "details": [{"message": f"Path {request.url.path} not found"}],
            "status_code": 404,
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle unhandled internal errors."""
    logger.error(
        f"Internal error: {exc}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": [{"message": "An unexpected error occurred. Check server logs."}],
            "status_code": 500,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for any unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )

    detail = str(exc) if settings.debug else "An unexpected error occurred"
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": [{"message": detail}],
            "status_code": 500,
        },
    )


@app.get("/")
async def root():
    return {"service": settings.app_name, "status": "running", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy"}
