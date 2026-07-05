"""
Sentinel API — Main Application

FastAPI app serving XGBoost predictions for fleet notification triage.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import math
import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.logging_config import setup_logging
from api.models import HealthResponse
from api.services.model_service import ModelService
from api.services.db_service import DatabaseService

setup_logging()
logger = logging.getLogger("sentinel.api")

# ============================================================================
# GLOBALS (initialized on startup)
# ============================================================================

_model_service: ModelService = None
_db_service: DatabaseService = None
_start_time: float = None


def get_model_service() -> ModelService:
    return _model_service


def get_db_service() -> DatabaseService:
    return _db_service


# ============================================================================
# LIFESPAN (startup / shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_service, _db_service, _start_time

    # --- Startup ---
    logger.info("Starting Sentinel API...")
    _start_time = time.time()

    _model_service = ModelService()
    _db_service = DatabaseService()

    logger.info("Sentinel API ready")

    yield

    # --- Shutdown ---
    logger.info("Shutting down Sentinel API...")
    if _db_service:
        _db_service.close()
    logger.info("Cleanup complete")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Sentinel API",
    description="Context-aware alert filtering for autonomous vehicle fleets",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — configurable via env var
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Generic exception handler — fixed detail string: exception text can carry
# connection strings, hostnames, and internal paths (full detail is logged)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def _finite_safe(value):
    """Replace non-finite floats (inf/nan) with strings, recursively.

    Validation errors echo the offending input back to the client, but a
    rejected `1e999` payload puts float('inf') in the error detail — which
    the strict JSON response encoder refuses, turning a correct 422 into
    a 500."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _finite_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    from fastapi.encoders import jsonable_encoder
    return JSONResponse(
        status_code=422,
        content={"detail": _finite_safe(jsonable_encoder(exc.errors()))},
    )


# ============================================================================
# ROUTES
# ============================================================================

from api.routes.predict import router as predict_router
from api.routes.alerts import router as alerts_router
from api.routes.stats import router as stats_router

app.include_router(predict_router, prefix="/api", tags=["Predict"])
app.include_router(alerts_router, prefix="/api", tags=["Alerts"])
app.include_router(stats_router, prefix="/api", tags=["Stats"])


# Plain `def`: health_check() does sync DB I/O — threadpool, not event loop
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Check service health: model loaded, DB connected."""
    db_ok = False
    if _db_service:
        db_ok = _db_service.health_check()

    model_ok = _model_service is not None
    uptime = time.time() - _start_time if _start_time else 0

    return HealthResponse(
        status="healthy" if (model_ok and db_ok) else "degraded",
        model_loaded=model_ok,
        db_connected=db_ok,
        model_features=len(_model_service.feature_columns) if model_ok else 0,
        model_threshold=_model_service.threshold if model_ok else 0,
        uptime_seconds=round(uptime, 1),
    )
