"""
Sentinel API — Main Application

FastAPI app serving XGBoost predictions for fleet notification triage.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models import HealthResponse
from api.services.model_service import ModelService
from api.services.db_service import DatabaseService

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
    print("\n🚀 Starting Sentinel API...")
    _start_time = time.time()

    _model_service = ModelService()
    _db_service = DatabaseService()

    print("✅ Sentinel API ready\n")

    yield

    # --- Shutdown ---
    print("\n🛑 Shutting down Sentinel API...")
    if _db_service:
        _db_service.close()
    print("✅ Cleanup complete\n")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Sentinel API",
    description="Context-aware alert filtering for autonomous vehicle fleets",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow React frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
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
