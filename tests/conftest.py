"""
Shared fixtures for Sentinel test suite.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from copy import deepcopy

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# SAMPLE PAYLOADS
# ============================================================================

@pytest.fixture
def sample_payload():
    """A complete valid 'stuck' notification payload."""
    return {
        "vehicle_id": "vehicle_001",
        "speed": 0.0,
        "expected_speed": 35.0,
        "road_type": "downtown",
        "traffic_condition": "heavy",
        "construction_zone": "none",
        "notification_type": "stuck",
        "notification_subtype": None,
        "ev_distance": None,
        "pedestrian_density": 0.3,
        "object_in_path": False,
        "time_since_stop": 120.0,
        "hour_of_day": 14,
    }


@pytest.fixture
def all_notification_payloads():
    """One valid payload per notification type."""
    base = {
        "vehicle_id": "vehicle_test",
        "speed": 30.0,
        "expected_speed": 35.0,
        "road_type": "main_road",
        "traffic_condition": "moderate",
        "construction_zone": "none",
        "pedestrian_density": 0.3,
        "object_in_path": False,
        "time_since_stop": 0.0,
        "hour_of_day": 14,
    }

    payloads = {
        "stuck": {
            **base,
            "notification_type": "stuck",
            "notification_subtype": None,
            "speed": 0.0,
            "time_since_stop": 120.0,
            "ev_distance": None,
        },
        "verification_request": {
            **base,
            "notification_type": "verification_request",
            "notification_subtype": "object_query",
            "pedestrian_density": 0.6,
            "object_in_path": True,
            "ev_distance": None,
        },
        "emergency_vehicle_alert": {
            **base,
            "notification_type": "emergency_vehicle_alert",
            "notification_subtype": None,
            "ev_distance": 50.0,
        },
        "speed_anomaly": {
            **base,
            "notification_type": "speed_anomaly",
            "notification_subtype": None,
            "speed": 15.0,
            "ev_distance": None,
        },
        "impact_l0": {
            **base,
            "notification_type": "impact_l0",
            "notification_subtype": None,
            "speed": 25.0,
            "ev_distance": None,
        },
        "passenger_assist": {
            **base,
            "notification_type": "passenger_assist",
            "notification_subtype": None,
            "speed": 0.0,
            "ev_distance": None,
        },
    }
    return payloads


# ============================================================================
# MODEL SERVICE (session-scoped — loads once)
# ============================================================================

@pytest.fixture(scope="session")
def model_service():
    """Real ModelService loading actual model artifacts from ml/."""
    from api.services.model_service import ModelService
    ml_dir = os.path.join(PROJECT_ROOT, "ml")
    return ModelService(model_dir=ml_dir)


# ============================================================================
# MOCK DB SERVICE
# ============================================================================

@pytest.fixture
def mock_db_service():
    """Mocked DatabaseService — no real DB connection needed."""
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.store_prediction.return_value = 1
    mock.get_recent_alerts.return_value = {
        "alerts": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }
    mock.get_stats.return_value = {
        "time_window_hours": 24,
        "total_alerts": 100,
        "total_flagged": 30,
        "total_suppressed": 70,
        "overall_fp_rate": 0.25,
        "by_type": [],
    }
    mock.get_stats_by_type.return_value = {
        "notification_type": "stuck",
        "total": 50,
        "flagged": 15,
        "suppressed": 35,
        "avg_confidence": 0.85,
        "accuracy": 0.80,
    }
    mock.get_model_health.return_value = {
        "status": "healthy",
        "total_predictions": 100,
        "pct_flagged": 30.0,
        "pct_suppressed": 70.0,
        "avg_confidence": 0.85,
        "accuracy": 0.80,
        "confidence_buckets": {"high": 60, "medium": 30, "low": 10},
        "flagged_by_type": {"stuck": 10, "verification_request": 20},
        "suppressed_by_type": {"stuck": 30, "verification_request": 40},
    }
    return mock


# ============================================================================
# FASTAPI TEST CLIENT
# ============================================================================

@pytest.fixture
def client(model_service, mock_db_service):
    """FastAPI TestClient with real model, mocked DB — bypasses lifespan."""
    import api.main as main_module
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Patch globals before creating client
    original_model = main_module._model_service
    original_db = main_module._db_service
    original_start = main_module._start_time

    main_module._model_service = model_service
    main_module._db_service = mock_db_service
    main_module._start_time = 1000000000.0

    # Create a new app WITHOUT the lifespan to avoid DB connection attempts
    test_app = FastAPI()

    # Copy routes and middleware from the real app
    for route in main_module.app.routes:
        test_app.routes.append(route)

    # Copy middleware
    from fastapi.middleware.cors import CORSMiddleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c

    # Restore
    main_module._model_service = original_model
    main_module._db_service = original_db
    main_module._start_time = original_start
