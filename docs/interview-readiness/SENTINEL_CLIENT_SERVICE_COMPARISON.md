# Sentinel Client And Service Comparison

This repo has one primary web client and several service layers. This comparison helps explain boundaries without inventing extra platforms.

## Components

| Component | Files | Role | Depends on | Tested by |
|---|---|---|---|---|
| React dashboard client | `dashboard/src/App.tsx`, `dashboard/src/components/*` | Presents monitoring, charts, alert feed, and simulation UI | `dashboard/src/services/api.ts` | `dashboard/src/__tests__/*.test.tsx` |
| Frontend API client | `dashboard/src/services/api.ts` | Encapsulates fetch calls and demo-mode switching | FastAPI routes or bundled JSON | `dashboard/src/__tests__/api.test.ts` |
| Demo predictor | `dashboard/src/services/demoPredict.ts` | Provides heuristic static-demo predictions | Notification payload | `dashboard/src/__tests__/demoPredict.test.ts` |
| FastAPI app | `api/main.py`, `api/routes/*.py` | Validates requests, exposes routes, coordinates services | `ModelService`, `DatabaseService` | `tests/integration/*` |
| Model service | `api/services/model_service.py` | Loads XGBoost artifacts, engineers features, predicts | `ml/xgboost_model.json`, `ml/xgboost_config.joblib` | `tests/unit/test_model_service.py`, `tests/unit/test_feature_engineering.py` |
| Database service | `api/services/db_service.py` | Stores predictions and computes dashboard stats | Postgres/TimescaleDB | `tests/unit/test_db_service.py` |
| ML scripts | `ml/*.py` | Prepare data, train/evaluate model, save artifacts | `vehicle_metrics`, numpy/joblib files | `tests/ml/*` |
| Fleet simulation scripts | `fleet_data/*.py` | Generate simulated vehicle metrics and baseline stats | Postgres/TimescaleDB | `tests/unit/test_fleet_data.py` |

## Boundary Notes

### Client vs API

The dashboard is not responsible for feature engineering or model inference in full-stack mode. It submits payloads and displays API responses.

Exception:
In static demo mode, `demoPredict.ts` provides heuristic predictions so the UI can be demonstrated without API/DB.

### API routes vs services

Routes handle HTTP concerns:

- request validation
- response models
- auth dependency
- exception handling

Services handle domain/infrastructure concerns:

- model loading and inference
- database connection pooling
- SQL queries and aggregation

### Model service vs ML training scripts

`ModelService` serves committed artifacts. It does not train. Training lives under `ml/`.

### Database service vs database setup

`DatabaseService` creates and uses the `predictions` table.

`setup_database.py` creates the `vehicle_metrics` table for simulation/training and is demo-reset oriented.

## Interview-Safe Boundary Claim

> Sentinel separates presentation, HTTP routing, model inference, persistence, and offline ML jobs into distinct modules. The boundaries are not enterprise-heavy, but they are clear enough to test and explain.

## Weak Boundary Areas

- API route handlers expose raw exception text in HTTP 500 details.
- Frontend API helpers have tested non-2xx handling for selected paths, but component-level error coverage is not exhaustive.
- `ml/prepare_data.py` should import `FEATURE_COLUMNS` directly to strengthen training/inference parity.
- DB integration is mostly tested through mocks in the current automated suite.
