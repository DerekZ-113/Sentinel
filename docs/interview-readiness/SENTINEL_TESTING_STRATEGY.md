# Sentinel Testing Strategy

## Current Test Inventory

### Python

Command verified in this pass:

```bash
python3 -m pytest tests/ -q
```

Result:

```text
243 passed in 1.57s
```

Coverage areas:

- Pydantic model validation: `tests/unit/test_models.py`
- Feature engineering: `tests/unit/test_feature_engineering.py`
- Model service initialization and predictions: `tests/unit/test_model_service.py`
- DB service SQL behavior with mocked connections: `tests/unit/test_db_service.py`
- Fleet simulation behavior: `tests/unit/test_fleet_data.py`
- API endpoints with real model and mocked DB: `tests/integration/`
- Model artifact and feature parity checks: `tests/ml/`

### Manual Docker Smoke

Command verified in this pass:

```bash
docker-compose -p sentinel_smoke_20260527_183951 up --build
```

Evidence artifact:

- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md`

Observed result:

- DB, API, and dashboard containers started after two Dockerfile fixes found by the smoke run.
- `/health` returned `status: healthy`, `model_loaded: true`, `db_connected: true`, and `model_features: 28`.
- `/api/stats?hours=24` returned 1000 seeded alerts.
- `POST /api/predict` returned a valid prediction for `smoke_vehicle_001`.
- `/api/alerts?limit=5` returned total 1001 and included `smoke_vehicle_001`.
- `http://localhost:3000` returned 200, nginx proxied `/api/stats`, and the rendered dashboard DOM showed `1,001` plus `smoke_vehicle_001`.

Scope:

- This is one local manual smoke run, not an automated CI test or comprehensive DB integration suite.

### Frontend

Command verified in this pass:

```bash
cd dashboard
npm test -- --run --reporter=dot
```

Result:

```text
Test Files 8 passed (8)
Tests 62 passed (62)
```

Coverage areas:

- API client URL/body behavior and tested non-2xx handling: `dashboard/src/__tests__/api.test.ts`
- Demo predictor behavior: `dashboard/src/__tests__/demoPredict.test.ts`
- Dashboard panels and components:
  - `AlertFeed.test.tsx`
  - `DemoBanner.test.tsx`
  - `FPRateChart.test.tsx`
  - `ModelHealth.test.tsx`
  - `OverviewCards.test.tsx`
  - `SimulatePanel.test.tsx`

### Lint And Build

Commands verified in this pass:

```bash
cd dashboard
npm run lint
npm run build
```

Result:

- Lint passed.
- Build passed.
- Build warning: main JS chunk is 905.83 kB after minification, above Vite's 500 kB warning threshold.

## What The Current Tests Prove

- The model artifacts can be loaded in the local test environment.
- The prediction endpoint accepts valid payloads and rejects invalid payloads.
- Optional API-key auth works for the prediction route.
- The model-serving feature vector has the expected shape and tested feature values.
- The model-serving path preserves `ev_distance=0` as a real value instead of treating it as missing.
- The saved model config matches the shared `FEATURE_COLUMNS` list.
- DB service methods use expected SQL operations against mocked connections.
- Dashboard services and main panels render or call expected APIs in tested scenarios.

## What The Current Tests Do Not Prove

- The automated tests do not prove real TimescaleDB integration; that evidence currently comes from one manual Docker smoke run.
- They do not rerun the full training pipeline or reproduce the 64% false-positive reduction.
- They do not prove production security.
- They do not prove static demo deployment is live.
- They do not test every UI-level error state or every route-specific non-2xx case.
- They do not automate Docker Compose startup end to end.

## Interview-Safe Testing Claim

> Sentinel has automated tests across API validation, model-serving feature engineering, model artifact parity, mocked database service behavior, fleet simulation logic, React dashboard behavior, and selected frontend API unhappy paths. The main backend and frontend test suites pass locally. A local Docker Compose smoke run also verified DB/API/model/seed/dashboard behavior. Remaining evidence gaps are metric reproduction, static demo-mode service-switch coverage, automated Docker smoke coverage, and broader UI-level failure coverage.

## Recommended Next Tests

### 1. Static demo mode API-service tests

Purpose:
Separate static demo confidence from full-stack confidence.

Suggested tests:
- With `VITE_DEMO_MODE=true`, `fetchAlerts()` returns bundled JSON and does not call `fetch`.
- With `VITE_DEMO_MODE=true`, `postPredict()` uses `demoPredict()`.

### 4. Automated Docker smoke script

Purpose:
Turn the manually recorded Docker smoke path into a repeatable command.

Suggested checks:
- Start an isolated Compose project.
- Verify `/health`, `/api/stats`, `POST /api/predict`, `/api/alerts`, dashboard HTTP 200, and nginx proxying.
- Tear down only the isolated project and volume.

### 5. Metrics reproduction log

Purpose:
Turn headline model metrics into interview-grade evidence.

Suggested artifact:
Run the simulation/training flow and store command output, sample counts, metrics, and caveats in `SENTINEL_METRICS_REPRODUCTION.md`.
