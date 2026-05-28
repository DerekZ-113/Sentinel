# Sentinel Architecture Overview

## Project Purpose

Sentinel is a full-stack autonomous-vehicle fleet notification triage project. Based on the current repository, it simulates fleet notifications, engineers contextual features, trains an XGBoost classifier, serves predictions through a FastAPI API, stores prediction records in Postgres/TimescaleDB, and displays monitoring views in a React/TypeScript dashboard.

Interview-safe framing:

> Sentinel is a portfolio-scale, full-stack ML systems project for reducing alert fatigue in a simulated autonomous-vehicle fleet. It demonstrates data simulation, feature engineering, supervised classification, API serving, persistence, monitoring UI, Docker orchestration, and test coverage.

Avoid framing it as:

> A production fleet platform, a real deployed AV operations system, or a live-scale production monitoring system.

## Current Technical Surfaces

### Frontend / Client

- `dashboard/src/App.tsx` lays out the dashboard sections and navigation.
- `dashboard/src/components/` contains monitoring and simulation UI components:
  - `OverviewCards.tsx`
  - `AlertFeed.tsx`
  - `TypeBreakdown.tsx`
  - `FPRateChart.tsx`
  - `SimulatePanel.tsx`
  - `ModelHealth.tsx`
  - `DemoBanner.tsx`
- `dashboard/src/services/api.ts` defines the frontend API client and TypeScript response shapes.
- `dashboard/src/services/demoPredict.ts` provides a static-demo heuristic predictor when `VITE_DEMO_MODE=true`.
- `dashboard/src/data/*.json` contains generated static demo data.

### Backend / API

- `api/main.py` creates the FastAPI app, configures CORS, initializes singleton services in lifespan startup, registers routers, and exposes `/health`.
- `api/models.py` defines Pydantic request/response schemas and validation enums.
- `api/routes/predict.py` exposes `POST /api/predict`.
- `api/routes/alerts.py` exposes `GET /api/alerts`.
- `api/routes/stats.py` exposes aggregate stats, model health, false-positive trend, and per-type stats endpoints.
- `api/auth.py` implements optional API-key auth for prediction requests.

### Database / Storage

- `api/services/db_service.py` manages a `psycopg2.pool.ThreadedConnectionPool`.
- `DatabaseService` creates and queries a `predictions` table used by the API/dashboard.
- `setup_database.py` creates the `vehicle_metrics` TimescaleDB hypertable used by simulation/training workflows.
- `docker-compose.yml` runs TimescaleDB/Postgres as the `db` service.

### ML Pipeline / Jobs

- `fleet_data/generate_fleet_data.py` simulates 500 vehicles over 7 days and writes `vehicle_metrics`.
- `fleet_data/baseline_alerter.py` computes the all-notifications baseline.
- `ml/prepare_data.py` loads data, engineers features, scales them, and saves processed arrays.
- `ml/train_classifier.py` trains/evaluates the XGBoost classifier and saves model artifacts.
- `ml/run_pipeline.py` runs the preparation and training scripts in order.
- `scripts/seed_demo.py` runs a smaller simulation and sends notifications through the API.
- `scripts/export_demo_data.py` generates static dashboard JSON data without requiring Docker, DB, or API.

### Auth

- `api/auth.py` checks `X-API-Key` against `API_KEY`.
- Auth is intentionally skipped when `API_KEY` is empty.
- The dashboard sends `VITE_API_KEY` when present.

### Testing Infrastructure

- `tests/` contains pytest unit, integration, and ML tests.
- `dashboard/src/__tests__/` contains Vitest and React Testing Library tests.
- `.github/workflows/ci.yml` runs Python tests, Python coverage, frontend tests, and frontend lint on push/PR to `main`.

### Deployment / Config

- `docker-compose.yml` runs DB, API, and dashboard.
- `Dockerfile.api` builds the FastAPI container.
- `Dockerfile.dashboard` builds and serves the Vite dashboard through nginx. Docker Compose builds force `VITE_DEMO_MODE=false`, so the dashboard uses live API data even though `dashboard/.env.production` supports static demo builds.
- `nginx.conf` proxies `/api`, `/health`, `/docs`, and `/openapi.json` to the API.
- `dashboard/vercel.json` and static demo data support static dashboard deployment.
- `.env.example` documents expected environment variables.

### Docs

- `README.md` is the primary project narrative and quick-start doc.
- `CLAUDE.md` and `AGENTS.md` provide local developer/agent guidance.
- `docs/interview-readiness/` contains this interview-focused documentation set.

## Surfaces Not Present

- No autonomous agent runtime is implemented in the application.
- No packaged command-line interface exists; there are Python scripts with CLI-style entrypoints.
- No real external fleet ingestion service is implemented.
- No production migration tool is configured; schema setup is script-based and demo-oriented.
- No live hosted environment is proven by this repo alone.

## High-Level Flow

```text
Fleet simulation
  -> vehicle_metrics in TimescaleDB
  -> feature engineering
  -> XGBoost training/evaluation
  -> committed model artifacts
  -> FastAPI prediction service
  -> predictions table
  -> React dashboard monitoring and simulation UI
```

Static demo mode uses a parallel path:

```text
Mini simulation + ModelService
  -> dashboard/src/data/*.json
  -> Vite static app with VITE_DEMO_MODE=true
  -> dashboard views without API/DB
```

## Architecture Tradeoffs

- The project prioritizes explainability and interview readability over scale.
- The model uses engineered tabular features rather than a larger or less explainable architecture.
- The database setup is convenient for demos but is not a production migration strategy.
- The API uses module-level service singletons, which is simple and testable here, but less flexible than dependency-injection wiring for larger systems.
- The frontend API client now checks non-2xx responses, but UI-level error coverage is not exhaustive across every route.
- A local Docker smoke run verifies DB/API/model/seed/dashboard behavior, but this remains manual evidence rather than an automated deployment or CI guarantee.
