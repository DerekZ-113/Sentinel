# Sentinel Interview Bank

## One-Minute Project Pitch

Sentinel is a full-stack ML systems project for simulated autonomous-vehicle fleet notification triage. The problem is alert fatigue: if every notification is sent to an operator, many are false positives and important events can be buried. The repo simulates fleet events, engineers contextual features, trains an XGBoost classifier, serves predictions through FastAPI, stores predictions in Postgres/TimescaleDB, and shows monitoring plus simulation workflows in a React/TypeScript dashboard.

The safe claim is not that this is a production AV platform. The safe claim is that it is an interview-ready demo of owning an inherited or AI-assisted codebase end to end: understanding the data flow, validating feature parity, testing model-serving behavior, documenting risks, and setting responsible boundaries around simulated results.

## Strongest Current Claims

### Claim: Full-stack system, not just a notebook

Safe wording:
> I built Sentinel as a full-stack ML application around a simulated AV fleet alert problem. It has a data simulator, feature pipeline, trained XGBoost artifacts, FastAPI inference service, Postgres/Timescale persistence, and a React dashboard.

Evidence:
- `fleet_data/generate_fleet_data.py`
- `ml/prepare_data.py`
- `ml/train_classifier.py`
- `api/main.py`
- `api/services/model_service.py`
- `api/services/db_service.py`
- `dashboard/src/App.tsx`
- `docker-compose.yml`

### Claim: Feature parity was treated as a correctness risk

Safe wording:
> A key risk was silent model-serving drift, so the repo centralizes categorical encoding maps and tests that the saved model config uses the expected 28-feature order.

Evidence:
- `ml/constants.py`
- `tests/ml/test_encoding_parity.py`
- `tests/unit/test_feature_engineering.py`

### Claim: API validation and optional auth are implemented and tested

Safe wording:
> The prediction API validates request shape and bounds with Pydantic, and it supports optional API-key auth for prediction writes.

Evidence:
- `api/models.py`
- `api/auth.py`
- `tests/integration/test_api_predict.py`

### Claim: The dashboard is tested and has both full-stack and static demo modes

Safe wording:
> The React dashboard can call the live API or run from generated static JSON in demo mode, and the frontend has Vitest coverage for services and key components.

Evidence:
- `dashboard/src/services/api.ts`
- `dashboard/src/services/demoPredict.ts`
- `dashboard/src/data/*.json`
- `scripts/export_demo_data.py`
- `dashboard/src/__tests__/`

### Claim: The repo is currently verifiable

Safe wording:
> In my current local pass, the backend tests, frontend tests, lint, and dashboard build all pass. The dashboard build has a known bundle-size warning, which I would treat as a P2 optimization rather than a correctness issue.

Evidence:
- `python3 -m pytest tests/ -q --tb=short` -> 242 passed
- `npm test -- --run --reporter=dot` -> 59 passed
- `npm run lint` -> passed
- `npm run build` -> passed with 905.83 kB chunk warning

## Claims To Phrase Carefully

### 64% false-positive reduction

Do say:
> On the simulated dataset used by this repo, the documented XGBoost evaluation reduces false-positive rate from 60.8% to 21.7%, roughly a 64% reduction.

Do not say:
> This reduces false positives by 64% in production.

Why:
The metric is simulated and documented by training/evaluation scripts, but it was not rerun in this first pass.

### 2 million alerts eliminated daily

Do say:
> The README extrapolates the simulated workload to roughly 2 million fewer false alarms daily for a 500-vehicle simulation.

Do not say:
> It saved operators from 2 million real alerts per day.

Why:
This is an extrapolated simulation result, not real deployed operational evidence.

### Production readiness

Do say:
> The repo demonstrates production-shaped concerns like API validation, connection pooling, Docker Compose, nginx proxying, CI, and monitoring UI.

Do not say:
> It is production-grade.

Why:
Schema setup is script-based, auth is optional and narrow, read endpoints are public, and there is no migration system or live deployment proof.

### AI-assisted development

Do say:
> I treat AI-assisted code like inherited code: inspect the current behavior, write or run targeted tests, map claims to repo evidence, and document risks before using the project in interviews.

Do not say:
> AI built it, so I am not responsible for the details.

Why:
Interview defensibility comes from ownership, not authorship purity.

## Technical Deep Dives To Prepare

### Feature Engineering

Know:
- Why `speed_ratio`, `speed_deviation`, and stopped flags exist.
- Why time is encoded as sine/cosine.
- Why `ev_distance`, pedestrian density, object-in-path, traffic, and construction context matter.
- Which interaction features encode operational intuition.

Files:
- `ml/constants.py`
- `ml/prepare_data.py`
- `api/services/model_service.py`
- `tests/unit/test_feature_engineering.py`

### Model Serving

Know:
- How `ModelService` loads `xgboost_model.json` and `xgboost_config.joblib`.
- How fallback scaler reconstruction works when `scaler.joblib` is absent.
- Why feature order matters.
- How thresholding converts raw XGBoost score to `needs_intervention`.

Files:
- `api/services/model_service.py`
- `tests/unit/test_model_service.py`
- `tests/ml/test_model_artifacts.py`

### API Design

Know:
- What each route returns.
- How Pydantic validation protects the model.
- Why DB writes happen after prediction.
- Which endpoints are protected by API key and which are public.

Files:
- `api/main.py`
- `api/models.py`
- `api/routes/*.py`
- `api/auth.py`

### Database Design

Know:
- Difference between `vehicle_metrics` and `predictions`.
- Why the API owns `predictions`.
- Why simulation/training uses `vehicle_metrics`.
- Why `setup_database.py` dropping `vehicle_metrics` is demo behavior.

Files:
- `api/services/db_service.py`
- `setup_database.py`
- `fleet_data/generate_fleet_data.py`

### Frontend Contract

Know:
- Which components fetch their own data.
- How demo mode switches away from API calls.
- What data shapes are shared between backend responses and frontend interfaces.
- Current limitation: UI-level error-state coverage is not exhaustive across every route.

Files:
- `dashboard/src/services/api.ts`
- `dashboard/src/App.tsx`
- `dashboard/src/components/*.tsx`

## Good Interview Stories

### Story: Preventing silent ML inference bugs

Setup:
Feature order mismatch can create wrong predictions without obvious runtime errors.

Action:
Use shared constants for categorical maps, save model config with feature columns, and add tests that compare config feature order and map identity.

Result:
The repo has concrete regression tests for feature-count and feature-order parity.

### Story: Turning a model into a system

Setup:
A model metric alone is not interview-strong unless users can call it and observe it.

Action:
Expose predictions through FastAPI, store them in a database, and build dashboard views for alert history, stats, FP trend, simulation, and model health.

Result:
The repo demonstrates model serving and monitoring-shaped workflows rather than only offline analysis.

### Story: Owning AI-assisted code responsibly

Setup:
Some code may have been generated quickly.

Action:
Inspect repo surfaces, identify unsupported claims, run tests, document architecture, and create a backlog ranked by interview evidence value.

Result:
The project becomes easier to explain, debug, and evolve without pretending every line was handwritten.
