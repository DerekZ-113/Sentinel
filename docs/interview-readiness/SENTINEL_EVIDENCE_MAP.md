# Sentinel Evidence Map

Use this map to separate implemented behavior, tested behavior, documented behavior, and future work. Support levels use these labels:

- Strongly supported
- Partially supported
- Implemented but untested
- Documented but not implemented
- Aspirational / future work
- Unsupported / should remove

## Claim 1

Claim:
Sentinel is a full-stack ML-backed alert triage system for simulated AV fleet notifications.

Support level:
Strongly supported, if phrased as a simulated portfolio/demo system rather than a production fleet system.

Evidence in repo:
- `README.md:5-9` describes the project as a Dockerized FastAPI + React + XGBoost system and frames the data as a 500-vehicle simulation.
- `docker-compose.yml:3-47` defines DB, API, and dashboard services.
- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md` records a local Docker Compose smoke pass across DB, API, model loading, seed, prediction insert, nginx, and dashboard rendering.
- `fleet_data/generate_fleet_data.py:1-10` describes the simulated fleet dataset.
- `ml/prepare_data.py:43-72` loads simulation data from `vehicle_metrics`; `ml/train_classifier.py:34-146` trains/evaluates an XGBoost classifier.
- `api/main.py:48-57` initializes `ModelService` and `DatabaseService`; `api/routes/predict.py:19-39` runs prediction and storage.
- `dashboard/src/App.tsx:12-18` defines dashboard sections for overview, alerts, type breakdown, FP trend, simulation, and model health.

Implemented:
Yes. The repo has simulation scripts, feature engineering, model artifacts, API serving, DB persistence code, and a dashboard.

Tested:
Partially. API/model/feature tests exist, DB tests are mocked, frontend tests cover service calls and components, and one local Docker full-stack smoke run is recorded.

Manually verified:
Current verification run:
- `python3 -m pytest tests/ -q` -> 243 passed
- `npm test -- --run --reporter=dot` in `dashboard/` -> 8 files, 62 tests passed
- `npm run lint` in `dashboard/` -> passed
- `npm run build` in `dashboard/` -> passed with a 905.83 kB chunk-size warning
- `docker-compose -p sentinel_smoke_20260527_183951 up --build` -> DB, API, and dashboard started after Dockerfile fixes; `/health` reported healthy with model loaded, DB connected, and 28 model features; `/api/stats` returned 1000 seeded alerts; `POST /api/predict` inserted `smoke_vehicle_001`; `/api/alerts` and rendered dashboard DOM showed 1001 total alerts and `smoke_vehicle_001`.

Documented:
Yes in `README.md` and this interview-readiness doc set.

Risk / weakness:
Only one local manual Docker smoke run is recorded. Do not imply production deployment, production scale, real fleet deployment, automated Docker CI coverage, or live operational use.

Interview-safe wording:
Sentinel is a simulated AV fleet notification triage project that connects data simulation, XGBoost inference, a FastAPI service, Postgres/Timescale persistence code, and a React monitoring dashboard.

Smallest strengthening step:
Automate the Docker smoke path or add a short script that reruns `/health`, `/api/stats`, `POST /api/predict`, `/api/alerts`, and dashboard load checks.

## Claim 2

Claim:
The model uses 28 engineered features with shared feature ordering and categorical encodings.

Support level:
Partially supported. The 28-feature model-serving path and categorical map sharing are strongly supported; the "shared feature ordering" wording is too broad because `ml/prepare_data.py` still defines a local feature list.

Evidence in repo:
- `ml/constants.py:6-43` defines categorical maps and `FEATURE_COLUMNS`.
- `api/services/model_service.py:18-21` imports maps from `ml.constants`.
- `api/services/model_service.py:114-197` engineers and assembles 28 inference features.
- `ml/prepare_data.py:18-21` imports categorical maps from `ml.constants`.
- `ml/prepare_data.py:200-243` defines a local `feature_columns` list instead of importing `FEATURE_COLUMNS`.
- `tests/ml/test_encoding_parity.py:31-54` verifies model-service map identity and saved config feature order/count.
- `tests/unit/test_feature_engineering.py:35-55` maps the 28 inference feature positions, includes an `ev_distance=0` regression for inference normalization, and `tests/unit/test_feature_engineering.py:377-393` checks inference feature-vector order.

Implemented:
Yes for 28 inference features and shared categorical maps. Partially for feature-order sharing.

Tested:
Yes for saved model config parity, inference feature order, and a valid zero-value EV-distance edge case. Not enough to prove training code cannot drift because the training prep list is duplicated.

Manually verified:
The Python suite passed in the current run: `python3 -m pytest tests/ -q` reported `243 passed in 1.57s`.

Documented:
Yes. `README.md:184-220` lists the 28 features.

Risk / weakness:
The phrase "shared feature ordering" overstates the implementation. The saved model config is tested against constants, but training prep still has a duplicated list.

Interview-safe wording:
The repo uses 28 engineered features. Categorical maps are shared through `ml/constants.py`, and tests check that the saved model config and inference feature order match the expected 28 columns.

Smallest strengthening step:
Refactor `ml/prepare_data.py` to use `FEATURE_COLUMNS` directly, then add or update a test proving `prepare_training_data()` returns that exact order.

## Claim 3

Claim:
The API exposes prediction, alert history, aggregate stats, model health, false-positive trend, per-type stats, and health endpoints.

Support level:
Strongly supported.

Evidence in repo:
- `api/routes/predict.py:19-39` implements `POST /api/predict`.
- `api/routes/alerts.py:18-35` implements `GET /api/alerts`.
- `api/routes/stats.py:17-28` implements `GET /api/stats`.
- `api/routes/stats.py:31-42` implements `GET /api/stats/model-health`.
- `api/routes/stats.py:45-59` implements `GET /api/stats/fp-over-time`.
- `api/routes/stats.py:62-73` implements `GET /api/stats/{notification_type}`.
- `api/main.py:120-137` implements `GET /health`.
- `tests/integration/test_api_predict.py:10-40`, `tests/integration/test_api_stats.py:13-107`, and `tests/integration/test_api_alerts.py:10-62` cover these endpoint families with a real model fixture and mocked DB service.

Implemented:
Yes.

Tested:
Yes for route behavior against mocked DB responses. Prediction uses the real model fixture from `tests/conftest.py:109-114`.

Manually verified:
The Python suite passed in the current run.

Documented:
Partially. `README.md:136-145` documents most endpoints but does not list `/api/stats/fp-over-time`.

Risk / weakness:
Route handlers include raw exception text in 500 responses, e.g. `api/routes/predict.py:40-42`, `api/routes/alerts.py:33-35`, and `api/main.py:97-104`.

Interview-safe wording:
The FastAPI app exposes prediction, alert-history, stats, model-health, FP-trend, per-type stats, and health endpoints. The integration tests exercise these routes with the real model and a mocked persistence layer.

Smallest strengthening step:
Update the README endpoint table to include `/api/stats/fp-over-time` and add a short note that current tests mock the DB layer.

## Claim 4

Claim:
Prediction requests are validated with Pydantic.

Support level:
Strongly supported.

Evidence in repo:
- `api/models.py:17-51` defines request enums.
- `api/models.py:58-77` defines `NotificationPayload` with numeric bounds and optional fields.
- `api/routes/predict.py:19-20` binds the prediction route to `NotificationPayload`.
- `tests/integration/test_api_predict.py:42-63` covers missing fields, invalid speed, invalid road type, and pedestrian-density boundaries.
- `tests/unit/test_models.py:22-45` covers enum values, `tests/unit/test_models.py:52-154` covers payload validation, and `tests/unit/test_models.py:161-212` covers response model construction.

Implemented:
Yes.

Tested:
Yes for representative validation failures and response model shapes.

Manually verified:
The Python suite passed in the current run.

Documented:
Partially. `README.md:136-147` lists the API endpoints but does not show request validation constraints.

Risk / weakness:
The public docs do not show a canonical valid payload or common 422 cases.

Interview-safe wording:
`POST /api/predict` validates notification payloads with Pydantic enums and bounds before model inference.

Smallest strengthening step:
Add one valid `POST /api/predict` payload and one 422 example to the frontend/backend contract or README.

## Claim 5

Claim:
The API supports optional API-key auth for prediction requests.

Support level:
Strongly supported for `POST /api/predict` only.

Evidence in repo:
- `api/auth.py:7-16` reads `API_KEY` at request time and rejects wrong/missing keys only when `API_KEY` is set.
- `api/routes/predict.py:19` attaches `Depends(verify_api_key)` to `POST /api/predict`.
- `dashboard/src/services/api.ts:177-180` sends `VITE_API_KEY` as `X-API-Key` when present.
- `tests/integration/test_api_predict.py:71-89` covers skipped auth, rejected wrong key, and accepted correct key.

Implemented:
Yes, for prediction writes.

Tested:
Yes.

Manually verified:
The Python suite passed in the current run.

Documented:
Partially. `.env.example` includes `API_KEY`; the README does not clearly state which routes are protected.

Risk / weakness:
Read-only endpoints are not protected. This is optional demo auth, not comprehensive application security.

Interview-safe wording:
Sentinel has optional API-key protection on prediction writes. If `API_KEY` is unset, auth is intentionally disabled for local/demo use.

Smallest strengthening step:
Add a short security-boundary note listing protected and unprotected routes.

## Claim 6

Claim:
The database layer uses connection pooling and stores prediction records for dashboard queries.

Support level:
Partially supported. The code and mocked unit tests support the query behavior, and one local Docker smoke run verified startup, DB connection, seeded stats, prediction insertion, and dashboard reads against a live Timescale/Postgres container.

Evidence in repo:
- `api/services/db_service.py:24-37` creates a `ThreadedConnectionPool`.
- `api/services/db_service.py:45-81` creates the `predictions` table and indexes.
- `api/services/db_service.py:87-131` inserts prediction records.
- `api/services/db_service.py:137-179`, `api/services/db_service.py:181-256`, `api/services/db_service.py:258-298`, and `api/services/db_service.py:304-380` query alert/feed stats and model health.
- `tests/unit/test_db_service.py:1-5` explicitly says all DB interactions are mocked.
- `tests/unit/test_db_service.py:41-60`, `tests/unit/test_db_service.py:84-128`, `tests/unit/test_db_service.py:135-229`, and `tests/unit/test_db_service.py:282-374` cover pool/table/write/query behavior with mocks.
- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md` records `/health`, `/api/stats`, `POST /api/predict`, `/api/alerts`, and dashboard rendering against a live Docker DB.

Implemented:
Yes.

Tested:
Yes with mocked psycopg2 connections/cursors. Additionally, one manual Docker smoke run exercised the live DB-backed API and dashboard path.

Manually verified:
The Python suite passed in the current run. The local Docker smoke run passed after two Dockerfile fixes, with `/health` reporting `db_connected: true`, `/api/stats` returning 1000 seeded alerts, `POST /api/predict` returning a prediction, and `/api/alerts` showing `smoke_vehicle_001`.

Documented:
Partially. README mentions TimescaleDB and dashboard data flow, but does not clearly distinguish `vehicle_metrics` from `predictions`.

Risk / weakness:
The live DB evidence is a manual smoke run, not comprehensive DB integration coverage or a CI gate. Mocked DB tests still do most of the automated coverage. `setup_database.py:33-35` drops `vehicle_metrics`, which is demo reset behavior.

Interview-safe wording:
The API persistence layer uses a psycopg2 connection pool and unit-tested SQL methods for storing predictions and computing dashboard stats. A local Docker smoke run verified DB connection, seeded stats, prediction insertion, and dashboard reads against Timescale/Postgres, but this is not comprehensive automated DB integration testing.

Smallest strengthening step:
Convert the manual Docker smoke checklist into a repeatable script or CI/manual pre-demo command.

## Claim 7

Claim:
The project includes a React/TypeScript dashboard for monitoring and simulating predictions.

Support level:
Strongly supported.

Evidence in repo:
- `dashboard/src/App.tsx:12-18` defines dashboard navigation sections.
- `dashboard/src/App.tsx:156-183` renders overview, alert feed, type breakdown, FP trend, simulation, and model health sections.
- `dashboard/src/components/SimulatePanel.tsx:42-65` submits prediction payloads through `postPredict()`.
- `dashboard/src/services/api.ts:23-120` defines TypeScript response/request shapes.
- `dashboard/src/__tests__/*.test.tsx` and `dashboard/src/__tests__/api.test.ts` test core components and service calls.

Implemented:
Yes.

Tested:
Yes with Vitest and React Testing Library.

Manually verified:
Frontend tests, lint, and production build passed in the current run.

Documented:
Yes at a high level in `README.md:151-159`.

Risk / weakness:
`dashboard/src/services/api.ts` now checks `res.ok` through a shared `fetchJson()` helper and tests selected 401/422/500 paths. Broader UI-level failure handling is still not exhaustive.

Interview-safe wording:
Sentinel includes a React/TypeScript dashboard with monitoring panels and an interactive prediction form, covered by component and service tests.

Smallest strengthening step:
Add static demo-mode service-switch tests proving demo helpers do not call `fetch`.

## Claim 8

Claim:
The dashboard can run in static demo mode without the backend.

Support level:
Partially supported. The switch is implemented and the heuristic predictor is tested, but the service-level demo-mode switch is not explicitly tested and static deployment was not manually opened in this review.

Evidence in repo:
- `dashboard/src/services/api.ts:9-17` imports bundled JSON and reads `VITE_DEMO_MODE`.
- `dashboard/src/services/api.ts:126-199` returns bundled JSON or `demoPredict()` when demo mode is true.
- `dashboard/src/services/demoPredict.ts:33-101` implements heuristic predictions.
- `dashboard/src/data/*.json` contains static dashboard data.
- `scripts/export_demo_data.py:1-11` documents local static data generation without Docker/API/DB.
- `scripts/export_demo_data.py:265-280` runs simulation, loads the model, predicts, and computes static JSON outputs.
- `dashboard/src/__tests__/demoPredict.test.ts:25-97` tests heuristic demo predictions.

Implemented:
Yes.

Tested:
Partially. `demoPredict()` is tested. The `VITE_DEMO_MODE=true` branches in `api.ts` are not directly tested.

Manually verified:
Frontend tests and build passed. Static demo mode was not opened or separately built in this review. The Docker dashboard smoke instead verified live API mode after forcing `VITE_DEMO_MODE=false` in `Dockerfile.dashboard`.

Documented:
Partially. The main README focuses on Docker Compose full-stack startup.

Risk / weakness:
`demoPredict.ts` is a heuristic, not XGBoost inference. Do not use static demo mode as evidence of model accuracy.

Interview-safe wording:
The dashboard has a static demo mode backed by generated JSON and a heuristic predictor, useful for presenting the UI without running the API or DB.

Smallest strengthening step:
Add tests that force `VITE_DEMO_MODE=true` and assert `fetchHealth()`, `fetchAlerts()`, and `postPredict()` do not call `fetch`.

## Claim 9

Claim:
The repo includes an end-to-end ML workflow from simulation through training artifacts.

Support level:
Partially supported. The scripts and artifacts exist, but this review did not rerun the full workflow and CI does not run it.

Evidence in repo:
- `fleet_data/generate_fleet_data.py:1-10` describes simulation output.
- `fleet_data/generate_fleet_data.py:475-493` configures 500 vehicles, 7 days, and expected record count.
- `fleet_data/baseline_alerter.py:26-65` computes all-notifications baseline metrics.
- `ml/prepare_data.py:43-72` loads `vehicle_metrics`; `ml/prepare_data.py:78-186` engineers features; `ml/prepare_data.py:192-276` prepares training arrays.
- `ml/train_classifier.py:34-146` trains/evaluates XGBoost; `ml/train_classifier.py:268-282` saves model artifacts.
- `ml/run_pipeline.py:41-45` runs data preparation and training in sequence.
- `ml/xgboost_model.json` and `ml/xgboost_config.joblib` are committed.
- `tests/ml/test_model_artifacts.py:19-37` verifies committed artifact existence and config shape.

Implemented:
Yes, as scripts and committed artifacts.

Tested:
Partially. Tests cover feature engineering, model artifacts, and real-model prediction behavior. The full DB-backed training pipeline is not run by CI.

Manually verified:
Model artifacts loaded successfully during the Python test run.

Documented:
Yes at a high level in `README.md`.

Risk / weakness:
Calling this "end-to-end" can sound like a tested production pipeline. It is better described as an offline script workflow unless a full rerun is documented.

Interview-safe wording:
The repo contains an offline ML workflow from simulated DB records through feature preparation, XGBoost training/evaluation, and committed model artifacts. The full workflow still needs a reproduction log.

Smallest strengthening step:
Create `SENTINEL_METRICS_REPRODUCTION.md` with exact commands, DB assumptions, dataset size, random seed behavior, and training output.

## Claim 10

Claim:
The README-reported 64% false-positive reduction is supported by the repo.

Support level:
Partially supported. The repo documents the number and contains the code path that computes it, but there is no committed reproduction log or regression test for the exact metric.

Evidence in repo:
- `README.md:7-9` states the headline metric and simulation caveat.
- `README.md:73-104` lists performance and operator-impact tables.
- `fleet_data/baseline_alerter.py:26-65` computes baseline false-positive and precision metrics from `vehicle_metrics`.
- `ml/train_classifier.py:131-146` computes baseline/model FP rate, precision, recall, F1, ROC-AUC, PR-AUC, and FP reduction.
- `ml/train_classifier.py:258-266` prints a README-style summary table.
- `ml/xgboost_model.json` and `ml/xgboost_config.joblib` are present and artifact tests pass.

Implemented:
The metric calculation is implemented in training code, and model artifacts exist.

Tested:
No direct test asserts the exact 64% reduction. Existing tests validate artifacts and prediction behavior, not the README metric.

Manually verified:
Not rerun in this review. Current verification loaded the model and passed tests, but did not regenerate the dataset or retrain.

Documented:
Yes in `README.md`.

Risk / weakness:
This is the highest overclaim risk. It is a simulated-data result, and this review did not reproduce it.

Interview-safe wording:
The README reports a 60.8% to 21.7% false-positive-rate reduction on the simulated dataset. I would use that only with the simulation caveat and note that the next evidence step is a metrics reproduction log.

Smallest strengthening step:
Run the simulation/training flow once, save the command output and assumptions in a metrics reproduction doc, and link it from this evidence map.

## Claim 11

Claim:
The project has meaningful automated test coverage.

Support level:
Strongly supported as a qualitative claim. Do not claim a coverage percentage until the coverage command is rerun and recorded.

Evidence in repo:
- `tests/` contains pytest unit, integration, and ML tests.
- `tests/conftest.py:109-114` loads the real model artifact for tests; `tests/conftest.py:121-160` supplies a mocked DB service.
- `dashboard/src/__tests__/` contains Vitest and React Testing Library tests.
- `.github/workflows/ci.yml:24-28` runs Python tests and coverage in CI.
- `.github/workflows/ci.yml:47-51` runs frontend tests and lint in CI.
- `requirements-test.txt:1-3` and `dashboard/vitest.config.ts:4-10` configure test dependencies/tools.

Implemented:
Yes.

Tested:
Yes. Recent local runs: 243 Python tests passed, and the current frontend run passed 62 tests.

Manually verified:
Current commands passed:
- `python3 -m pytest tests/ -q --tb=short`
- `npm test -- --run --reporter=dot`
- `npm run lint`
- `npm run build`

Documented:
Partially. `README.md`, `AGENTS.md`, and this doc set mention test commands.

Risk / weakness:
The Python coverage command was not rerun in this review. The automated test suite is strong for local behavior but still relies on mocks for most DB behavior. Live Docker/DB confidence is based on one manual smoke run, not automation. Metric reproduction remains open.

Interview-safe wording:
The repo has automated tests across API validation, feature engineering, model artifacts, mocked DB service behavior, real-model prediction behavior, and dashboard components/services. The main local test suites pass.

Smallest strengthening step:
Run and record `python3 -m pytest tests/ --cov=api --cov=fleet_data --cov-report=term-missing`, plus an automated version of the recorded Docker smoke path.

## Review Summary

### Claims safe to use in interviews today

- Claim 1, with simulated/demo caveat.
- Claim 3, API endpoints, with note that DB is mocked in route tests.
- Claim 4, Pydantic request validation.
- Claim 5, optional API-key auth for `POST /api/predict`.
- Claim 6, DB persistence path, only as a manually smoke-tested local Docker path rather than comprehensive DB integration testing.
- Claim 7, React/TypeScript dashboard with monitoring and simulation UI.
- Claim 11, automated test suite exists and currently passes, without claiming a coverage percentage.

### Claims that require tests first

- Claim 2, if phrased as fully shared training/inference feature ordering. Refactor `ml/prepare_data.py` to import `FEATURE_COLUMNS` and test it.
- Claim 6, if phrased as comprehensive automated DB integration. Add a repeatable Docker/Postgres smoke script or CI-style check.
- Claim 8, if using static demo mode as a tested deployment path. Add explicit `VITE_DEMO_MODE=true` tests and/or a manual demo-mode build note.
- Claim 9, if using "end-to-end pipeline" to mean freshly rerunnable. Add a reproduction log.
- Claim 10, if using the exact 64% metric as more than a README-reported simulated result. Reproduce and archive the metric.

### Claims that should be avoided

- Production-grade platform.
- Real fleet deployment or real operational alert reduction.
- Comprehensive security.
- Fully tested live TimescaleDB integration.
- Static demo mode as evidence of XGBoost model accuracy.
- A guaranteed 64% false-positive reduction outside the repo's simulated dataset.

### Claims that belong only in future work

- Production migration strategy.
- Live external fleet ingestion.
- Comprehensive auth/authorization.
- Real deployment evidence.
- Automated metric-regression gate for the training pipeline.
- Exhaustive UI-level error-handling robustness for all route-specific non-2xx API responses.
