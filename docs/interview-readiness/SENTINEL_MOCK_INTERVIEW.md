# Sentinel Mock Interview

## Opening

Interviewer:
What is Sentinel?

Safe answer:
Sentinel is a full-stack ML systems project for simulated autonomous-vehicle fleet alert triage. It takes simulated fleet notifications, engineers contextual features, runs an XGBoost classifier, serves predictions through FastAPI, stores predictions in Postgres/TimescaleDB, and renders monitoring views in a React/TypeScript dashboard.

Pressure question:
Was this used in a real fleet?

Safe answer:
No. The repo should be framed as a simulated portfolio project inspired by AV operations problems, not as a deployed production fleet system.

Admit uncertainty:
I would not claim real-world performance without real operational data and a deployed evaluation loop.

## Architecture

Interviewer:
Walk me through the data flow.

Safe answer:
The simulation creates vehicle telemetry and notification context. The ML preparation step loads those records, creates 28 features, scales them, and trains an XGBoost classifier. The API loads committed model artifacts and maps incoming notification payloads to the same feature vector. Predictions are stored in a `predictions` table and the dashboard reads recent alerts, stats, FP trend, and model health.

Pressure question:
How do you know training and inference use the same features?

Safe answer:
The repo has `ml/constants.py` for shared maps and feature columns, and tests verify the saved model config matches the expected 28-feature order. Feature engineering tests also check individual inference features.

Admit uncertainty:
One maintainability improvement is to make `ml/prepare_data.py` import `FEATURE_COLUMNS` directly instead of keeping a local list, even though parity tests currently guard the saved config.

## Model

Interviewer:
Why XGBoost?

Safe answer:
This is tabular, categorical-and-context-heavy data. XGBoost is a good fit because it handles non-linear interactions well, trains quickly, and is explainable enough for feature importance discussion. The repo also documents that an earlier VAE anomaly-detection approach was removed after it failed to separate false positives from real interventions.

Pressure question:
Can you defend the 64% false-positive reduction?

Safe answer:
Only with the right caveat. The README reports a reduction from 60.8% to 21.7% false-positive rate on the simulated dataset. I would present that as simulation evidence, not production evidence. A next step is to capture a metrics reproduction log with exact commands and outputs.

Admit uncertainty:
I did not rerun the full training pipeline in this first pass, so I would not claim fresh reproduction of that metric yet.

## API

Interviewer:
What does the FastAPI service do?

Safe answer:
It exposes prediction and dashboard-support endpoints. `POST /api/predict` validates a notification payload, runs the model service, stores the result, and returns a prediction. There are read endpoints for alerts, aggregate stats, model health, false-positive trend, per-type stats, and `/health`.

Pressure question:
Is the API secure?

Safe answer:
It has optional API-key auth on prediction requests. If `API_KEY` is empty, auth is skipped for local/demo use. I would not call this comprehensive production security because read endpoints are public and there is no user model or authorization layer.

Admit uncertainty:
For production, I would define route-level security requirements, hide internal error details, and add stronger auth/authorization.

## Database

Interviewer:
Why TimescaleDB?

Safe answer:
The simulated training data is time-series vehicle telemetry, so TimescaleDB is a reasonable Postgres extension choice. The API prediction store uses normal SQL queries over prediction timestamps and notification types. A local Docker smoke run verified DB connection, seeding, prediction insertion, stats, alerts, and dashboard reads against a live Timescale/Postgres container.

Pressure question:
Are migrations production-ready?

Safe answer:
No. Schema setup is script-based. `setup_database.py` drops and recreates `vehicle_metrics`, which is appropriate for demo reset flows but not durable production migrations.

Admit uncertainty:
I would add Alembic or another migration path before treating this as a durable service.

## Frontend

Interviewer:
What does the dashboard show?

Safe answer:
It shows overview stats, recent alerts, type breakdown, FP-rate trend, an interactive simulation form, and model health views like confidence buckets, prediction split, accuracy, and status.

Pressure question:
Does the dashboard handle API errors well?

Safe answer:
Partially. Components have some loading and retry behavior, and tests cover key rendering paths. The shared API client now checks `res.ok` and has tests for representative 401, 422, and 500 responses, but I would not claim exhaustive UI-level failure coverage for every route.

Admit uncertainty:
I would add broader component-level error-state tests before claiming robust frontend error handling.

## Testing

Interviewer:
What tests exist?

Safe answer:
The backend has pytest coverage for Pydantic models, feature engineering, model artifacts, DB service behavior with mocks, fleet simulation logic, and API endpoints with a real model plus mocked DB. The frontend has Vitest/RTL tests for services and core dashboard components. There is also one recorded local Docker smoke run for the full-stack DB/API/model/seed/dashboard path.

Pressure question:
What passed most recently?

Safe answer:
In this pass, `python3 -m pytest tests/ -q` passed 243 tests, `npm test -- --run --reporter=dot` passed 62 frontend tests, `npm run lint` passed, and `npm run build` passed with a known chunk-size warning. The Docker smoke run also passed locally after two Dockerfile fixes, with `/health`, seeded stats, prediction insert, alerts persistence, nginx proxying, and rendered dashboard data verified.

Admit uncertainty:
The automated tests still do not comprehensively prove DB integration because most DB assertions use mocks. The Docker evidence is one local manual smoke run, not CI automation. I still need a metrics reproduction artifact for the headline false-positive reduction.

## AI-Assisted Development

Interviewer:
How much of this was AI-generated?

Safe answer:
Some code may have been built with heavy AI assistance. I do not frame that as a negative or hide it. My ownership standard is whether I can inspect the repo, explain behavior, run tests, find unsupported claims, fix edge cases, and document the system responsibly.

Pressure question:
So how do I know you understand it?

Safe answer:
I can walk through the feature pipeline, API contract, model-serving path, DB queries, dashboard data flow, and test strategy. I can also point to current weaknesses, like metric reproduction and broader UI-level failure coverage.

Admit uncertainty:
If I have not personally rerun a path, I will say so and describe the exact evidence-building step.

## Debugging Scenario

Interviewer:
The model starts making strange predictions after a feature change. What do you check?

Safe answer:
First I would check feature order and categorical encoding parity. Then I would compare a representative payload through `ModelService.engineer_features()` and the training feature logic. I would run the feature engineering tests, encoding parity tests, and model service tests. If the issue is data-specific, I would add a regression payload with expected feature values.

Pressure question:
What kind of bug could silently slip through?

Safe answer:
A feature column reorder or truthiness bug could be silent. One example found in this pass was `ev_distance or 999`, which treated `0` as missing; it is now covered by a regression test and fixed with explicit `None` handling.

## Ownership Close

Interviewer:
What would you work on next?

Safe answer:
I would first create a metric reproduction artifact, automate the Docker Compose smoke test, add static demo-mode service-switch coverage, broaden UI-level error-state tests, and clarify security/demo boundaries in docs.
