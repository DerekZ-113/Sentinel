# Sentinel Interview-Readiness Backlog

Ranked by interview value, not cosmetic polish.

## P0: Evidence Needed Before Using A Claim Confidently

### P0-1: Reproduce and archive model metrics

Why this matters in interviews:
The headline 64% false-positive reduction is the most impressive claim and the easiest to challenge. It needs a reproducible evidence note.

Files likely involved:
- `fleet_data/generate_fleet_data.py`
- `fleet_data/baseline_alerter.py`
- `ml/prepare_data.py`
- `ml/train_classifier.py`
- `ml/run_pipeline.py`
- `README.md`
- `docs/interview-readiness/SENTINEL_EVIDENCE_MAP.md`

Suggested test or doc artifact:
- Add `docs/interview-readiness/SENTINEL_METRICS_REPRODUCTION.md` with commands, dataset assumptions, sample sizes, seed behavior, output metrics, and caveats.

Risk of overclaiming:
High. Without this, the README metrics are documented but not freshly reproducible in the interview-readiness pass.

Estimated effort:
Medium.

### P0-2: Fixed - Document full-stack manual demo verification

Why this matters in interviews:
If you say the system runs end to end, you should be able to point to a current checklist proving Docker Compose, API, DB, seed, and dashboard behavior.

Files involved:
- `docker-compose.yml`
- `entrypoint.sh`
- `setup_database.py`
- `scripts/seed_demo.py`
- `api/main.py`
- `dashboard/src/App.tsx`
- `Dockerfile.api`
- `Dockerfile.dashboard`
- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md`

Evidence added:
- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md` records a local Docker smoke run across DB startup, API health, model load, seeding, stats, prediction insert, alerts persistence, nginx proxying, and rendered dashboard data.
- `Dockerfile.api` now copies `ml/__init__.py` and `ml/constants.py` so the container can import shared ML constants.
- `Dockerfile.dashboard` now forces `VITE_DEMO_MODE=false` for Docker Compose builds so the dashboard uses live API data instead of static demo JSON.

Risk of overclaiming:
Lower for the local Docker path, because one smoke pass is recorded. Still medium if described as production deployment, comprehensive DB integration testing, or automated CI evidence.

Verification:
`docker-compose -p sentinel_smoke_20260527_183951 up --build` passed after the Dockerfile fixes. `/health`, `/docs`, `/api/stats`, `POST /api/predict`, `/api/alerts`, dashboard HTTP, nginx `/api/stats`, and rendered dashboard DOM checks passed. Cleanup with `docker-compose -p sentinel_smoke_20260527_183951 down -v` removed the isolated volume.

### P0-3: Fixed - EV distance zero regression

Why this matters in interviews:
`ev_distance=0` is a valid value. The model-serving path now handles it explicitly instead of treating it as missing.

Files involved:
- `api/services/model_service.py`
- `tests/unit/test_feature_engineering.py`
- `docs/interview-readiness/SENTINEL_EVIDENCE_MAP.md`

Evidence added:
- `tests/unit/test_feature_engineering.py::TestContextFeatures::test_ev_distance_zero_normalized` asserts `ev_distance=0` normalizes to `0.0`.
- `ModelService.engineer_features()` uses `ev_distance if ev_distance is not None else 999`.

Risk of overclaiming:
Low for this specific inference edge case. This does not address broader training/inference feature-list duplication.

Verification:
`python3 -m pytest tests/ -q` passed with `243 passed in 1.57s`.

## P1: High Interview Value, Targeted Work

### P1-1: Refactor training feature columns to import `FEATURE_COLUMNS`

Why this matters in interviews:
It strengthens the story that training and inference share a single source of truth.

Files likely involved:
- `ml/prepare_data.py`
- `tests/ml/test_encoding_parity.py`
- `tests/ml/test_prepare_data.py`

Suggested test or doc artifact:
- Use `FEATURE_COLUMNS` directly in `prepare_training_data()`.
- Keep or add a test that fails if training uses a divergent list.

Risk of overclaiming:
Medium. Current constants and config are tested, but the local list is still a maintainability smell.

Estimated effort:
Small.

### P1-2: Partially strengthened - API client non-2xx handling

Why this matters in interviews:
Frontend reliability questions often probe unhappy paths. The shared frontend API client now rejects tested non-2xx responses with status-aware errors.

Files involved:
- `dashboard/src/services/api.ts`
- `dashboard/src/__tests__/api.test.ts`

Evidence added:
- `postPredict()` rejects 401 responses with status and auth detail.
- `postPredict()` rejects 422 responses with formatted FastAPI validation detail.
- `fetchStats()` rejects 500 responses with status and backend detail.
- Live API helpers route through a shared `fetchJson()` helper.

Risk of overclaiming:
Low for the tested API helper behavior. Medium for broader UI-level failure coverage across every route and component.

Verification:
`npm test -- --run --reporter=dot` passed with 8 files and 62 tests. `npm run lint` and `npm run build` passed; build retains the known 905.83 kB chunk warning.

### P1-3: Document frontend/backend API contract with examples

Why this matters in interviews:
It makes the project easier to explain and shows you understand the boundary between UI and API.

Files likely involved:
- `api/models.py`
- `api/routes/*.py`
- `dashboard/src/services/api.ts`
- `docs/interview-readiness/SENTINEL_FRONTEND_BACKEND_CONTRACT.md`

Suggested test or doc artifact:
- Keep the contract doc updated with payload/response examples and route protection notes.

Risk of overclaiming:
Low to medium.

Estimated effort:
Small.

### P1-4: Add static demo mode tests

Why this matters in interviews:
Demo mode is useful for presentations, but it should be clearly verified separately from full-stack mode.

Files likely involved:
- `dashboard/src/services/api.ts`
- `dashboard/src/services/demoPredict.ts`
- `dashboard/src/__tests__/api.test.ts`
- `dashboard/src/__tests__/demoPredict.test.ts`

Suggested test or doc artifact:
- Add tests that force `VITE_DEMO_MODE=true` and verify `fetch` is not called for static data.

Risk of overclaiming:
Medium if demo mode is used in interviews without explaining that it is static/heuristic.

Estimated effort:
Small.

### P1-5: Add security boundary note

Why this matters in interviews:
Security questions are predictable. A concise boundary note prevents accidental overclaiming.

Files likely involved:
- `api/auth.py`
- `.env.example`
- `README.md`
- `docs/interview-readiness/SENTINEL_TESTING_STRATEGY.md`

Suggested test or doc artifact:
- Add a doc section listing protected/unprotected routes and local/demo assumptions.

Risk of overclaiming:
Medium.

Estimated effort:
Small.

## P2: Nice-To-Have Hardening

### P2-1: Add automated Docker-backed smoke script

Why this matters in interviews:
It proves integration beyond mocks without requiring a large test suite.

Files likely involved:
- `docker-compose.yml`
- `entrypoint.sh`
- `scripts/`
- `docs/interview-readiness/SENTINEL_QA_CHECKLIST.md`

Suggested test or doc artifact:
- A simple smoke script that starts an isolated Compose project, runs `/health`, `/api/predict`, `/api/stats`, `/api/alerts`, and dashboard load checks, then tears down only that project.

Risk of overclaiming:
Medium.

Estimated effort:
Medium.

### P2-2: Reduce dashboard production bundle size

Why this matters in interviews:
The build passes but warns that the main JS chunk is 905.83 kB. This is a concrete frontend performance improvement.

Files likely involved:
- `dashboard/vite.config.ts`
- `dashboard/src/App.tsx`
- chart-heavy components using `recharts`

Suggested test or doc artifact:
- Add code splitting around chart-heavy sections.
- Record before/after build output.

Risk of overclaiming:
Low. This is performance polish, not a correctness blocker.

Estimated effort:
Small to medium.

### P2-3: Replace raw exception details in API 500 responses

Why this matters in interviews:
It shows you can distinguish demo debugging convenience from production security hygiene.

Files likely involved:
- `api/main.py`
- `api/routes/predict.py`
- `api/routes/alerts.py`
- `api/routes/stats.py`
- `tests/integration/`

Suggested test or doc artifact:
- Add tests that 500s return stable public messages while logs keep exception details.

Risk of overclaiming:
Medium if calling the API production-ready.

Estimated effort:
Small.

### P2-4: Clarify schema setup vs migrations

Why this matters in interviews:
`setup_database.py` drops `vehicle_metrics`, which is fine for a demo reset but not durable operations.

Files likely involved:
- `setup_database.py`
- `entrypoint.sh`
- `README.md`

Suggested test or doc artifact:
- Add a doc note explaining demo reset behavior and future migration plan.

Risk of overclaiming:
Medium.

Estimated effort:
Small.

### P2-5: Run and record coverage

Why this matters in interviews:
Coverage is less important than test quality, but a coverage snapshot helps quantify the existing suite.

Files likely involved:
- `requirements-test.txt`
- `.github/workflows/ci.yml`
- `docs/interview-readiness/SENTINEL_TESTING_STRATEGY.md`

Suggested test or doc artifact:
- Add a coverage note with command output and interpretation.

Risk of overclaiming:
Low.

Estimated effort:
Small.

## P3: Future Polish

### P3-1: Replace default Vite README

Why this matters in interviews:
`dashboard/README.md` is still template text. This does not block the project, but it weakens polish.

Files likely involved:
- `dashboard/README.md`

Suggested test or doc artifact:
- Write dashboard-specific setup, demo mode, and testing notes.

Risk of overclaiming:
Low.

Estimated effort:
Small.

### P3-2: Add architecture diagram image or Mermaid diagram

Why this matters in interviews:
Useful for presentation, but current text architecture is enough for a first pass.

Files likely involved:
- `README.md`
- `docs/interview-readiness/SENTINEL_ARCHITECTURE_OVERVIEW.md`

Suggested test or doc artifact:
- Add one Mermaid system diagram.

Risk of overclaiming:
Low.

Estimated effort:
Small.

### P3-3: Add route-level OpenAPI examples

Why this matters in interviews:
Examples improve Swagger usefulness but do not change core evidence.

Files likely involved:
- `api/models.py`
- `api/routes/*.py`

Suggested test or doc artifact:
- Add Pydantic examples and verify `/docs` manually.

Risk of overclaiming:
Low.

Estimated effort:
Small.
