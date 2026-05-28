# Sentinel QA Checklist

Use this before an interview demo or resume conversation.

## Repo State

- [ ] Confirm current branch and dirty worktree:

```bash
git status --short --branch
```

- [ ] Confirm no unrelated generated artifacts are being presented as intentional work.
- [ ] Confirm `docs/interview-readiness/` reflects the current repo state.

## Fast Automated Verification

- [ ] Run Python tests:

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected current baseline:

```text
243 passed
```

- [ ] Run frontend tests:

```bash
cd dashboard
npm test -- --run --reporter=dot
```

Expected current baseline:

```text
8 test files passed
62 tests passed
```

- [ ] Run frontend lint:

```bash
cd dashboard
npm run lint
```

Expected current baseline:

```text
No lint errors
```

- [ ] Run frontend build:

```bash
cd dashboard
npm run build
```

Expected current baseline:

```text
Build passes
Known warning: main JS chunk is about 905.83 kB
```

## Full-Stack Manual Verification

Current recorded smoke evidence:

- `docs/interview-readiness/SENTINEL_DOCKER_SMOKE_TEST.md`
- Last recorded result: local Docker smoke pass on 2026-05-27 PDT, branch `main`, commit `66efd64`.
- Rerun this before a live demo because the recorded smoke is manual evidence, not a CI gate.

Run:

```bash
SMOKE_PROJECT=sentinel_smoke_$(date +%Y%m%d_%H%M%S)
docker-compose -p "$SMOKE_PROJECT" up --build
```

Check:

- [ ] API is reachable at `http://localhost:8000/health`.
- [ ] Health response reports `model_loaded: true`.
- [ ] Health response reports `db_connected: true`.
- [ ] Swagger docs load at `http://localhost:8000/docs`.
- [ ] Dashboard loads at `http://localhost:3000`.
- [ ] Dashboard overview cards render non-empty values after seeding.
- [ ] Recent alerts table renders prediction records.
- [ ] Model health panel renders status, confidence distribution, and prediction split.
- [ ] FP rate trend chart renders buckets.
- [ ] Simulate panel can submit a notification and show a prediction.
- [ ] Stop only the isolated smoke project when finished:

```bash
docker-compose -p "$SMOKE_PROJECT" down -v
```

## API Contract Spot Checks

Health:

```bash
curl http://localhost:8000/health
```

Stats:

```bash
curl "http://localhost:8000/api/stats?hours=24"
```

Prediction:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "qa_vehicle_001",
    "speed": 0,
    "expected_speed": 35,
    "road_type": "downtown",
    "traffic_condition": "heavy",
    "construction_zone": "none",
    "notification_type": "stuck",
    "notification_subtype": null,
    "ev_distance": null,
    "pedestrian_density": 0.3,
    "object_in_path": false,
    "time_since_stop": 120,
    "hour_of_day": 14
  }'
```

Expected:

- 200 response when `API_KEY` is empty.
- JSON includes `needs_intervention`, `confidence`, `raw_score`, and `timestamp`.

Auth check when `API_KEY` is set:

- [ ] Missing or wrong `X-API-Key` returns 401.
- [ ] Correct `X-API-Key` returns 200.

## Static Demo Mode Verification

Build or run dashboard with:

```bash
cd dashboard
VITE_DEMO_MODE=true npm run build
```

Check:

- [ ] Dashboard can render from `dashboard/src/data/*.json`.
- [ ] Demo banner appears.
- [ ] Simulate panel returns heuristic predictions without API/DB.
- [ ] Do not describe demo-mode predictions as XGBoost inference.

## Claim Hygiene

Before an interview, confirm you can say each of these accurately:

- [ ] "Simulated AV fleet notifications", not real production fleet data.
- [ ] "Portfolio-scale full-stack ML system", not production-grade platform.
- [ ] "Optional API-key auth for prediction writes", not comprehensive security.
- [ ] "Tests pass locally", only if you ran them recently.
- [ ] "Documented 64% reduction on the simulated dataset", not real-world measured savings.
- [ ] "Static demo mode uses generated JSON and a heuristic predictor", not the live backend.

## Known Issues To Be Honest About

- [ ] UI-level error-state coverage is not exhaustive across every route.
- [ ] Full training metric reproduction has not been captured in this first pass.
- [ ] Docker full-stack smoke verification is recorded once, but should be rerun before a live demo and is not automated in CI.
- [ ] Dashboard build has a chunk-size warning.
