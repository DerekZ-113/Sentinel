# Sentinel Docker Smoke Test

## Summary

Final result: pass, with a headless-Chrome tooling caveat described below.

This was a local Docker Compose smoke verification across TimescaleDB/Postgres, FastAPI startup, model loading, demo seeding, prediction insertion, nginx dashboard serving, nginx API proxying, and dashboard rendering against live API data.

This does not prove production deployment, production security, comprehensive DB integration coverage, or CI automation.

## Date And Repo State

- Date: 2026-05-27 PDT. Container logs are in UTC and show 2026-05-28.
- Branch: `main`
- Commit: `66efd64`
- Worktree: dirty. The repo contained uncommitted interview-readiness pass changes before and during this smoke run.
- Final isolated Compose project: `sentinel_smoke_20260527_183951`

## Environment Assumptions

- Docker Desktop was started locally with `docker desktop start`.
- Docker CLI: `29.0.1`
- Docker Compose: `v2.40.3-desktop.1`
- Local ports used: `5432`, `8000`, `3000`
- `API_KEY` was empty, matching local-demo behavior in `.env.example`.
- `curl`, `jq`, `rg`, and installed Chrome were available.
- Codex sandboxed localhost requests returned `Operation not permitted`, so localhost smoke probes were run with escalated local execution.

## Startup Bugs Found And Fixed

The smoke run exposed two Docker-specific startup/config bugs:

1. API image missing shared ML constants
   - Failing project: `sentinel_smoke_20260527_1820`
   - Command: `docker-compose -p sentinel_smoke_20260527_1820 up --build`
   - Failure: `ModuleNotFoundError: No module named 'ml.constants'`
   - Fix: `Dockerfile.api` now copies `ml/__init__.py` and `ml/constants.py` into the image before model artifacts.

2. Docker dashboard built in static demo mode
   - Project: `sentinel_smoke_20260527_1833`
   - Symptom: dashboard DOM showed the demo banner and static `1,000` alert total instead of the live `1,001` total after the smoke prediction.
   - Cause: `dashboard/.env.production` sets `VITE_DEMO_MODE=true` and was copied into the Docker build.
   - Fix: `Dockerfile.dashboard` now sets `VITE_DEMO_MODE=false` for Docker Compose builds. Static demo mode remains available through explicit `VITE_DEMO_MODE=true` builds.

## Commands And Observed Results

### Build And Start

Command:

```bash
SMOKE_PROJECT=sentinel_smoke_20260527_183951
docker-compose -p "$SMOKE_PROJECT" up --build
```

Observed:

- DB, API, and dashboard containers started.
- `docker-compose.yml` emitted the existing warning that the `version` attribute is obsolete.
- Dashboard build passed with the known Vite chunk warning. Final JS chunk in Docker build was `614.43 kB`, above the 500 kB warning threshold.
- TimescaleDB init emitted local-container warnings about missing locales, local trust auth, and Timescale background workers during initialization/shutdown tuning.

Final service status before cleanup:

```text
api         Up 2 minutes             0.0.0.0:8000->8000/tcp
dashboard   Up 2 minutes             0.0.0.0:3000->3000/tcp
db          Up 2 minutes (healthy)   0.0.0.0:5432->5432/tcp
```

### API Health

Command:

```bash
curl -fsS http://localhost:8000/health | tee /tmp/sentinel-health.json | jq
jq -e '.model_loaded == true and .db_connected == true and .model_features == 28' /tmp/sentinel-health.json
```

Observed:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "db_connected": true,
  "model_features": 28,
  "model_threshold": 0.5,
  "uptime_seconds": 19.7
}
```

Result: pass. The `jq -e` check returned `true`.

### API Docs

Command:

```bash
curl -fsS -o /tmp/sentinel-docs.html -w "%{http_code}\n" http://localhost:8000/docs
rg "Swagger UI|Sentinel API" /tmp/sentinel-docs.html
```

Observed:

```text
200
<title>Sentinel API - Swagger UI</title>
```

Result: pass.

### Seeded Stats

Command:

```bash
curl -fsS "http://localhost:8000/api/stats?hours=24" \
  | tee /tmp/sentinel-stats.json \
  | jq '{time_window_hours,total_alerts,total_flagged,total_suppressed,types: [.by_type[].notification_type]}'
jq -e '.total_alerts > 0' /tmp/sentinel-stats.json
```

Observed:

```json
{
  "time_window_hours": 24,
  "total_alerts": 1000,
  "total_flagged": 230,
  "total_suppressed": 770,
  "types": [
    "verification_request",
    "stuck",
    "speed_anomaly",
    "passenger_assist",
    "emergency_vehicle_alert",
    "impact_l0"
  ]
}
```

Result: pass. The `jq -e` check returned `true`.

### Prediction Insert

Command:

```bash
curl -fsS -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id":"smoke_vehicle_001","speed":0,"expected_speed":35,"road_type":"main_road","traffic_condition":"heavy","construction_zone":"none","notification_type":"stuck","notification_subtype":null,"ev_distance":null,"pedestrian_density":0.3,"object_in_path":false,"time_since_stop":120,"hour_of_day":14,"needs_intervention_actual":true}' \
  | tee /tmp/sentinel-predict.json \
  | jq '{vehicle_id,notification_type,needs_intervention,confidence,raw_score,timestamp}'
```

Observed:

```json
{
  "vehicle_id": "smoke_vehicle_001",
  "notification_type": "stuck",
  "needs_intervention": false,
  "confidence": 0.730031430721283,
  "raw_score": 0.26996856927871704,
  "timestamp": "2026-05-28T01:40:55.905967Z"
}
```

Result: pass.

### Alerts Persistence

Command:

```bash
curl -fsS "http://localhost:8000/api/alerts?limit=5" \
  | tee /tmp/sentinel-alerts.json \
  | jq '{total,vehicles: [.alerts[].vehicle_id],types: [.alerts[].notification_type]}'
jq -e '[.alerts[].vehicle_id] | index("smoke_vehicle_001") != null' /tmp/sentinel-alerts.json
```

Observed:

```json
{
  "total": 1001,
  "vehicles": [
    "smoke_vehicle_001",
    "vehicle_012",
    "vehicle_009",
    "vehicle_006",
    "vehicle_045"
  ],
  "types": [
    "stuck",
    "speed_anomaly",
    "speed_anomaly",
    "stuck",
    "stuck"
  ]
}
```

Result: pass. The `jq -e` check returned `true`.

### Dashboard And Nginx API Proxy

Command:

```bash
curl -fsS -o /tmp/sentinel-dashboard.html -w "%{http_code}\n" http://localhost:3000/
curl -fsS "http://localhost:3000/api/stats?hours=24" | jq -e '.total_alerts > 0'
```

Observed:

```text
200
true
```

Result: pass.

### Dashboard Rendered DOM

Command attempted with installed Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --disable-gpu \
  --disable-background-networking \
  --disable-component-update \
  --disable-sync \
  --disable-extensions \
  --metrics-recording-only \
  --no-first-run \
  --no-default-browser-check \
  --user-data-dir=/tmp/sentinel-chrome-smoke-live \
  --virtual-time-budget=8000 \
  --dump-dom http://localhost:3000 \
  > /tmp/sentinel-dashboard-dom.html
```

Observed:

- Chrome wrote `/tmp/sentinel-dashboard-dom.html`.
- The Chrome process did not exit cleanly in this macOS sandbox and was killed after DOM capture.
- Nginx logs showed Chrome loaded `/`, JS/CSS assets, `/health`, `/api/stats`, `/api/alerts`, `/api/stats/model-health`, and `/api/stats/fp-over-time`.

DOM checks:

```bash
rg -o "Overview|Recent Alerts|Model Health|Simulate Notification" /tmp/sentinel-dashboard-dom.html | sort -u
rg -o "smoke_vehicle_001|1,001" /tmp/sentinel-dashboard-dom.html | sort -u
if rg -q "Demo mode" /tmp/sentinel-dashboard-dom.html; then echo "Demo mode banner found"; exit 1; else echo "Demo mode banner not found"; fi
```

Observed:

```text
Model Health
Overview
Recent Alerts
Simulate Notification
1,001
smoke_vehicle_001
Demo mode banner not found
```

Result: pass for rendered DOM content and live dashboard data. Tooling caveat: raw Chrome process cleanup was manual after DOM output.

### API Logs

Command:

```bash
docker-compose -p sentinel_smoke_20260527_183951 logs api \
  | rg "Database ready|Setting up database schema|Seeding demo data|Seed complete|Sentinel API ready|Model loaded|No scaler found"
```

Observed:

```text
Database ready
Setting up database schema...
No scaler found - using reconstructed fallback
Model loaded: 28 features, threshold=0.5, best_iter=499
Sentinel API ready
Seeding demo data...
Seed complete! Sent: 1000  |  Failed: 0  |  Time: 2.8s
Sentinel API ready on port 8000
```

Result: pass. Model loading, DB setup, API startup, and seed completion were all visible in logs.

### Cleanup

Command:

```bash
docker-compose -p sentinel_smoke_20260527_183951 down -v
```

Observed:

```text
Volume sentinel_smoke_20260527_183951_sentinel_data  Removed
Network sentinel_smoke_20260527_183951_default  Removed
```

Post-cleanup check:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}' | rg "sentinel_smoke_20260527_183951|NAMES"
```

Observed:

```text
NAMES     STATUS
```

Result: pass. No final smoke project containers remained running.

## Final Assessment

Pass.

The local Docker full-stack path was verified after two small Dockerfile fixes. Evidence covers:

- TimescaleDB container startup and healthy status.
- API startup, DB connection, model load, and fallback scaler warning.
- Seed script inserting 1000 predictions through `/api/predict`.
- `/health`, `/docs`, `/api/stats`, `/api/predict`, and `/api/alerts`.
- Nginx serving the dashboard and proxying `/api/stats`.
- Dashboard rendering live API-backed data including `1,001` total alerts and `smoke_vehicle_001`.

## Known Limitations

- This is one local manual smoke run, not an automated CI gate.
- It does not prove production deployment or production security.
- It does not comprehensively test every DB query or migration behavior.
- DB unit tests remain mostly mocked.
- `setup_database.py` remains demo-oriented and resets `vehicle_metrics`.
- The dashboard build still emits a chunk-size warning.
- Headless Chrome wrote the DOM successfully but required manual process cleanup in this local macOS sandbox.
- The 64% false-positive reduction was not reproduced in this pass.
