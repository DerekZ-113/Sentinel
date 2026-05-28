# Sentinel Frontend / Backend Contract

This document maps the current React dashboard contract to FastAPI routes and Pydantic response models.

## Contract Summary

Frontend API client:

- `dashboard/src/services/api.ts`

Backend schemas:

- `api/models.py`

Backend routes:

- `api/routes/predict.py`
- `api/routes/alerts.py`
- `api/routes/stats.py`
- `api/main.py`

## Route Map

| Frontend function | Backend route | Backend model | Auth | Main consumer |
|---|---|---|---|---|
| `fetchHealth()` | `GET /health` | `HealthResponse` | None | `App.tsx` sidebar/status |
| `fetchStats(hours)` | `GET /api/stats?hours=...` | `StatsResponse` | None | `App.tsx`, `OverviewCards`, `TypeBreakdown` |
| `fetchAlerts(limit, offset, notificationType)` | `GET /api/alerts` | `AlertsResponse` | None | `AlertFeed` |
| `fetchModelHealth(hours)` | `GET /api/stats/model-health` | `ModelHealthResponse` | None | `ModelHealth` |
| `fetchFPOverTime(hours, buckets)` | `GET /api/stats/fp-over-time` | `FPOverTimeResponse` | None | `FPRateChart` |
| `postPredict(payload)` | `POST /api/predict` | `PredictionResponse` | Optional `X-API-Key` | `SimulatePanel` |

## Prediction Request

Frontend type:

```ts
export interface NotificationPayload {
  vehicle_id: string;
  speed: number;
  expected_speed: number;
  road_type: string;
  traffic_condition: string;
  construction_zone: string;
  notification_type: string;
  notification_subtype?: string | null;
  ev_distance?: number | null;
  pedestrian_density: number;
  object_in_path: boolean;
  time_since_stop: number;
  hour_of_day?: number;
}
```

Backend validation:

- `speed >= 0`
- `expected_speed >= 0`
- `road_type` must be one of `highway`, `main_road`, `residential`, `downtown`, `school_zone`
- `traffic_condition` must be one of `light`, `moderate`, `heavy`, `standstill`
- `construction_zone` must be one of `none`, `temporary`, `persistent`, `flagger`
- `notification_type` must be one of `verification_request`, `emergency_vehicle_alert`, `stuck`, `speed_anomaly`, `impact_l0`, `passenger_assist`
- `notification_subtype`, when present, must be one of `object_query`, `traffic_signal_verify`, `lane_mapping_verify`
- `ev_distance >= 0` when present
- `0 <= pedestrian_density <= 1`
- `time_since_stop >= 0`
- `0 <= hour_of_day <= 23` when present

Example request:

```json
{
  "vehicle_id": "sim_1234",
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
}
```

Example response shape:

```json
{
  "vehicle_id": "sim_1234",
  "notification_type": "stuck",
  "needs_intervention": false,
  "confidence": 0.87,
  "raw_score": 0.13,
  "timestamp": "2026-05-27T00:00:00Z"
}
```

## Demo Mode Contract

When `VITE_DEMO_MODE=true`, `dashboard/src/services/api.ts` does not call the backend for dashboard data. It returns:

- `dashboard/src/data/alerts.json`
- `dashboard/src/data/stats.json`
- `dashboard/src/data/model-health.json`
- `dashboard/src/data/health.json`
- `dashboard/src/data/fp-over-time.json`

For `postPredict()`, demo mode calls `dashboard/src/services/demoPredict.ts`.

Important caveat:

`demoPredict.ts` is a heuristic mirror for presentation. It is not XGBoost inference and should not be used as model-performance evidence.

## Known Contract Gaps

- Frontend API helpers now check `res.ok` and selected tests cover 401, 422, and 500 responses.
- Broader UI-level error rendering is not exhaustively tested for every route.
- README endpoint table should mention `/api/stats/fp-over-time`.
- Route-level auth documentation should clarify that only `POST /api/predict` uses API-key auth.

## Interview-Safe Contract Claim

> The frontend/backend contract is explicit enough for a portfolio system: Pydantic schemas define backend payloads, TypeScript interfaces mirror response shapes on the client, and tests cover key API client calls, selected non-2xx frontend handling, and FastAPI validation. The next hardening step is broader UI-level error coverage.
