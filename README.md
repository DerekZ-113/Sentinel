# Sentinel

![Python](https://img.shields.io/badge/Python-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-009688) ![React](https://img.shields.io/badge/React-61DAFB) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6) ![Docker](https://img.shields.io/badge/Docker_Compose-2496ED) ![XGBoost](https://img.shields.io/badge/XGBoost-orange) 

**Context-aware alert filtering for autonomous vehicle fleets**

Sentinel is a full-stack system that reduces operator alert fatigue by learning which fleet notifications actually need human intervention. It combines an XGBoost classifier trained on 28 engineered features with a FastAPI inference service and a React monitoring dashboard, all containerized with Docker Compose.

The model achieves a **64% reduction in false positives** and eliminates roughly **2 million unnecessary alerts daily** on a 500-vehicle fleet simulation.

---

## Screenshots

### Main Dashboard
Overview cards, live alert feed, and per-type breakdown.

![Sentinel Dashboard](Asset/sentinel_main_view.png)

### Model Health Monitoring
Confidence distribution, prediction split, accuracy tracking, and status indicators.

![Model Health](Asset/sentinel_model_health.png)

---

## Quick Start

The entire stack runs with one command. No manual database setup, no separate API launch. The entrypoint script handles schema creation, auto-seeding, and startup.

```bash
# Clone and start
git clone https://github.com/DerekZ-113/Sentinel.git
cd Sentinel
docker-compose up --build

# That's it. Open the dashboard:
# http://localhost:3000
```

On first run, the API container waits for the database, sets up the schema, seeds 1,000 demo notifications through the model, and starts serving. The dashboard pulls from the API and renders immediately.

**API docs (Swagger):** http://localhost:8000/docs

---

## The Problem

AV fleet operators are drowning in notifications:

| Notification Type | What It Means | Reality |
|-------------------|---------------|---------|
| **Object Query** | "Is something blocking me?" | 83% are false positives (someone just walked by) |
| **EV Alert** | "Emergency vehicle detected" | 70% are false positives (EV too far away) |
| **Stuck** | "I can't move" | 61% are false positives (traffic or red light) |

**Baseline: 60.8% of all notifications are false positives.**

When everything is an alert, nothing is an alert. Real issues get buried in noise.

---

## The Solution

Sentinel uses an XGBoost classifier with interaction features built from operational experience to predict which notifications actually need operator intervention.

The key insight is that context matters. A "stuck" notification during rush hour traffic is almost always a false positive. A "stuck" notification on a clear highway probably needs attention. The model learns these contextual patterns through 11 hand-crafted interaction features that encode the operational knowledge behind when an alert is real versus noise.

---

## Results

### Overall Performance

| Metric | Baseline | Sentinel | Improvement |
|--------|----------|----------|-------------|
| False Positive Rate | 60.8% | 21.7% | -64% |
| Precision | 39.2% | 78.3% | +100% |
| Recall | 100% | 86.6% | - |
| F1 Score | - | 82.2% | - |
| ROC-AUC | - | 0.946 | - |

### Per-Notification-Type Breakdown

| Type | Baseline FP | Sentinel FP | Reduction |
|------|-------------|-------------|-----------|
| verification_request/object_query | 82.9% | 0.0% | -100% |
| emergency_vehicle_alert | 70.0% | 0.0% | -100% |
| speed_anomaly | 57.2% | 0.5% | -99% |
| stuck | 61.1% | 38.0% | -38% |
| impact_l0 | 47.5% | 42.7% | -10% |
| verification_request/lane_mapping_verify | 30.7% | 28.9% | -6% |
| verification_request/traffic_signal_verify | 9.6% | 9.4% | -2% |
| passenger_assist | 0.0% | 0.0% | N/A |

### Operator Impact

| Metric | Baseline | Sentinel |
|--------|----------|----------|
| Alerts per day | 3.8M | 1.7M |
| False alarms per day | 2.3M | 360K |
| Workload | 100% | 43% |

**2 million false alarms eliminated daily.**

---

## Architecture

```
docker-compose up
     |
     v
+------------------------------------------------------------+
|                    Docker Compose                          |
|                                                            |
|  +-------------+    +--------------+    +---------------+  |
|  | React/TS    |--->|   FastAPI    |--->|  TimescaleDB  |  |
|  | Dashboard   |    |   Backend    |    |               |  |
|  | :3000       |    |   :8000      |    |  :5432        |  |
|  | (nginx)     |    |              |    |               |  |
|  +-------------+    +------+-------+    +---------------+  |
|                            |                               |
|                     +------+-------+                       |
|                     |   XGBoost    |                       |
|                     |   Model      |                       |
|                     |  (loaded)    |                       |
|                     +--------------+                       |
+------------------------------------------------------------+
```

The nginx layer in the dashboard container proxies `/api/*` requests to FastAPI, so the frontend and backend share the same origin in production. During development, CORS is configured for `localhost:5173` (Vite dev server).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Send a notification payload, get back a prediction with confidence score |
| GET | `/api/alerts` | Recent predictions with optional type filter and pagination |
| GET | `/api/stats` | Aggregate stats over a time window (total, flagged, suppressed, FP rate) |
| GET | `/api/stats/{type}` | Per-notification-type breakdown |
| GET | `/api/stats/model-health` | Model health metrics: accuracy, confidence distribution, prediction split |
| GET | `/health` | Service health check (model loaded, DB connected, uptime) |

All endpoints return JSON. Full schema documentation is available at `/docs` when the API is running.

---

## Dashboard

The React dashboard has five panels:

- **Overview Cards** - Total alerts, flagged count, suppressed count, and model FP rate at a glance.
- **Alert Feed** - Scrollable table of recent notifications showing vehicle ID, type, model prediction, confidence score, and ground truth when available.
- **Type Breakdown** - Horizontal bar chart comparing flagged vs suppressed counts for each notification type.
- **Simulate** - Interactive form where you can input notification parameters and get a real-time prediction from the API. Useful for testing edge cases.
- **Model Health** - Confidence distribution (high/medium/low buckets), prediction split (flagged vs suppressed), accuracy, and a status indicator (healthy/warning/degraded).

---

## Feature Importance

The interaction features built from operational patterns became the top predictors:

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | `object_query_moving` | 21.8% | Interaction |
| 2 | `object_in_path` | 19.1% | Context |
| 3 | `ev_far_away` | 7.0% | Interaction |
| 4 | `ev_close` | 6.7% | Interaction |
| 5 | `object_query_low_ped` | 6.5% | Interaction |
| 6 | `is_stopped` | 5.9% | Context |
| 7 | `ev_distance_normalized` | 5.9% | Context |
| 8 | `notification_subtype_encoded` | 5.8% | Base |
| 9 | `notification_type_encoded` | 5.2% | Base |
| 10 | `object_query_high_ped` | 3.9% | Interaction |

6 of the top 10 features are interaction features - patterns I learned from working operations.

---

## Features (28 total)

### Base Features (17)

| Feature | Description |
|---------|-------------|
| `speed_ratio` | actual_speed / expected_speed |
| `speed_deviation` | actual_speed - expected_speed |
| `is_stopped` | speed < 5 mph |
| `expected_stopped` | expected_speed < 5 mph |
| `road_type_encoded` | highway, main_road, residential, downtown, school_zone |
| `traffic_encoded` | light, moderate, heavy, standstill |
| `construction_encoded` | none, temporary, persistent, flagger |
| `notification_type_encoded` | 6 notification types |
| `notification_subtype_encoded` | 3 subtypes for verification_request |
| `ev_distance_normalized` | distance to emergency vehicle |
| `pedestrian_density` | nearby pedestrian activity (0-1) |
| `object_in_path` | is there actually an obstruction |
| `time_since_stop_normalized` | how long vehicle has been stopped |
| `hour_sin`, `hour_cos` | cyclical time encoding |
| `high_traffic` | heavy traffic or construction |
| `high_pedestrian` | high pedestrian area |

### Interaction Features (11)

| Feature | What It Captures | Signal |
|---------|------------------|--------|
| `stuck_in_traffic` | Stuck + heavy traffic | Strong FP indicator |
| `stuck_in_construction` | Stuck + construction zone | Strong FP indicator |
| `stuck_clear_road` | Stuck + clear conditions | Real intervention likely |
| `object_query_high_ped` | Object query + busy area | Strong FP indicator |
| `object_query_low_ped` | Object query + empty area | Real intervention likely |
| `object_query_moving` | Object query + vehicle moving | Real intervention likely |
| `ev_far_away` | EV alert + far distance (>200m) | Strong FP indicator |
| `ev_close` | EV alert + close (<50m) | Real intervention likely |
| `speed_anomaly_in_traffic` | Slow + heavy traffic | Strong FP indicator |
| `speed_anomaly_clear` | Slow + clear road | Real intervention likely |
| `impact_rough_road` | Impact + residential/downtown | FP indicator (speed bumps) |

---

## Development Journey

### Phase 1: ML Pipeline

**Attempt 1: VAE Anomaly Detection.**
Trained a Variational Autoencoder on false positives only, expecting real interventions to have high reconstruction error. Result: 1.05x separation ratio. The model couldn't distinguish FPs from real interventions because the VAE learns global feature distributions and missed the contextual interactions.

**Attempt 2: VAE + Interaction Features.**
Added the 11 interaction features to help the VAE see the patterns. Still 1.05x separation. The interaction features got diluted across all notification types in the global distribution.

*The VAE code has been removed from the repository and the PyTorch dependency dropped. The experiments are documented here for the learning value.*

**Attempt 3: XGBoost Classifier.**
Supervised classification with interaction features. 64% FP reduction, 0.946 ROC-AUC. The interaction features became top predictors. For tabular data with categorical features, feature engineering often matters more than model architecture.

### Phase 2: Production System

Built the inference and monitoring layer on top of the trained model:
- FastAPI backend serving real-time predictions with Pydantic validation and connection pooling
- React/TypeScript dashboard with five monitoring panels and interactive simulation
- Model health tracking with confidence distribution, accuracy monitoring, and status indicators
- Docker Compose orchestration with automatic schema setup and demo seeding
- nginx reverse proxy for unified frontend/API routing

---

## Project Structure

```
sentinel/
├── api/
│   ├── main.py                  # FastAPI app, lifespan, health endpoint
│   ├── models.py                # Pydantic request/response schemas
│   ├── auth.py                  # API key authentication
│   ├── logging_config.py        # Structured logging setup
│   ├── routes/
│   │   ├── predict.py           # POST /api/predict
│   │   ├── alerts.py            # GET /api/alerts
│   │   └── stats.py             # GET /api/stats, /stats/model-health
│   └── services/
│       ├── model_service.py     # XGBoost loading, feature engineering
│       └── db_service.py        # TimescaleDB queries, connection pool
├── dashboard/
│   └── src/
│       ├── App.tsx              # Layout with sidebar navigation
│       ├── components/
│       │   ├── AlertFeed.tsx    # Recent alerts table
│       │   ├── ModelHealth.tsx  # Model monitoring panel
│       │   ├── OverviewCards.tsx # Summary stat cards
│       │   ├── SimulatePanel.tsx # Interactive prediction form
│       │   └── TypeBreakdown.tsx # Per-type bar chart
│       └── services/api.ts      # API client with TypeScript interfaces
├── fleet_data/
│   ├── generate_fleet_data.py   # 500-vehicle fleet simulation
│   ├── baseline_alerter.py      # Baseline false positive analysis
│   └── useful_queries.sql       # SQL analysis queries
├── ml/
│   ├── constants.py             # Shared encoding maps (single source of truth)
│   ├── prepare_data.py          # Feature engineering (28 features)
│   ├── train_classifier.py      # XGBoost training and evaluation
│   ├── run_pipeline.py          # End-to-end ML pipeline
│   ├── xgboost_model.json       # Trained model (committed)
│   └── xgboost_config.joblib    # Feature columns + threshold
├── scripts/
│   └── seed_demo.py             # Generate and seed 1,000 demo predictions
├── docker-compose.yml           # Full stack: DB + API + Dashboard
├── Dockerfile.api               # Python 3.11 with model files
├── Dockerfile.dashboard         # Multi-stage: node build, nginx serve
├── nginx.conf                   # Proxy /api/ to FastAPI, SPA fallback
├── entrypoint.sh                # DB wait, schema setup, auto-seed, start
├── tests/
│   ├── unit/                    # Unit tests (models, services, features)
│   ├── integration/             # API endpoint tests
│   └── ml/                      # ML pipeline and parity tests
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── .env.example                 # Environment variable template
├── setup_database.py            # TimescaleDB schema and hypertable
└── requirements.txt
```

---

## Tech Stack

- **Backend:** FastAPI, Pydantic, psycopg2 connection pooling
- **Frontend:** React 19, TypeScript, Recharts, Tailwind CSS
- **ML:** XGBoost, scikit-learn, NumPy
- **Database:** TimescaleDB (time-series optimized PostgreSQL)
- **Infrastructure:** Docker Compose, nginx reverse proxy, multi-stage builds
- **Testing:** pytest (240+ tests), Vitest (42+ tests), GitHub Actions CI

---

## Notification Types

| Type | Subtype | Description | Baseline FP Rate |
|------|---------|-------------|------------------|
| **verification_request** | object_query | "Is something in my path?" | 83% |
| | traffic_signal_verify | "Is this signal correct?" | 10% |
| | lane_mapping_verify | "Do lanes match my map?" | 31% |
| **emergency_vehicle_alert** | - | "EV detected nearby" | 70% |
| **stuck** | - | "I can't move forward" | 61% |
| **speed_anomaly** | - | "I'm slower than expected" | 57% |
| **impact_l0** | - | "Low-speed impact detected" | 48% |
| **passenger_assist** | - | "Rider requested help" | 0% (always real) |

---

## Author

**Derek Zhang**  
MS Computer Science, Northeastern University  
[LinkedIn](https://linkedin.com/in/derek-zhang-963169230) | [GitHub](https://github.com/DerekZ-113)

*Built from 3 years at Zoox working with autonomous vehicle fleet operations.*
