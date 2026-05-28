# Sentinel Pipeline And Job Comparison

Sentinel does not implement an autonomous agent runtime. It does include several scripts and jobs that serve different roles in the data, ML, demo, and startup flow.

## Pipeline / Job Matrix

| Name | File | Purpose | Inputs | Outputs | Interview-safe claim |
|---|---|---|---|---|---|
| Fleet data generator | `fleet_data/generate_fleet_data.py` | Simulate 500 vehicles over 7 days | TimescaleDB connection env vars | `vehicle_metrics` rows | Generates simulated training/evaluation data |
| Baseline alerter | `fleet_data/baseline_alerter.py` | Analyze all-notifications-as-alerts baseline | `vehicle_metrics` | Console baseline stats | Establishes simulated baseline FP rate |
| Data preparation | `ml/prepare_data.py` | Load DB records and engineer ML features | `vehicle_metrics` | `X_train.npy`, `X_all.npy`, `y_all.npy`, `notif_types.npy`, `scaler.joblib` | Converts raw simulation records into model features |
| Model training | `ml/train_classifier.py` | Train/evaluate XGBoost classifier | Numpy arrays from data prep | `xgboost_model.json`, `xgboost_config.joblib` | Produces committed model artifacts and printed metrics |
| Pipeline runner | `ml/run_pipeline.py` | Run prep and training in order | Existing DB data | Model artifacts and metrics output | Provides a single script for the offline ML workflow |
| Database setup | `setup_database.py` | Create demo training schema | DB connection env vars | `vehicle_metrics` hypertable and indexes | Initializes demo/training storage |
| API entrypoint | `entrypoint.sh` | Wait for DB, setup schema, start API, seed if empty | Docker Compose env | Running API and seeded `predictions` | Automates local full-stack demo startup |
| Demo seeder | `scripts/seed_demo.py` | Generate mini simulation and call live API | Running API/DB | `predictions` rows | Populates dashboard data through real API path |
| Static demo exporter | `scripts/export_demo_data.py` | Generate static JSON dashboard data | Local model artifacts | `dashboard/src/data/*.json` | Enables dashboard demo without API/DB |

## Main Distinctions

### Training pipeline vs serving path

Training pipeline:

```text
vehicle_metrics -> prepare_data.py -> train_classifier.py -> model artifacts
```

Serving path:

```text
POST /api/predict -> ModelService -> DatabaseService -> predictions table
```

Interview-safe wording:

> The repo separates offline model training from online inference. Training builds artifacts from simulated data; serving loads those artifacts and applies the same feature semantics to request payloads.

### Full-stack demo vs static demo

Full-stack demo:

```text
docker-compose -> DB + API + nginx dashboard -> seed_demo.py -> predictions table
```

Static demo:

```text
export_demo_data.py -> dashboard/src/data/*.json -> VITE_DEMO_MODE=true
```

Interview-safe wording:

> The full-stack path exercises API and DB behavior. The static demo path is for presentation without infrastructure and should not be treated as live model-serving evidence.

### Simulation data vs prediction records

Simulation data:

- Stored in `vehicle_metrics`.
- Used for training/evaluation.
- Created by `setup_database.py` and `fleet_data/generate_fleet_data.py`.

Prediction records:

- Stored in `predictions`.
- Used by dashboard read endpoints.
- Created by `DatabaseService._ensure_predictions_table()` and `store_prediction()`.

Interview-safe wording:

> `vehicle_metrics` is the training/simulation table; `predictions` is the API/dashboard table.

## Risks And Gaps

- `setup_database.py` drops `vehicle_metrics`, which is demo reset behavior rather than production migration behavior.
- Full training metrics are not captured in this first-pass documentation run.
- `scripts/seed_demo.py` posts to a hardcoded `http://localhost:8000`.
- Static demo prediction uses a heuristic, not XGBoost.
- There is no application-level agent workflow despite `AGENTS.md` developer guidance.
