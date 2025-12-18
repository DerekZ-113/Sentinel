# Sentinel

**Context-aware notification triage for autonomous vehicle fleets**

A machine learning system that reduces operator alert fatigue by learning which notifications actually need intervention. Achieved **64% reduction in false positives** and **2M fewer unnecessary alerts daily** on a 500-vehicle fleet simulation.

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

Sentinel uses an **XGBoost classifier** with domain-knowledge interaction features to predict which notifications actually need operator intervention.

**Key insight**: Context matters. A "stuck" notification during rush hour traffic is almost always a false positive. A "stuck" notification on a clear highway probably needs attention.

---

## Results

### Overall Performance

| Metric | Baseline | Sentinel | Improvement |
|--------|----------|----------|-------------|
| False Positive Rate | 60.8% | 21.7% | ↓ 64% |
| Precision | 39.2% | 78.3% | ↑ 100% |
| Recall | 100% | 86.6% | - |
| F1 Score | - | 82.2% | - |
| ROC-AUC | - | 0.946 | - |

### Per-Notification-Type Breakdown

| Type | Baseline FP | Sentinel FP | Reduction |
|------|-------------|-------------|-----------|
| verification_request/object_query | 82.9% | 0.0% | ↓ 100% |
| emergency_vehicle_alert | 70.0% | 0.0% | ↓ 100% |
| speed_anomaly | 57.2% | 0.5% | ↓ 99% |
| stuck | 61.1% | 38.0% | ↓ 38% |
| impact_l0 | 47.5% | 42.7% | ↓ 10% |
| verification_request/lane_mapping_verify | 30.7% | 28.9% | ↓ 6% |
| verification_request/traffic_signal_verify | 9.6% | 9.4% | ↓ 2% |
| passenger_assist | 0.0% | 0.0% | N/A |

### Operator Impact

| Metric | Baseline | Sentinel |
|--------|----------|----------|
| Alerts per day | 3.8M | 1.7M |
| False alarms per day | 2.3M | 360K |
| Workload | 100% | 43% |

**🎯 2 million false alarms eliminated daily**

---

## Feature Importance

The interaction features engineered from domain knowledge became the top predictors:

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

**6 of the top 10 features are interaction features** — domain knowledge encoded directly into the model.

---

## Architecture

```
Fleet Telemetry (8M+ records, 500 vehicles)
         │
         ▼
┌─────────────────────────────────────────┐
│  Feature Engineering (28 features)      │
│  - Speed context (ratio, deviation)     │
│  - Road context (type, traffic)         │
│  - Notification context (type, subtype) │
│  - Situational (EV distance, pedestrians│
│  - Interaction features (domain knowledge│
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  XGBoost Classifier                     │
│  - 500 trees, max_depth=6               │
│  - Class-balanced weighting             │
│  - PR-AUC optimized                     │
└─────────────────────────────────────────┘
         │
         ▼
   P(needs_intervention)
         │
    >0.5 = Flag for operator
```

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

### Interaction Features (11) — Domain Knowledge Encoded

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

### Attempt 1: VAE Anomaly Detection

**Hypothesis**: Train a Variational Autoencoder on false positives only. Real interventions should have high reconstruction error (they look "different" from FPs).

**Result**: 1.05x separation ratio — model couldn't distinguish FPs from real interventions.

**Why it failed**: VAE learns global feature distributions. It learned "stuck notifications look like X" and "heavy traffic looks like Y" separately, but couldn't learn "stuck + heavy traffic = FP."

### Attempt 2: VAE + Interaction Features

**Hypothesis**: Add explicit interaction features to help the VAE see the patterns.

**Result**: Still 1.05x separation. The interaction features got diluted across all notification types in the global distribution.

### Attempt 3: XGBoost Classifier

**Hypothesis**: Supervised classification with interaction features will directly learn the decision boundary.

**Result**: 64% FP reduction, 0.946 ROC-AUC. Interaction features became top predictors.

### Key Learning

**For tabular data with categorical features, feature engineering often matters more than model architecture.** The same interaction features that failed in a VAE became dominant predictors in a tree-based classifier.

---

## Project Structure

```
sentinel/
├── fleet_data/
│   ├── generate_fleet_data.py   # 500-vehicle simulation
│   ├── baseline_alerter.py      # Baseline analysis
│   └── useful_queries.sql       # SQL analysis queries
├── ml/
│   ├── prepare_data.py          # Feature engineering (28 features)
│   ├── train_classifier.py      # XGBoost training + evaluation
│   ├── run_pipeline.py          # End-to-end pipeline
│   ├── vae_model.py             # VAE architecture (historical)
│   ├── train_vae.py             # VAE training (historical)
│   └── vae_alerter.py           # VAE evaluation (historical)
├── setup_database.py            # TimescaleDB schema
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/sentinel.git
cd sentinel
pip install -r requirements.txt

# 2. Start TimescaleDB
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb:latest-pg14

# 3. Initialize database
python setup_database.py

# 4. Generate fleet data
cd fleet_data
python generate_fleet_data.py

# 5. Run ML pipeline
cd ../ml
python run_pipeline.py

# Output: XGBoost model + evaluation results
```

---

## Tech Stack

- **Simulation**: Python fleet simulator with realistic traffic patterns
- **Database**: TimescaleDB (time-series optimized PostgreSQL)
- **ML**: XGBoost, scikit-learn
- **Historical**: PyTorch VAE (documented for learning journey)

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

Derek Zhang  
MS Computer Science, Northeastern University

*Built from real-world experience in autonomous vehicle fleet operations.*
