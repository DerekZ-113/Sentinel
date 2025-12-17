# Sentinel

**Context-aware notification triage for autonomous vehicle fleets**

A machine learning system that reduces operator alert fatigue by learning which notifications actually need intervention — trained on 18M+ telemetry records from a 500-vehicle fleet, achieving significant reductions in false positive rates across all notification types.

---

## The Problem

AV fleet operators are drowning in notifications:

| Notification Type | What It Means | Reality |
|-------------------|---------------|---------|
| **Object Query** | "Is something blocking me?" | 90% of the time, someone just walked by |
| **EV Alert** | "Emergency vehicle detected" | Usually on a different road or too far away |
| **Stuck** | "I can't move" | Often just traffic or a red light |

> *"95% of notifications don't need intervention"* — the problem we're solving.

When everything is an alert, nothing is an alert. Real issues get buried in noise.

---

## The Solution

Sentinel uses a **Variational Autoencoder (VAE)** to learn the patterns of false positives:

1. **Train on FPs only**: The model learns "what does a notification that doesn't need intervention look like?"
2. **Measure surprise**: Real interventions look different from what the model learned
3. **Triage intelligently**: High reconstruction error = likely needs real intervention

**The key insight**: Context matters. A "stuck" notification during rush hour traffic is almost always a false positive. A "stuck" notification on a clear highway at 2 AM probably needs attention.

---

## Results

### Overall Performance

| Metric | Baseline | VAE | Improvement |
|--------|----------|-----|-------------|
| False Positive Rate | ~70% | ~25% | ↓ 65% |
| Precision | ~30% | ~75% | ↑ 150% |
| Daily Alerts | ~15,000 | ~5,000 | ↓ 67% |

### Per-Notification-Type Breakdown

| Type | Baseline FP | VAE FP | Reduction |
|------|-------------|--------|-----------|
| verification_request/object_query | 90% | ~30% | ↓ 67% |
| verification_request/traffic_signal_verify | 10% | ~5% | ↓ 50% |
| verification_request/lane_mapping_verify | 30% | ~12% | ↓ 60% |
| emergency_vehicle_alert | 70% | ~25% | ↓ 64% |
| stuck | 65% | ~22% | ↓ 66% |
| speed_anomaly | 50% | ~18% | ↓ 64% |
| impact_l0 | 40% | ~15% | ↓ 63% |
| passenger_assist | 0% | 0% | N/A |

---

## Notification Types

| Type | Subtype | Description | Typical FP Cause |
|------|---------|-------------|------------------|
| **verification_request** | object_query | "Is something in my path?" | Pedestrian walked by |
| | traffic_signal_verify | "Is this traffic signal correct?" | Map slightly outdated |
| | lane_mapping_verify | "Do these lanes match my map?" | Construction zone |
| **emergency_vehicle_alert** | - | "EV detected nearby" | EV on different road/too far |
| **stuck** | - | "I can't move forward" | Traffic, red light, construction |
| **speed_anomaly** | - | "I'm slower than expected" | Heavy traffic |
| **impact_l0** | - | "Low-speed impact detected" | Speed bump, rough road |
| **passenger_assist** | - | "Rider requested help" | Always real (human-initiated) |

---

## Architecture

```
Fleet Telemetry (18M+ records, 500 vehicles, 7 days)
         │
         ▼
┌─────────────────────────────────────────┐
│  Feature Engineering (18 features)      │
│  - Speed context (ratio, deviation)     │
│  - Road context (type, traffic)         │
│  - Notification context (type, subtype) │
│  - Situational (EV distance, pedestrians)│
│  - Time patterns (cyclical hour)        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  VAE (trained on false positives)       │
│  18 → 128 → 64 → 16 → 64 → 128 → 18     │
│  + Dropout + BatchNorm                  │
└─────────────────────────────────────────┘
         │
         ▼
   Reconstruction Error
         │
    High Error = Likely Needs Intervention
```

---

## Features

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
| `high_traffic` | derived: heavy traffic or construction |
| `high_pedestrian` | derived: high pedestrian area |

---

## Project Structure

```
sentinel/
├── fleet_data/
│   ├── generate_fleet_data.py   # 500-vehicle simulation
│   ├── baseline_alerter.py      # Baseline analysis
│   └── useful_queries.sql       # SQL analysis queries
├── ml/
│   ├── prepare_data.py          # Feature engineering
│   ├── vae_model.py             # VAE architecture
│   ├── train_vae.py             # GPU training pipeline
│   └── vae_alerter.py           # Evaluation & results
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

# 4. Generate fleet data (~25-30 min for 18M records)
cd fleet_data
python generate_fleet_data.py

# 5. Analyze baseline
python baseline_alerter.py

# 6. Prepare ML data
cd ../ml
python prepare_data.py

# 7. Train VAE (GPU recommended)
python train_vae.py

# 8. Evaluate
python vae_alerter.py

# 9. View training curves
tensorboard --logdir=runs
```

---

## Tech Stack

- **Simulation**: Python fleet simulator with realistic traffic patterns
- **Database**: TimescaleDB (time-series optimized PostgreSQL)
- **ML Framework**: PyTorch with CUDA support
- **Training**: GPU-accelerated with early stopping, LR scheduling
- **Monitoring**: TensorBoard

---

## Key Insights

1. **Train on false positives**: The VAE learns "what does a notification that doesn't need help look like?" Real interventions are outliers.

2. **Context is everything**: 
   - "Stuck" + traffic jam = false positive
   - "Stuck" + clear highway = real problem
   - "Object query" + downtown + rush hour = someone walked by
   - "Object query" + highway + 2 AM = something's actually there

3. **Not all notifications are equal**: Object queries have 90% FP rate. Passenger assists have 0%. The model learns these differences.

---

## Dataset

- **18M+ records** — 500 vehicles × 7 days × 5-second intervals
- **6 notification types** with subtypes
- **Ground truth labels** — `needs_intervention` based on context
- **Realistic patterns** — Rush hour traffic, construction zones, pedestrian activity

---

## Author

Derek Zhang  
MS Computer Science, Northeastern University

*Built from real-world experience in autonomous vehicle operations at Zoox.*
