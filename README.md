# Sentinel

**Context-aware notification triage for autonomous vehicle fleets**

A machine learning system that reduces operator alert fatigue by learning which notifications actually need intervention — trained on telemetry records from a 500-vehicle fleet simulation.

---

## The Problem

AV fleet operators are drowning in notifications:

| Notification Type | What It Means | Reality |
|-------------------|---------------|---------|
| **Object Query** | "Is something blocking me?" | ~83% are false positives (someone just walked by) |
| **EV Alert** | "Emergency vehicle detected" | ~70% are false positives (EV too far away) |
| **Stuck** | "I can't move" | ~61% are false positives (traffic or red light) |

> *"Most notifications don't need intervention"* — the problem we're solving.

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

*Results will be updated after training with interaction features.*

### Baseline (No ML)

| Notification Type | Volume | False Positive Rate |
|-------------------|--------|---------------------|
| stuck | 1.7M | 61.0% |
| verification_request/object_query | 800K | 82.9% |
| speed_anomaly | 690K | 57.2% |
| emergency_vehicle_alert | 187K | 70.2% |
| verification_request/traffic_signal_verify | 158K | 9.7% |
| verification_request/lane_mapping_verify | 109K | 31.2% |
| passenger_assist | 88K | 0.0% |
| impact_l0 | 57K | 47.2% |
| **Overall** | **3.8M** | **60.8%** |

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
│  VAE (trained on false positives)       │
│  28 → 256 → 128 → 32 → 128 → 256 → 28   │
│  + Dropout + BatchNorm                  │
└─────────────────────────────────────────┘
         │
         ▼
   Reconstruction Error
         │
    High Error = Likely Needs Intervention
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
| `high_traffic` | derived: heavy traffic or construction |
| `high_pedestrian` | derived: high pedestrian area |

### Interaction Features (11) — Domain Knowledge Encoded

| Feature | What It Captures | Expected Signal |
|---------|------------------|-----------------|
| `stuck_in_traffic` | Stuck + heavy traffic | Strong FP indicator |
| `stuck_in_construction` | Stuck + construction zone | Strong FP indicator |
| `stuck_clear_road` | Stuck + clear conditions | Strong REAL indicator |
| `object_query_high_ped` | Object query + busy area | Strong FP indicator |
| `object_query_low_ped` | Object query + empty area | Likely REAL indicator |
| `object_query_moving` | Object query + vehicle moving | More likely REAL |
| `ev_far_away` | EV alert + far distance (>200m) | Strong FP indicator |
| `ev_close` | EV alert + close (<50m) | Strong REAL indicator |
| `speed_anomaly_in_traffic` | Slow + heavy traffic | Strong FP indicator |
| `speed_anomaly_clear` | Slow + clear road | Likely REAL indicator |
| `impact_rough_road` | Impact + residential/downtown | FP indicator (speed bumps) |

**Why interaction features?** The VAE treats features independently. It can learn "stuck notifications look like X" and "heavy traffic looks like Y" separately, but struggles to learn "stuck + heavy traffic = false positive." Interaction features encode this domain knowledge directly.

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

# 4. Generate fleet data
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

1. **Feature engineering > model complexity**: For tabular data, explicit interaction features often matter more than deeper networks. The model can't easily learn "stuck + traffic = FP" from independent features.

2. **Train on false positives**: The VAE learns "what does a notification that doesn't need help look like?" Real interventions are outliers with high reconstruction error.

3. **Context is everything**: 
   - "Stuck" + traffic jam = false positive
   - "Stuck" + clear highway = real problem
   - "Object query" + downtown + rush hour = someone walked by
   - "Object query" + highway + 2 AM = something's actually there

4. **Not all notifications are equal**: Object queries have 83% FP rate. Passenger assists have 0%. The model learns these differences.

---

## Dataset

- **8M+ records** — 500 vehicles × 1 day × 5-second intervals
- **6 notification types** with subtypes
- **Ground truth labels** — `needs_intervention` based on context
- **Realistic patterns** — Rush hour traffic, construction zones, pedestrian activity

---

## Challenges & Learnings

### Initial Results Were Weak

First model achieved only 5% FP reduction with 1.06x separation ratio — essentially useless. The reconstruction errors for FPs and real interventions were nearly identical.

### Root Cause

The VAE was treating features independently. It learned "stuck notifications look like X" and "heavy traffic looks like Y" separately, but didn't learn that "stuck + heavy traffic = false positive."

### Solution

Added explicit interaction features that encode domain knowledge directly. Instead of hoping the model discovers these relationships, we tell it: `stuck_in_traffic = (notification_type == 'stuck') & (traffic >= heavy)`.

This is a classic lesson in ML: **for tabular data, feature engineering often matters more than model architecture**.

---

## Author

Derek Zhang  
MS Computer Science, Northeastern University

*Built from real-world experience in autonomous vehicle fleet operations.*
