# Sentinel

**Context-aware anomaly detection for autonomous vehicle fleets**

A machine learning system that reduces false positive alerts by understanding operational context — trained on 10M+ telemetry records, achieving 54% reduction in false positives compared to traditional threshold-based alerting.

---

## The Problem

Traditional fleet monitoring systems generate excessive false positives:

| Scenario | Baseline Alerter | Reality |
|----------|-----------------|---------|
| Vehicle stopped at red light | ⚠️ ALERT: Maybe stuck! | Normal operation |
| Vehicle slow in construction zone | ⚠️ ALERT: Speed anomaly! | Expected behavior |
| Vehicle stopped on clear highway | ⚠️ ALERT: Maybe stuck! | **Actual problem** |

Operators waste time investigating false alarms while real issues get buried in noise.

---

## The Solution

Sentinel uses a **Variational Autoencoder (VAE)** trained on normal operational data to learn context-aware patterns:

- **Stopped + expected_speed ≈ 0** (traffic jam) → Normal
- **Stopped + expected_speed = 65 mph** (highway) → **Anomaly**

The model learns "what's normal given the context" rather than applying rigid thresholds.

---

## Results

| Metric | Baseline | VAE | Improvement |
|--------|----------|-----|-------------|
| False Positive Rate | 34.5% | 15.9% | **↓ 54%** |
| Precision | 65.5% | 84.1% | **↑ 28%** |

---

## Architecture

```
Fleet Telemetry (10.8M records)
         │
         ▼
┌─────────────────────────────────────┐
│  Feature Engineering                │
│  - speed_ratio (actual/expected)    │
│  - speed_deviation                  │
│  - context encoding                 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  VAE (trained on normal data only)  │
│  Input(7) → 64 → 32 → 8 → 32 → 64   │
│  + Dropout + BatchNorm              │
└─────────────────────────────────────┘
         │
         ▼
   Reconstruction Error
         │
    High Error = Anomaly
```

---

## Tech Stack

- **Simulation**: Python fleet simulator with realistic traffic patterns
- **Database**: TimescaleDB (time-series optimized PostgreSQL)
- **ML Framework**: PyTorch with CUDA support
- **Training**: GPU-accelerated (tested on RTX 3060)
- **Monitoring**: TensorBoard

---

## Dataset

- **10.8M records** — 200 vehicles × 7 days × 5-second intervals
- **~80% normal** — Context-appropriate behavior (traffic, construction, etc.)
- **~20% anomalies** — Injected faults with ground truth labels
  - `stuck`: Vehicle stopped when expected to move (2-10 min duration)
  - `wrong_speed`: Vehicle at 30-50% of expected speed (1-5 min duration)

---

## Project Structure

```
sentinel/
├── fleet_data/
│   ├── generate_fleet_data.py   # Simulation engine (200 vehicles, 7 days)
│   ├── baseline_alerter.py      # Threshold-based alerter (34.5% FP rate)
│   └── useful_queries.sql       # Analysis queries
├── ml/
│   ├── prepare_data.py          # Feature engineering pipeline
│   ├── vae_model.py             # VAE architecture with dropout/batchnorm
│   ├── train_vae.py             # GPU training with TensorBoard
│   └── vae_alerter.py           # Anomaly detection & evaluation
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

# 4. Generate fleet data (~15-20 min for 10M records)
cd fleet_data
python generate_fleet_data.py

# 5. Prepare ML data
cd ../ml
python prepare_data.py

# 6. Train VAE (GPU recommended)
python train_vae.py

# 7. Evaluate
python vae_alerter.py

# 8. View training curves
tensorboard --logdir=runs
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 7 → 64 → 32 → 8 → 32 → 64 → 7 |
| Dropout | 0.2 |
| Batch Size | 2048 |
| Optimizer | Adam (lr=0.001) |
| Early Stopping | Patience=10 |
| Device | CUDA (auto-detected) |

---

## Key Insights

1. **Feature engineering matters**: Raw speed is useless. `speed_ratio = actual/expected` captures context.

2. **Train on normal only**: VAE learns "normal" patterns. Anomalies have high reconstruction error because the model hasn't seen them.

3. **Context is everything**: A stopped vehicle isn't always a problem — it depends on traffic, construction, and expected speed.

---

## Author

Derek Zhang  
MS Computer Science, Northeastern University

*Built from real-world experience in autonomous vehicle operations.*
