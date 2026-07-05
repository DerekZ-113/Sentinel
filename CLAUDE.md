# Sentinel — Developer Guide

## Commands

```bash
# Full stack (Docker)
docker-compose up --build          # Dashboard :3000, API :8000, DB :5432

# Local dev
uvicorn api.main:app --reload --port 8000
cd dashboard && npm run dev        # Vite dev server on :3000 (proxies /api + /health to :8000)

# Tests
pytest tests/ -v                   # ~285 Python tests (+7 opt-in real-SQL)
SENTINEL_DB_TESTS=1 pytest tests/db/  # real-SQL suite (needs a scratch DB; truncates predictions)
cd dashboard && npm test           # ~120 Vitest tests
cd dashboard && npm run lint       # ESLint
cd dashboard && npx tsc -b         # type check (CI-gated)

# Coverage
pytest tests/ --cov=api --cov=fleet_data --cov=ml --cov-report=term-missing
```

## Architecture

FastAPI backend (:8000) + React/TypeScript dashboard (:3000 via nginx) + TimescaleDB (:5432). XGBoost classifier with 28 engineered features serves predictions via `/api/predict`.

Services (`ModelService`, `DatabaseService`) are module-level singletons in `api/main.py`, initialized in the lifespan context manager. Routes access them via `from api.main import get_model_service, get_db_service`.

## Key Patterns

- **Encoding maps** live in `ml/constants.py` — single source of truth for both training (`ml/prepare_data.py`) and inference (`api/services/model_service.py`). Never duplicate these maps.
- **Feature column order** in `model_service.engineer_features()` must exactly match `FEATURE_COLUMNS` in `ml/constants.py`. Wrong order = silent prediction errors.
- **No scaler**: the model consumes raw engineered features (XGBoost is tree-based; scaling added nothing and once caused silent training/serving skew). Model config is plain JSON (`ml/model_config.json`) — never reintroduce pickled artifacts.
- **Event-grouped splits**: training rows come in near-duplicate per-event runs (`ml/groups.npy` carries event_ids from `prepare_data.assign_event_ids`). Any train/val/test split must group by event_id — a row-level split leaks siblings and inflates metrics.
- **ML pipeline**: `python -m ml.run_pipeline [--max-events N]` from anywhere (steps run as modules, artifacts land in `ml/`). Data regen: `python fleet_data/generate_fleet_data.py --seed 42 --notifications-only --truncate`.
- **DB connections**: Always `conn = self._get_conn()` in try block, `self._put_conn(conn)` in finally. Uses `psycopg2.pool.ThreadedConnectionPool`.
- **API auth**: `api/auth.py` reads `API_KEY` env var inside the function body on every call (not at module load) so tests can monkeypatch. Auth skipped when env var is empty.
- **Logging**: Use `logging.getLogger("sentinel.<module>")`. No `print()` in production code. Exception: `ml/train_classifier.py` uses print for interactive CLI output tables.
- **Error handling**: Route handlers wrap bodies in try/except, raise `HTTPException(status_code=500)`. Generic exception handler in `api/main.py` catches unhandled errors.

## Environment Variables

See `.env.example`. Key vars: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `CORS_ORIGINS` (comma-separated), `API_KEY` (empty = no auth), `LOG_LEVEL` (default INFO), `ACCEPT_GROUND_TRUTH_LABELS` (default true — set false to strip client-supplied `needs_intervention_actual`). No dotenv loader — Compose reads `.env`; local workflows export vars. Docker/CI install pinned deps from `requirements.lock` (regen instructions in its header); macOS local dev uses `requirements.txt`.

## Testing

- Python: pytest with fixtures in `tests/conftest.py`. `model_service` fixture is session-scoped (loads real XGBoost model once). `client` fixture uses real model + mocked DB (no real database needed) and copies the real app's exception handlers.
- `tests/db/` runs real SQL against a scratch database — opt-in via `SENTINEL_DB_TESTS=1` (CI provides a TimescaleDB service container; it TRUNCATEs `predictions`).
- Dashboard: Vitest + React Testing Library. Mock `globalThis.fetch` in tests.
- CI: GitHub Actions on push/PR to main — pytest (with coverage + real-SQL suite, Python 3.11 matching the image) + vitest + tsc + eslint.

## Gotchas

- `hour_of_day=0` is valid (midnight). Use `if x is not None`, never `x or default` for this field.
- `ml/__init__.py` must not import heavy deps — torch was removed for this reason. Keep it lightweight.
- `NOTIFICATION_TYPE_MAP` includes `None: 0` for training data. Inference never sees None (Pydantic validates), so this is safe.
- `conftest.py` creates a FastAPI app without lifespan to avoid DB connection attempts in tests. Routes are copied from the real app.
- Dashboard `AlertFeed` and `ModelHealth` fetch their own data independently (live mode). `App.tsx` only fetches health + stats on mount.
- Demo mode (`VITE_DEMO_MODE=true`): all surfaces derive from the replay engine singleton (`dashboard/src/demo/`). Demo/live component selection happens via module-scope `DEMO_MODE` ternaries in `App.tsx` — this is what lets Rollup tree-shake fixtures + demo modules out of live builds. Don't move those ternaries into render logic.
- Demo component tests inject `createReplayEngine({pool, rng: mulberry32(seed), now})` via props (see `src/__tests__/helpers.ts`) — never set `VITE_DEMO_MODE` in tests.
- Regenerating fixtures: `python3 -m scripts.export_demo_data` from repo root, deterministic under `random.seed(42)`. Fixture values are a function of (generator code, model artifacts) — regenerate whenever either changes; between model changes, avoid adding `random` calls to the simulation path so values stay byte-identical.
