#!/bin/bash
# Sentinel API Entrypoint
# Waits for DB (bounded), runs idempotent schema setup, seeds once
# (marker-guarded), starts API with graceful SIGTERM shutdown.

set -e

# libpq/psycopg2 read PG* from the environment — no password splicing
# into command lines (visible in `ps`, breaks on special characters)
export PGHOST="${DB_HOST:-localhost}"
export PGPORT="${DB_PORT:-5432}"
export PGDATABASE="${DB_NAME:-postgres}"
export PGUSER="${DB_USER:-postgres}"
export PGPASSWORD="${DB_PASSWORD:-password}"

db_probe() {
    python -c "import psycopg2; psycopg2.connect().close()" 2>/dev/null
}

echo "⏳ Waiting for database at $PGHOST:$PGPORT (max 60s)..."
for i in $(seq 1 60); do
    if db_probe; then
        break
    fi
    if [ "$i" = "60" ]; then
        echo "❌ Database not reachable after 60s — giving up"
        exit 1
    fi
    sleep 1
done
echo "✅ Database ready"

# Idempotent schema setup (CREATE IF NOT EXISTS; destructive reset only
# via an explicit `python setup_database.py --reset`)
echo "📦 Setting up database schema..."
python setup_database.py

# Start API in background for seeding
echo "🚀 Starting API..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Forward SIGTERM/SIGINT to uvicorn so `docker stop` shuts down gracefully
# instead of hanging the grace period and getting SIGKILLed mid-request
trap 'kill -TERM "$API_PID" 2>/dev/null' TERM INT

echo "⏳ Waiting for API (max 60s)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "❌ API process exited during startup"
        exit 1
    fi
    if [ "$i" = "60" ]; then
        echo "❌ API not responding after 60s — giving up"
        kill -TERM "$API_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Seed guard: a completion marker, not a row count. A count probe can't
# tell "empty" from "probe failed" (would double-seed on a transient error)
# or from "interrupted mid-seed" (would keep truncated demo data forever).
SEEDED=$(python - <<'PY'
import psycopg2
conn = psycopg2.connect()
cur = conn.cursor()
cur.execute("SELECT EXISTS (SELECT 1 FROM seed_status)")
print("yes" if cur.fetchone()[0] else "no")
conn.close()
PY
) || {
    echo "❌ Seed-status probe failed — refusing to guess; not seeding"
    SEEDED="probe-failed"
}

if [ "$SEEDED" = "no" ]; then
    echo "🌱 Seeding demo data..."
    if python -m scripts.seed_demo; then
        python - <<'PY'
import psycopg2
conn = psycopg2.connect()
cur = conn.cursor()
cur.execute("INSERT INTO seed_status (note) VALUES ('entrypoint demo seed')")
conn.commit()
conn.close()
PY
        echo "✅ Seed complete (marker written)"
    else
        echo "⚠️ Seeding failed — API stays up; seed will retry on next start"
    fi
elif [ "$SEEDED" = "yes" ]; then
    echo "✅ Demo data already seeded, skipping"
fi

echo "✅ Sentinel API ready on port 8000"
wait "$API_PID"
