#!/bin/bash
# Sentinel API Entrypoint
# Waits for DB, runs schema setup, seeds if empty, starts API

set -e

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}

echo "⏳ Waiting for database at $DB_HOST:$DB_PORT..."
until python -c "import psycopg2; psycopg2.connect(host='$DB_HOST', port=$DB_PORT, database='postgres', user='postgres', password='password')" 2>/dev/null; do
    sleep 1
done
echo "✅ Database ready"

# Run schema setup
echo "📦 Setting up database schema..."
python setup_database.py

# Start API in background for seeding
echo "🚀 Starting API..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "⏳ Waiting for API..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 1
done

# Seed if database is empty
PRED_COUNT=$(python -c "
import psycopg2
conn = psycopg2.connect(host='$DB_HOST', port=$DB_PORT, database='postgres', user='postgres', password='password')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM predictions')
print(cur.fetchone()[0])
conn.close()
" 2>/dev/null || echo "0")

if [ "$PRED_COUNT" = "0" ]; then
    echo "🌱 Seeding demo data..."
    python -m scripts.seed_demo
else
    echo "✅ Database already has $PRED_COUNT predictions, skipping seed"
fi

# Bring API back to foreground
echo "✅ Sentinel API ready on port 8000"
wait $API_PID
