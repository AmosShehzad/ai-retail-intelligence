#!/bin/bash
set -e

echo "Checking database..."
NEEDS_DATA=1
if [ -f /app/database/retail.db ]; then
    COUNT=$(python -c "
import sqlite3
try:
    conn = sqlite3.connect('/app/database/retail.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM products')
    print(c.fetchone()[0])
    conn.close()
except Exception:
    print(0)
" 2>/dev/null)
    if [ "$COUNT" -gt "0" ] 2>/dev/null; then
        NEEDS_DATA=0
    fi
fi

if [ "$NEEDS_DATA" -eq "1" ]; then
    echo "Database empty or missing - generating fresh data..."
    python database/db_manager.py
    python data/generate_realistic_data.py
else
    echo "Database already has data, skipping generation."
fi

echo "Checking FAISS index..."
if [ ! -f /app/faiss_index/retail.index ]; then
    echo "FAISS index missing - building..."
    python rag/embedder.py
else
    echo "FAISS index already exists, skipping build."
fi

echo "Starting FastAPI backend on port 8000..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

echo "Waiting 5 seconds for FastAPI to initialize..."
sleep 5

echo "Starting Streamlit on port ${PORT:-10000}..."
streamlit run frontend/dashboard.py \
    --server.port ${PORT:-10000} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false