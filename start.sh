#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

echo "Waiting 5 seconds for FastAPI to initialize..."
sleep 5

# Render sets $PORT automatically — Streamlit must bind to it
echo "Starting Streamlit on port ${PORT:-10000}..."
streamlit run frontend/dashboard.py \
    --server.port ${PORT:-10000} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false