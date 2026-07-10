# ══════════════════════════════════════════════════════════
# STAGE 1 — BUILDER
# Installs all Python packages including heavy ML libraries
# This stage is large — compilers, build tools included
# We copy only the results to Stage 2, not the build tools
# ══════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

# Install compilers needed to build faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements FIRST (before project code)
# Why: Docker caches this layer. If requirements.txt didn't change,
# Docker skips pip install on next build — saves 5+ minutes
COPY requirements.txt .

# Install packages to /root/.local so we can copy them to Stage 2
RUN pip install --user --no-cache-dir --default-timeout=1800 -r requirements.txt


# ══════════════════════════════════════════════════════════
# STAGE 2 — FINAL RUNTIME IMAGE
# Lean image — no compilers, just what's needed to RUN
# Copies installed packages from Stage 1
# ══════════════════════════════════════════════════════════
FROM python:3.11-slim AS final

# libgomp1 = required by faiss at runtime
# curl = needed for health checks
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make Python find the copied packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy your project code
COPY . .

# Create runtime directories (volumes will mount here)
RUN mkdir -p \
    /app/faiss_index \
    /app/data/processed \
    /app/data/raw \
    /app/database

# Document which ports this container uses
EXPOSE 8000
EXPOSE 8501

# Default command (overridden per service in docker-compose)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]