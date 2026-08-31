# Single-container deploy: builds the React frontend, then serves it (as static
# files) from the same FastAPI app that serves the API -- see the `_DIST` mount in
# backend/main.py. CPU-only inference (NFR-4 in docs/SRS_webapp.md), so no CUDA base
# image needed.

# ---- stage 1: build the frontend -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY webapp/frontend/package*.json ./
RUN npm ci
COPY webapp/frontend ./
RUN npm run build

# ---- stage 2: backend + built frontend --------------------------------------
FROM python:3.11-slim
WORKDIR /app

# opencv needs libgl/libglib even in "headless" mode on some slim bases
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt ./webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY scripts ./scripts
COPY webapp/backend ./webapp/backend
COPY reports ./reports
# Only the one checkpoint the demo actually serves is tracked in git (9MB); every
# other experimental checkpoint and the training dataset are excluded (see .gitignore).
COPY models ./models
COPY --from=frontend /app/frontend/dist ./webapp/frontend/dist

WORKDIR /app/webapp
EXPOSE 8000
# Render (and most PaaS hosts) inject $PORT and expect the app to bind it; default
# to 8000 for a plain local `docker run`. Shell form so $PORT actually expands.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
