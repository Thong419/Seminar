# Deployment Guide

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Project dependencies from `requirements.txt`

## One-Command Startup

```bash
docker compose up --build
```

This starts the FastAPI backend on port `8000` and the Streamlit frontend on port `8501`.

## Environment Variables

- `API_HOST` and `API_PORT` control the backend host and port.
- `STREAMLIT_HOST` and `STREAMLIT_PORT` control the frontend host and port.
- `MODEL_DIR` points to the deployed model artifacts.
- `MLFLOW_TRACKING_URI` is used by experiment metadata and deployment logs.

## Health Checks

- Backend: `GET /health`
- Frontend: root page `/`

Docker Compose includes health checks for both services.

## Deployment Validation Steps

1. Start the stack with `docker compose up --build`.
2. Confirm `http://localhost:8000/health` returns `{"status":"healthy"}`.
3. Confirm the Streamlit UI loads at `http://localhost:8501`.
4. Submit a sample article and confirm `/analyze` returns a complete response.
5. Confirm monitoring artifacts are written under `artifacts/monitoring/`.
