"""Application configuration access.

The module intentionally keeps behavior minimal at scaffold stage and exposes a
single typed settings object that can later be extended to load YAML and env
overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_name: str = "Fake News & Misinformation Detection System"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = 8501
    model_name: str = "roberta-base"
    model_dir: Path = Path("models/roberta")
    baseline_dir: Path = Path("models/baseline")
    mlflow_tracking_uri: str = "http://localhost:5000"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load settings from environment variables with safe defaults."""

    defaults = AppSettings()
    return AppSettings(
        app_env=os.getenv("APP_ENV", defaults.app_env),
        log_level=os.getenv("LOG_LEVEL", defaults.log_level),
        api_host=os.getenv("API_HOST", defaults.api_host),
        api_port=int(os.getenv("API_PORT", str(defaults.api_port))),
        streamlit_host=os.getenv("STREAMLIT_HOST", defaults.streamlit_host),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", str(defaults.streamlit_port))),
        model_name=os.getenv("MODEL_NAME", defaults.model_name),
        model_dir=Path(os.getenv("MODEL_DIR", str(defaults.model_dir))),
        baseline_dir=Path(os.getenv("BASELINE_DIR", str(defaults.baseline_dir))),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", defaults.mlflow_tracking_uri),
    )