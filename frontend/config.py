"""Frontend configuration for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os

import yaml


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    api_url: str = "http://localhost:8000"
    timeout_seconds: float = 30.0
    app_title: str = "Fake News & Misinformation Detection System"
    model_config_path: Path = Path("configs/model.yaml")
    dataset_config_path: Path = Path("configs/dataset.yaml")
    evaluation_config_path: Path = Path("configs/evaluation.yaml")
    metrics_artifact_path: Path = Path("artifacts/evaluation/metrics.json")


@lru_cache(maxsize=1)
def get_frontend_config() -> FrontendConfig:
    return FrontendConfig(
        api_url=os.getenv("API_URL", FrontendConfig.api_url),
        timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", str(FrontendConfig.timeout_seconds))),
    )


def load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    return data if isinstance(data, dict) else {}
