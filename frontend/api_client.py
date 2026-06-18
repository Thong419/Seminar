"""HTTP client used by the Streamlit frontend to call the FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from frontend.config import FrontendConfig


@dataclass(frozen=True, slots=True)
class APIError(Exception):
    message: str
    code: str = "backend_error"
    status_code: int = 500


class APIClient:
    def __init__(self, config: FrontendConfig) -> None:
        self.config = config

    def _request(self, method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        url = f"{self.config.api_url.rstrip('/')}{path}"
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise APIError("The backend took too long to respond.", code="timeout", status_code=504) from exc
        except requests.RequestException as exc:
            raise APIError("The backend is unavailable.", code="backend_unavailable", status_code=503) from exc

        if response.status_code >= 400:
            detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error = detail.get("error", {}) if isinstance(detail, dict) else {}
            raise APIError(
                message=str(error.get("message", detail or "The backend returned an error.")),
                code=str(error.get("code", "backend_error")),
                status_code=response.status_code,
            )

        return response.json()

    def analyze(self, text: str) -> dict[str, object]:
        return self._request("POST", "/analyze", {"text": text})

    def agent(self, text: str) -> dict[str, object]:
        return self._request("POST", "/agent", {"text": text})

    def version(self) -> dict[str, object]:
        return self._request("GET", "/version")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_name: str
    dataset: str
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
