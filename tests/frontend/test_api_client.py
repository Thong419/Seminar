from __future__ import annotations

from dataclasses import dataclass

import requests

from frontend.api_client import APIClient, APIError
from frontend.config import FrontendConfig


@dataclass
class DummyResponse:
    status_code: int
    payload: dict[str, object]
    content_type: str = "application/json"

    @property
    def headers(self) -> dict[str, str]:
        return {"content-type": self.content_type}

    def json(self) -> dict[str, object]:
        return self.payload


def test_api_client_analyze_returns_json(monkeypatch) -> None:
    def fake_request(**kwargs):
        return DummyResponse(200, {"prediction": "fake", "confidence": 0.91, "trust_score": 88})

    monkeypatch.setattr(requests, "request", fake_request)
    client = APIClient(FrontendConfig(api_url="http://backend", timeout_seconds=3))

    response = client.analyze("article text")

    assert response["prediction"] == "fake"
    assert response["trust_score"] == 88


def test_api_client_raises_structured_error_on_backend_error(monkeypatch) -> None:
    def fake_request(**kwargs):
        return DummyResponse(503, {"error": {"code": "missing_model", "message": "No model."}})

    monkeypatch.setattr(requests, "request", fake_request)
    client = APIClient(FrontendConfig(api_url="http://backend", timeout_seconds=3))

    try:
        client.analyze("article text")
    except APIError as exc:
        assert exc.code == "missing_model"
        assert exc.status_code == 503
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("APIError was not raised")
