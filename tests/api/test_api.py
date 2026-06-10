from __future__ import annotations

from fastapi.testclient import TestClient

from src.agents.state import EvidenceItem
from src.api.dependencies import get_predictor, get_workflow
from src.api.main import create_app


class DummyPrediction:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class DummyPredictor:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def predict(self, text: str) -> DummyPrediction:
        self.seen_texts.append(text)
        return DummyPrediction(label="fake", confidence=0.91)


class DummyWorkflow:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def run(self, article_text: str) -> dict[str, object]:
        self.seen_texts.append(article_text)
        return {
            "predicted_label": "fake",
            "confidence": 0.91,
            "trust_score": 88,
            "important_tokens": [{"token": "fake", "importance": 0.81}],
            "retrieved_evidence": [
                EvidenceItem(
                    title="Reuters verifies claim",
                    source="Reuters",
                    content="Reuters verifies the claim.",
                    relevance_score=0.95,
                    url="https://example.com/reuters",
                )
            ],
            "evidence_summary": "Retrieved 1 evidence item from 1 sources. Top sources: Reuters. Top evidence: Reuters verifies claim.",
            "explanation": "The article was classified as fake with confidence 0.91.",
            "explanation_details": {"final_decision": "confirmed_fake"},
        }


def build_client() -> TestClient:
    app = create_app()
    predictor = DummyPredictor()
    workflow = DummyWorkflow()
    app.dependency_overrides[get_predictor] = lambda: predictor
    app.dependency_overrides[get_workflow] = lambda: workflow
    app.state.test_predictor = predictor
    app.state.test_workflow = workflow
    return TestClient(app)


def test_health_endpoint_returns_healthy_status() -> None:
    client = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_version_endpoint_returns_model_version() -> None:
    client = build_client()

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"model_name": "roberta-base", "model_version": "1.0.0"}


def test_predict_endpoint_uses_predictor_dependency() -> None:
    client = build_client()

    response = client.post("/predict", json={"text": "article text"})

    assert response.status_code == 200
    assert response.json() == {"prediction": "fake", "confidence": 0.91}


def test_analyze_endpoint_returns_full_response() -> None:
    client = build_client()

    response = client.post("/analyze", json={"text": "article text"})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "fake"
    assert body["confidence"] == 0.91
    assert body["trust_score"] == 88
    assert body["important_tokens"][0]["token"] == "fake"
    assert body["evidence"][0]["source"] == "Reuters"
    assert "final_explanation" in body


def test_api_masks_pii_before_backend_call() -> None:
    client = build_client()

    client.post("/predict", json={"text": "Email john@example.com and visit https://example.com"})
    client.post("/analyze", json={"text": "Contact john@example.com at 12 Main Street."})

    assert client.app.state.test_predictor.seen_texts[0].startswith("Email [EMAIL] and visit [URL]")
    assert "john@example.com" not in client.app.state.test_workflow.seen_texts[0]
    assert "[EMAIL]" in client.app.state.test_workflow.seen_texts[0]
