from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.dependencies import get_agent_controller, get_monitoring_service
from src.api.main import create_app


class FakeMonitoringService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_prediction(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class FakeAgentController:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.seen_texts: list[str] = []

    def run(self, article_text: str, request_id: str = "agent_request") -> SimpleNamespace:
        self.seen_texts.append(article_text)
        return self.response


def build_response(*, prediction: str, human_review_state: str, conflict_flag: bool, trust_score: float, support_score: float, contradiction_score: float) -> SimpleNamespace:
    return SimpleNamespace(
        label=prediction,
        confidence=0.95 if prediction != "uncertain" else 0.48,
        trust_score=trust_score,
        risk_level="low" if human_review_state != "UNCERTAIN" else "high",
        human_review_state=human_review_state,
        conflict_flag=conflict_flag,
        decision_reason=f"decision for {human_review_state}",
        explanation=f"explanation for {human_review_state}",
        evidence_found=True,
        sources=[
            {
                "title": "Source",
                "source": "Wikipedia",
                "content": "Perseverance is a NASA rover exploring Mars.",
                "relevance_score": 0.9,
                "url": "https://en.wikipedia.org/wiki/Perseverance_(rover)",
                "query": "NASA Perseverance Mars",
                "provider": "wikipedia_html",
                "source_credibility": 0.9,
                "stance": "support" if human_review_state in {"REAL", "FAKE"} and not conflict_flag else "contradict",
                "matched_terms": ["nasa", "mars", "perseverance"],
            }
        ],
        important_tokens=[{"token": "NASA", "importance": 0.91}],
        evidence_summary="Retrieved evidence bundle.",
        claim="NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars.",
        queries=["NASA Perseverance Mars"],
        support_score=support_score,
        contradiction_score=contradiction_score,
        source_credibility_score=0.9,
        evidence_quality_score=0.85,
        trace=SimpleNamespace(
            to_dict=lambda: {
                "tool_traces": [
                    {"tool_name": "classifier", "execution_time_ms": 10.0, "output_data": {"label": prediction}},
                    {"tool_name": "evidence", "execution_time_ms": 20.0, "output_data": {"num_sources": 1}},
                    {"tool_name": "decision", "execution_time_ms": 30.0, "output_data": {"trust_score": trust_score}},
                    {"tool_name": "explainability", "execution_time_ms": 40.0, "output_data": {"num_tokens": 1}},
                ],
                "final_decision": prediction,
                "total_execution_time_ms": 100.0,
            }
        ),
    )


def build_client(response: SimpleNamespace) -> TestClient:
    app = create_app()
    controller = FakeAgentController(response)
    monitoring = FakeMonitoringService()
    app.dependency_overrides[get_agent_controller] = lambda: controller
    app.dependency_overrides[get_monitoring_service] = lambda: monitoring
    app.state.test_controller = controller
    app.state.test_monitoring = monitoring
    return TestClient(app)


def test_agent_endpoint_returns_real_outcome() -> None:
    client = build_client(build_response(prediction="real", human_review_state="REAL", conflict_flag=False, trust_score=0.89, support_score=0.84, contradiction_score=0.08))

    response = client.post("/agent", json={"text": "NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars."})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "real"
    assert body["human_review_state"] == "REAL"
    assert body["conflict_flag"] is False
    assert body["support_score"] > body["contradiction_score"]
    assert body["evidence_found"] is True
    assert body["trace"]["final_decision"] == "real"


def test_agent_endpoint_returns_fake_outcome() -> None:
    client = build_client(build_response(prediction="fake", human_review_state="FAKE", conflict_flag=False, trust_score=0.86, support_score=0.12, contradiction_score=0.79))

    response = client.post("/agent", json={"text": "A hoax article with fabricated details."})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "fake"
    assert body["human_review_state"] == "FAKE"
    assert body["conflict_flag"] is False
    assert body["contradiction_score"] > body["support_score"]


def test_agent_endpoint_returns_uncertain_outcome_for_conflict() -> None:
    client = build_client(build_response(prediction="fake", human_review_state="UNCERTAIN", conflict_flag=True, trust_score=0.51, support_score=0.82, contradiction_score=0.06))

    response = client.post("/agent", json={"text": "NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars."})

    assert response.status_code == 200
    body = response.json()
    assert body["human_review_state"] == "UNCERTAIN"
    assert body["conflict_flag"] is True
    assert body["support_score"] > body["contradiction_score"]


def test_agent_endpoint_returns_uncertain_when_evidence_is_weak() -> None:
    client = build_client(build_response(prediction="uncertain", human_review_state="UNCERTAIN", conflict_flag=False, trust_score=0.42, support_score=0.05, contradiction_score=0.04))

    response = client.post("/agent", json={"text": "A vague claim with no verifiable details."})

    assert response.status_code == 200
    body = response.json()
    assert body["human_review_state"] == "UNCERTAIN"
    assert body["trust_score"] < 0.5
