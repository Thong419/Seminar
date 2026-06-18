"""Deterministic integration tests for the Agent Controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.agent.controller import AgentController, AgentControllerConfig
from src.agent.decision_tool import DecisionTool


@dataclass
class FakePrediction:
    label: str
    confidence: float


class FakePredictor:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence
        self.seen_texts: list[str] = []

    def predict(self, text: str) -> FakePrediction:
        self.seen_texts.append(text)
        return FakePrediction(label=self.label, confidence=self.confidence)


class FakeClassifierTool:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence
        self.seen_texts: list[str] = []

    def run(self, article_text: str) -> dict[str, float | str]:
        self.seen_texts.append(article_text)
        return {"label": self.label, "confidence": self.confidence}


class FakeEvidenceTool:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.seen_texts: list[str] = []

    def run(self, article_text: str) -> dict[str, object]:
        self.seen_texts.append(article_text)
        return self.result


class FakeExplainabilityService:
    def explain(self, article_text: str, prediction: str, confidence: float, evidence: list[object], evidence_score: float):
        return SimpleNamespace(
            final_explanation=f"Explained {prediction} with confidence {confidence:.2f} and evidence_score {evidence_score:.2f}.",
            important_tokens=[{"token": "NASA", "importance": 0.91}, {"token": "Mars", "importance": 0.84}],
        )


def build_evidence_bundle(support_score: float, contradiction_score: float, quality_score: float, stance: str) -> dict[str, object]:
    return {
        "evidence_found": True,
        "claim": "Sample claim",
        "queries": ["sample query"],
        "summary": "Retrieved evidence bundle.",
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "source_credibility_score": 0.9,
        "evidence_quality_score": quality_score,
        "conflict_flag": stance == "conflict",
        "sources": [
            {
                "title": "Evidence source",
                "source": "Wikipedia",
                "content": "Perseverance is a NASA rover exploring Mars.",
                "relevance_score": 0.92,
                "url": "https://en.wikipedia.org/wiki/Perseverance_(rover)",
                "query": "NASA Perseverance Mars",
                "provider": "wikipedia_html",
                "source_credibility": 0.9,
                "stance": "support" if stance == "support" else "contradict",
                "matched_terms": ["nasa", "mars", "perseverance"],
            }
        ],
    }


def build_controller(label: str, confidence: float, evidence_bundle: dict[str, object], *, tracing: bool = True, trace_dir: Path | None = None) -> AgentController:
    config = AgentControllerConfig(
        confidence_threshold=0.85,
        enable_evidence_retrieval=True,
        enable_tracing=tracing,
        trace_artifact_dir=str(trace_dir or Path("artifacts/agent_traces/tests")),
    )
    return AgentController(
        predictor=FakePredictor(label=label, confidence=confidence),
        agent_config=config,
        classifier=FakeClassifierTool(label, confidence),
        evidence_tool=FakeEvidenceTool(evidence_bundle),
        decision_tool=DecisionTool(),
        explainability_service=FakeExplainabilityService(),
    )


class TestAgentController:
    def test_controller_always_retrieves_evidence(self, tmp_path: Path) -> None:
        controller = build_controller(
            label="fake",
            confidence=0.99,
            evidence_bundle=build_evidence_bundle(0.81, 0.08, 0.84, "support"),
            trace_dir=tmp_path / "traces",
        )

        result = controller.run("NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars.", request_id="req-1")

        assert result.evidence_found is True
        assert result.queries == ["sample query"]
        assert result.support_score > result.contradiction_score
        assert result.human_review_state in {"REAL", "UNCERTAIN"}
        assert result.trace is not None
        assert controller.evidence_tool.seen_texts == ["NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars."]
        assert "Evidence support" in result.explanation
        assert (tmp_path / "traces").exists()

    def test_controller_marks_uncertain_on_conflict(self) -> None:
        controller = build_controller(
            label="fake",
            confidence=0.98,
            evidence_bundle=build_evidence_bundle(0.83, 0.04, 0.88, "conflict"),
            tracing=False,
        )

        result = controller.run("NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars.")

        assert result.conflict_flag is True
        assert result.human_review_state == "UNCERTAIN"
        assert "UNCERTAIN" in result.explanation
        assert result.trace is None

    def test_controller_returns_real_when_evidence_aligns(self) -> None:
        controller = build_controller(
            label="real",
            confidence=0.93,
            evidence_bundle=build_evidence_bundle(0.84, 0.04, 0.87, "support"),
        )

        result = controller.run("NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars.")

        assert result.human_review_state == "REAL"
        assert result.conflict_flag is False
        assert result.risk_level == "low"
        assert result.evidence_summary == "Retrieved evidence bundle."

    def test_controller_result_fields_are_populated(self) -> None:
        controller = build_controller(
            label="fake",
            confidence=0.88,
            evidence_bundle=build_evidence_bundle(0.2, 0.7, 0.9, "conflict"),
        )

        result = controller.run("The WHO said there is no evidence that 5G causes COVID-19.")

        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "trust_score")
        assert hasattr(result, "risk_level")
        assert hasattr(result, "human_review_state")
        assert hasattr(result, "conflict_flag")
        assert hasattr(result, "decision_reason")
        assert hasattr(result, "explanation")
        assert hasattr(result, "evidence_found")
        assert hasattr(result, "sources")
        assert hasattr(result, "important_tokens")
        assert hasattr(result, "evidence_summary")
        assert hasattr(result, "queries")
        assert 0.0 <= result.trust_score <= 1.0
