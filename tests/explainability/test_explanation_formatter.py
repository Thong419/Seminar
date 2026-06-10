from src.agents.state import EvidenceItem
from src.explainability.explanation_formatter import TrustScoreWeights, format_explanation


def test_format_explanation_builds_unified_object() -> None:
    evidence = [
        EvidenceItem(
            title="Reuters verifies claim",
            source="Reuters",
            content="Reuters verifies the article.",
            relevance_score=0.92,
            url="https://example.com/reuters",
        )
    ]

    explanation = format_explanation(
        prediction="fake",
        confidence=0.91,
        evidence_score=0.84,
        source_trust=1.0,
        important_tokens=[{"token": "fake", "importance": 0.82}],
        evidence=evidence,
        weights=TrustScoreWeights(),
    )

    assert explanation.prediction == "fake"
    assert explanation.trust_score >= 0
    assert "Trust score" in explanation.final_explanation
    assert explanation.important_tokens[0]["token"] == "fake"
