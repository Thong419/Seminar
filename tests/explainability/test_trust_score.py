from src.explainability.explanation_formatter import TrustScoreWeights, calculate_trust_score


def test_calculate_trust_score_normalizes_to_hundred_scale() -> None:
    score = calculate_trust_score(
        confidence=0.9,
        evidence_score=0.8,
        source_trust=1.0,
        weights=TrustScoreWeights(confidence=0.5, evidence=0.3, source_trust=0.2),
    )

    assert score == 89
    assert 0 <= score <= 100
