from __future__ import annotations

from src.robustness.robustness_evaluator import RobustnessEvaluator


class DummyPrediction:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class DummyPredictor:
    def predict(self, text: str) -> DummyPrediction:
        lowered = text.lower()
        if "recipe" in lowered:
            return DummyPrediction("real", 0.42)
        if "fake" in lowered or "miracle" in lowered:
            return DummyPrediction("fake", 0.88)
        return DummyPrediction("fake", 0.76)


def test_robustness_evaluator_reports_adversarial_flips(tmp_path) -> None:
    evaluator = RobustnessEvaluator(DummyPredictor())

    report = evaluator.evaluate("The article claims a miracle cure was discovered overnight.")

    assert report.original_prediction == "fake"
    assert report.adversarial_results
    assert 0.0 <= report.robustness_score <= 1.0
