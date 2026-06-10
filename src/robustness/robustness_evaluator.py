"""Robustness evaluation and reporting for model perturbation tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from statistics import mean
from typing import Protocol, Sequence

from src.robustness.adversarial_tests import AdversarialCase, build_adversarial_cases


class PredictorLike(Protocol):
    def predict(self, text: str):
        ...


@dataclass(frozen=True, slots=True)
class AdversarialResult:
    name: str
    prediction: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    original_prediction: str
    original_confidence: float
    adversarial_results: list[AdversarialResult]
    prediction_flip_rate: float
    average_confidence_drop: float
    worst_case_confidence_drop: float
    robustness_score: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RobustnessEvaluator:
    def __init__(self, predictor: PredictorLike) -> None:
        self.predictor = predictor

    def evaluate(self, article_text: str) -> RobustnessReport:
        cases = build_adversarial_cases(article_text)
        original = self.predictor.predict(article_text)
        adversarial_results: list[AdversarialResult] = []
        drops: list[float] = []
        flips = 0

        for case in cases[1:]:
            prediction = self.predictor.predict(case.text)
            adversarial_results.append(
                AdversarialResult(
                    name=case.name,
                    prediction=str(getattr(prediction, "label", "uncertain")).lower(),
                    confidence=float(getattr(prediction, "confidence", 0.0)),
                )
            )
            original_label = str(getattr(original, "label", "uncertain")).lower()
            if adversarial_results[-1].prediction != original_label:
                flips += 1
            drops.append(max(0.0, float(getattr(original, "confidence", 0.0)) - adversarial_results[-1].confidence))

        prediction_flip_rate = flips / max(len(adversarial_results), 1)
        average_confidence_drop = mean(drops) if drops else 0.0
        worst_case_confidence_drop = max(drops) if drops else 0.0
        robustness_score = max(0.0, min(1.0, 1.0 - ((prediction_flip_rate + average_confidence_drop + worst_case_confidence_drop) / 3.0)))

        return RobustnessReport(
            original_prediction=str(getattr(original, "label", "uncertain")).lower(),
            original_confidence=float(getattr(original, "confidence", 0.0)),
            adversarial_results=adversarial_results,
            prediction_flip_rate=round(prediction_flip_rate, 4),
            average_confidence_drop=round(average_confidence_drop, 4),
            worst_case_confidence_drop=round(worst_case_confidence_drop, 4),
            robustness_score=round(robustness_score, 4),
        )

    def save_report(self, report: RobustnessReport, output_path: Path = Path("artifacts/robustness/robustness_report.json")) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return output_path
