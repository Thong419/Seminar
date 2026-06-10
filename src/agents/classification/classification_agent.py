"""Classification agent that delegates to the existing predictor interface."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.pipeline import ModelConfig
from src.inference.predictor import Predictor


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    label: str
    confidence: float


class ClassificationAgent:
    def __init__(self, predictor: Predictor, model_config: ModelConfig) -> None:
        self.predictor = predictor
        self.model_config = model_config

    def classify(self, text: str) -> ClassificationResult:
        prediction = self.predictor.predict(text)
        return ClassificationResult(label=prediction.label, confidence=prediction.confidence)
