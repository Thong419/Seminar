"""Classifier tool - thin wrapper around existing Predictor"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.inference.predictor import Predictor, Prediction


@dataclass
class ClassifierResult:
    label: str
    confidence: float


class ClassifierTool:
    """Thin wrapper that reuses the existing `Predictor`.

    Usage:
        predictor = Predictor(model_dir=..., model_config=...)
        tool = ClassifierTool(predictor)
        out = tool.run(article_text)

    The tool intentionally does not duplicate model loading code.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def run(self, article_text: str) -> dict[str, Any]:
        """Run classifier and return standardized dict.

        Returns:
            {
                "label": str,
                "confidence": float,
            }
        """
        pred: Prediction = self.predictor.predict(article_text)
        # Predictor.Prediction defines `.label` and `.confidence`
        return {"label": str(pred.label), "confidence": float(pred.confidence)}
