"""Inference service skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    text: str


@dataclass(frozen=True, slots=True)
class PredictionResponse:
    prediction: str
    confidence: float
    explanation: str
    evidence: list[dict[str, str]]


class InferenceService:
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raise NotImplementedError("Inference is not implemented yet.")