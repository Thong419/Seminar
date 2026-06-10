"""Prediction summary rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionCardData:
    prediction: str
    confidence: float
    trust_score: int
    final_decision: str


def prediction_badge_label(prediction: str) -> str:
    normalized = prediction.strip().lower().replace("_", " ")
    mapping = {
        "real": "REAL",
        "fake": "FAKE",
        "likely real": "LIKELY REAL",
        "likely fake": "LIKELY FAKE",
        "uncertain": "UNCERTAIN",
        "confirmed real": "REAL",
        "confirmed fake": "FAKE",
    }
    return mapping.get(normalized, normalized.upper())


def prediction_badge_color(prediction: str) -> str:
    normalized = prediction.strip().lower().replace("_", " ")
    if normalized in {"real", "confirmed real", "likely real"}:
        return "#0f766e"
    if normalized in {"fake", "confirmed fake", "likely fake"}:
        return "#b91c1c"
    return "#a16207"
