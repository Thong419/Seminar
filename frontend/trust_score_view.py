"""Trust score rendering helpers for the Streamlit frontend."""

from __future__ import annotations


def classify_trust_score(trust_score: int) -> str:
    if trust_score >= 80:
        return "High Trust"
    if trust_score >= 60:
        return "Medium Trust"
    return "Low Trust"


def trust_progress(trust_score: int) -> float:
    return max(0.0, min(1.0, trust_score / 100.0))
