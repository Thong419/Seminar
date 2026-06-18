"""Unified explanation formatting and trust scoring."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

import pandas as pd
import yaml

from src.agents.state import EvidenceItem, Verdict
from src.explainability.token_importance import save_token_importance


@dataclass(frozen=True, slots=True)
class TrustScoreWeights:
    confidence: float = 0.5
    evidence: float = 0.3
    source_trust: float = 0.2


@dataclass(frozen=True, slots=True)
class UnifiedExplanation:
    prediction: str
    confidence: float
    trust_score: float
    important_tokens: list[dict[str, str | float]]
    evidence_summary: str
    final_explanation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplanationFormatterConfig:
    artifact_dir: Path = Path("artifacts/explainability")
    top_k_tokens: int = 8
    max_misclassified_samples: int = 200
    weights: TrustScoreWeights = TrustScoreWeights()
    source_trust_scores: dict[str, float] | None = None


def load_explainability_config(path: Path) -> ExplanationFormatterConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Explainability configuration must be a mapping.")

    weights = data.get("trust_score_weights", {})
    if not isinstance(weights, dict):
        raise ValueError("trust_score_weights must be a mapping.")

    return ExplanationFormatterConfig(
        artifact_dir=Path(data.get("artifact_dir", ExplanationFormatterConfig.artifact_dir)),
        top_k_tokens=int(data.get("top_k_tokens", ExplanationFormatterConfig.top_k_tokens)),
        max_misclassified_samples=int(
            data.get("max_misclassified_samples", ExplanationFormatterConfig.max_misclassified_samples)
        ),
        weights=TrustScoreWeights(
            confidence=float(weights.get("confidence", TrustScoreWeights.confidence)),
            evidence=float(weights.get("evidence", TrustScoreWeights.evidence)),
            source_trust=float(weights.get("source_trust", TrustScoreWeights.source_trust)),
        ),
        source_trust_scores={str(key): float(value) for key, value in (data.get("source_trust_scores", {}) or {}).items()},
    )


def calculate_trust_score(
    confidence: float,
    evidence_score: float,
    source_trust: float,
    weights: TrustScoreWeights,
) -> int:
    raw_score = (
        weights.confidence * confidence
        + weights.evidence * evidence_score
        + weights.source_trust * source_trust
    )
    normalized = max(0.0, min(1.0, raw_score)) * 100.0
    return int(round(normalized))


def build_evidence_summary(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "No supporting evidence was retrieved."

    sources = sorted({item.source for item in evidence})
    top_titles = ", ".join(item.title for item in evidence[:3])
    return (
        f"Retrieved {len(evidence)} evidence items from {len(sources)} sources. "
        f"Top sources: {', '.join(sources[:3])}. Top evidence: {top_titles}."
    )


def format_explanation(
    prediction: str,
    confidence: float,
    evidence_score: float,
    source_trust: float,
    important_tokens: list[dict[str, str | float]],
    evidence: list[EvidenceItem],
    weights: TrustScoreWeights = TrustScoreWeights(),
    trust_score: float | None = None,
) -> UnifiedExplanation:
    if trust_score is None:
        trust_score = calculate_trust_score(confidence, evidence_score, source_trust, weights) / 100.0
    trust_score = max(0.0, min(1.0, float(trust_score)))
    evidence_summary = build_evidence_summary(evidence)
    final_explanation = (
        f"The article was classified as {prediction} with confidence {confidence:.2f}. "
        f"Trust score: {trust_score:.2%}. "
        f"{evidence_summary} "
        f"The strongest token signals were {', '.join(item['token'] for item in important_tokens[:5]) or 'not available'}."
    )
    return UnifiedExplanation(
        prediction=prediction,
        confidence=confidence,
        trust_score=trust_score,
        important_tokens=important_tokens,
        evidence_summary=evidence_summary,
        final_explanation=final_explanation,
    )


def save_explanation_artifacts(explanation: UnifiedExplanation, artifact_dir: Path) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shap_path = artifact_dir / "shap_values.json"
    token_path = artifact_dir / "token_importance.json"
    with shap_path.open("w", encoding="utf-8") as handle:
        json.dump(explanation.as_dict(), handle, indent=2)
    save_token_importance(explanation.important_tokens, token_path)
    return shap_path, token_path


def save_misclassified_samples(
    records: list[dict[str, Any]],
    output_path: Path,
    max_rows: int = ExplanationFormatterConfig.max_misclassified_samples,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if not frame.empty and len(frame) > 0:
        frame = frame.head(max_rows)
    frame.to_csv(output_path, index=False)
    return output_path
