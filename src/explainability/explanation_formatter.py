"""Unified explanation formatting and trust scoring."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

import pandas as pd
import yaml

from src.agent.state import EvidenceItem, Verdict
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
    """Format a structured explanation with explicit support/refute/uncertainty reasoning.

    Improvements over previous version:
    - Explicitly states which sources support and which refute.
    - Explains why the result is UNCERTAIN when applicable.
    - Does not combine irrelevant neutral sources into the support narrative.
    """
    if trust_score is None:
        trust_score = calculate_trust_score(confidence, evidence_score, source_trust, weights) / 100.0
    trust_score = max(0.0, min(1.0, float(trust_score)))

    evidence_summary = build_evidence_summary(evidence)

    # Partition evidence by stance
    support_items = [e for e in evidence if str(getattr(e, "stance", "")).lower() == "support"]
    refute_items = [e for e in evidence if str(getattr(e, "stance", "")).lower() in {"refute", "contradict"}]
    neutral_items = [e for e in evidence if str(getattr(e, "stance", "")).lower() == "neutral"]

    # Build the explanation narrative
    parts: list[str] = [
        f"The article was classified as {prediction.upper()} with confidence {confidence:.1%}."
    ]

    if support_items:
        titles = ", ".join(f"\"{e.title}\"" for e in support_items[:3])
        sources = ", ".join({e.source for e in support_items[:3]})
        parts.append(
            f"Evidence supporting this classification: {titles} "
            f"(from {sources}). "
            f"These sources confirm or align with the claim."
        )

    if refute_items:
        titles = ", ".join(f"\"{e.title}\"" for e in refute_items[:3])
        sources = ", ".join({e.source for e in refute_items[:3]})
        parts.append(
            f"Evidence refuting or contradicting the claim: {titles} "
            f"(from {sources}). "
            f"These sources dispute or debunk the core assertion."
        )

    if neutral_items and not support_items and not refute_items:
        titles = ", ".join(f"\"{e.title}\"" for e in neutral_items[:3])
        parts.append(
            f"Retrieved sources ({titles}) discuss the topic but do not explicitly "
            f"confirm or refute the specific claim. "
            f"They provide background context only."
        )

    # Uncertainty explanation
    if prediction.lower() == "uncertain" or (trust_score < 0.55):
        if support_items and refute_items:
            parts.append(
                "The result is UNCERTAIN because evidence is mixed: "
                f"{len(support_items)} source(s) support and {len(refute_items)} source(s) refute the claim. "
                "Human review is recommended."
            )
        elif not support_items and not refute_items:
            parts.append(
                "The result is UNCERTAIN because no evidence explicitly confirms or refutes the claim. "
                "Retrieved sources are topically related but do not address the specific assertion."
            )
        elif confidence < 0.72:
            parts.append(
                f"The result is UNCERTAIN because classifier confidence ({confidence:.1%}) "
                "is below the required threshold for an automatic decision."
            )

    # Token signals
    token_signal = ", ".join(str(item.get("token", "")) for item in important_tokens[:5])
    if token_signal:
        parts.append(f"Key linguistic signals: {token_signal}.")

    parts.append(f"Overall trust score: {trust_score:.1%}.")

    final_explanation = " ".join(parts)

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
