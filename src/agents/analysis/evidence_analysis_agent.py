"""Evidence analysis agent for relevance, agreement, and contradiction scoring."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.state import EvidenceItem


@dataclass(frozen=True, slots=True)
class EvidenceAnalysisResult:
    evidence_score: float
    agreement_score: float
    contradiction_score: float


class EvidenceAnalysisAgent:
    def analyze(self, article_text: str, evidence: list[EvidenceItem]) -> EvidenceAnalysisResult:
        if not evidence:
            return EvidenceAnalysisResult(evidence_score=0.0, agreement_score=0.0, contradiction_score=0.0)

        article_tokens = set(article_text.lower().split())
        relevance_scores: list[float] = []
        agreement_scores: list[float] = []
        contradiction_scores: list[float] = []

        for item in evidence:
            evidence_tokens = set(item.content.lower().split()) | set(item.title.lower().split())
            overlap = len(article_tokens & evidence_tokens)
            union = max(len(article_tokens | evidence_tokens), 1)
            lexical_overlap = overlap / union
            relevance_scores.append(0.5 * item.relevance_score + 0.5 * lexical_overlap)

            article_polarity = _claim_polarity(article_text)
            evidence_polarity = _claim_polarity(f"{item.title} {item.content}")
            if article_polarity == evidence_polarity:
                agreement_scores.append(1.0)
                contradiction_scores.append(0.0)
            elif article_polarity == 0 or evidence_polarity == 0:
                agreement_scores.append(0.5)
                contradiction_scores.append(0.5)
            else:
                agreement_scores.append(0.0)
                contradiction_scores.append(1.0)

        relevance = sum(relevance_scores) / len(relevance_scores)
        agreement = sum(agreement_scores) / len(agreement_scores)
        contradiction = sum(contradiction_scores) / len(contradiction_scores)
        evidence_score = max(0.0, min(1.0, (0.65 * relevance) + (0.25 * agreement) - (0.2 * contradiction)))

        return EvidenceAnalysisResult(
            evidence_score=evidence_score,
            agreement_score=agreement,
            contradiction_score=contradiction,
        )


def _claim_polarity(text: str) -> int:
    lowered = text.lower()
    if any(token in lowered for token in ("fake", "hoax", "false", "misleading", "untrue")):
        return -1
    if any(token in lowered for token in ("real", "true", "verified", "confirmed", "authentic")):
        return 1
    return 0
