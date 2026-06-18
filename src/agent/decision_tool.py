"""Decision tool: combines classifier and evidence outputs into a trust score and risk level."""
from __future__ import annotations

from typing import Any
import re

from src.agent.state import ReviewState


class DecisionTool:
    def __init__(self) -> None:
        # simple keywords for lightweight support/contradict detection
        self._support_keywords = [r"associate", r"linked", r"reduces", r"benefit", r"associated with", r"verified", r"confirmed", r"supports", r"reports"]
        self._contradict_keywords = [r"no evidence", r"not", r"contradict", r"refute", r"fails to", r"no support", r"false", r"hoax", r"misinformation", r"debunk"]

    def _count_matches(self, text: str, patterns: list[str]) -> int:
        text_l = (text or "").lower()
        count = 0
        for p in patterns:
            if re.search(p, text_l):
                count += 1
        return count

    def decide(self, classification: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        """Produce trust_score, decision_reason, risk_level, and review state.

        classification: {label, confidence}
        evidence: {evidence_found, sources: [ {title, url, snippet, relevance} ], summary}
        """
        label = str(classification.get("label", "uncertain")).lower()
        confidence = float(classification.get("confidence", 0.0))

        evidence_found = bool(evidence.get("evidence_found"))
        sources = evidence.get("sources") or []
        summary = evidence.get("summary") or ""
        evidence_quality_score = float(evidence.get("evidence_quality_score", 0.0))
        source_credibility_score = float(evidence.get("source_credibility_score", 0.0))
        support_score = float(evidence.get("support_score", 0.0))
        contradiction_score = float(evidence.get("contradiction_score", 0.0))

        support_count = 0
        contradict_count = 0
        # check summary
        support_count += self._count_matches(summary, self._support_keywords)
        contradict_count += self._count_matches(summary, self._contradict_keywords)
        # check sources snippets
        for s in sources:
            snip = (s.get("snippet") or "")
            support_count += self._count_matches(snip, self._support_keywords)
            contradict_count += self._count_matches(snip, self._contradict_keywords)

        # Fallback to aggregated scores when the evidence tool already computed them.
        if support_score == 0.0 and contradiction_score == 0.0 and evidence_found:
            total = max(support_count + contradict_count, 1)
            support_score = support_count / total
            contradiction_score = contradict_count / total

        aligned_score = contradiction_score if label == "fake" else support_score if label == "real" else max(support_score, contradiction_score)
        counter_score = support_score if label == "fake" else contradiction_score if label == "real" else min(support_score, contradiction_score)

        if evidence_found:
            trust_score = (
                0.45 * confidence
                + 0.25 * aligned_score
                + 0.2 * source_credibility_score
                - 0.1 * counter_score
            )
        else:
            trust_score = 0.85 * confidence

        # clamp
        trust_score = max(0.0, min(1.0, trust_score))

        if trust_score >= 0.8 and evidence_quality_score >= 0.45:
            risk_level = "low"
        elif trust_score >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        conflict_flag = evidence_found and evidence_quality_score >= 0.35 and counter_score > aligned_score + 0.08
        insufficient_confidence = confidence < 0.72
        if not evidence_found or evidence_quality_score < 0.35 or insufficient_confidence or conflict_flag:
            human_review_state = ReviewState.uncertain
        elif label == "fake":
            human_review_state = ReviewState.fake
        elif label == "real":
            human_review_state = ReviewState.real
        else:
            human_review_state = ReviewState.uncertain

        if not evidence_found:
            reason = (
                f"No external evidence found; relying on classifier confidence={confidence:.2f}. "
                f"Human review state forced to UNCERTAIN."
            )
        elif conflict_flag:
            reason = (
                f"Evidence conflicts with classifier. confidence={confidence:.2f}, support_score={support_score:.2f}, "
                f"contradiction_score={contradiction_score:.2f}, source_credibility={source_credibility_score:.2f}. "
                f"Human review state set to UNCERTAIN."
            )
        elif evidence_quality_score < 0.35:
            reason = (
                f"Evidence quality is low ({evidence_quality_score:.2f}). confidence={confidence:.2f}, "
                f"support_score={support_score:.2f}, contradiction_score={contradiction_score:.2f}. "
                f"Human review state set to UNCERTAIN."
            )
        elif insufficient_confidence:
            reason = (
                f"Classifier confidence is insufficient ({confidence:.2f}) even with evidence. "
                f"support_score={support_score:.2f}, contradiction_score={contradiction_score:.2f}."
            )
        else:
            reason = (
                f"Evidence found ({len(sources)} sources). confidence={confidence:.2f}, support_score={support_score:.2f}, "
                f"contradiction_score={contradiction_score:.2f}, source_credibility={source_credibility_score:.2f}."
            )

        return {
            "trust_score": float(trust_score),
            "decision_reason": reason,
            "risk_level": risk_level,
            "support_count": support_count,
            "contradict_count": contradict_count,
            "support_score": float(support_score),
            "contradiction_score": float(contradiction_score),
            "source_credibility_score": float(source_credibility_score),
            "evidence_quality_score": float(evidence_quality_score),
            "conflict_flag": conflict_flag,
            "human_review_state": human_review_state.value,
        }
