"""Decision tool: combines classifier and evidence outputs into a trust score and risk level.

Improvement over previous version:
    - Uses per-source stance labels (support/refute/neutral) computed by StanceDetector,
      rather than keyword scanning in the summary text.
    - Decision reason explicitly explains what evidence supports, what refutes, and why
      the result is uncertain when applicable.
    - Conflict detection is based on stance counts, not fragile text heuristics.
"""
from __future__ import annotations

from typing import Any

from src.agent.state import ReviewState


class DecisionTool:
    """Synthesise classification + evidence into a trust score, risk, and review state."""

    def decide(self, classification: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        """Produce trust_score, decision_reason, risk_level, and review state.

        Args:
            classification: {label, confidence}
            evidence: {evidence_found, sources, summary, support_score,
                       contradiction_score, source_credibility_score,
                       evidence_quality_score, conflict_flag}

        Returns:
            Decision dict with trust_score, risk_level, human_review_state,
            decision_reason, conflict_flag, and score fields.
        """
        label = str(classification.get("label", "uncertain")).lower()
        confidence = float(classification.get("confidence", 0.0))

        evidence_found = bool(evidence.get("evidence_found"))
        sources = evidence.get("sources") or []
        evidence_quality_score = float(evidence.get("evidence_quality_score", 0.0))
        source_credibility_score = float(evidence.get("source_credibility_score", 0.0))
        support_score = float(evidence.get("support_score", 0.0))
        contradiction_score = float(evidence.get("contradiction_score", 0.0))

        # ── Count stances from individual sources (most reliable signal) ──────
        support_count = sum(1 for s in sources if str(s.get("stance", "")).lower() == "support")
        refute_count = sum(
            1 for s in sources if str(s.get("stance", "")).lower() in {"refute", "contradict"}
        )
        neutral_count = len(sources) - support_count - refute_count

        # ── Trust score formula ────────────────────────────────────────────────
        # When evidence is found, align with classifier direction.
        if evidence_found:
            if label == "fake":
                # For FAKE claims: refuting evidence (contra the claim) aligns with classifier
                aligned_score = contradiction_score
                counter_score = support_score
            elif label == "real":
                # For REAL claims: supporting evidence aligns with classifier
                aligned_score = support_score
                counter_score = contradiction_score
            else:
                aligned_score = max(support_score, contradiction_score)
                counter_score = min(support_score, contradiction_score)

            trust_score = (
                0.40 * confidence
                + 0.30 * aligned_score
                + 0.20 * source_credibility_score
                - 0.10 * counter_score
            )
        else:
            trust_score = 0.85 * confidence

        trust_score = max(0.0, min(1.0, trust_score))

        # ── Risk level ─────────────────────────────────────────────────────────
        if trust_score >= 0.75 and evidence_quality_score >= 0.45:
            risk_level = "low"
        elif trust_score >= 0.50:
            risk_level = "medium"
        else:
            risk_level = "high"

        # ── Conflict detection ─────────────────────────────────────────────────
        # Use both per-source stance counts AND aggregate score signals.
        # This handles cases where the FakeEvidenceTool injects aggregate scores
        # that don't match per-source stances (e.g., in integration tests).
        # Day6 rules:
        # A) If classifier=fake but support is strong and contradiction is weak -> UNCERTAIN
        # B) If classifier=real but refute is strong and support is weak -> UNCERTAIN
        # C) Only output REAL/FAKE when classifier and evidence direction are aligned.
        strong_support = support_score >= 0.65
        strong_refute = contradiction_score >= 0.65
        weak_support = support_score <= 0.25
        weak_refute = contradiction_score <= 0.25

        rule_a_conflict = label == "fake" and strong_support and weak_refute
        rule_b_conflict = label == "real" and strong_refute and weak_support

        evidence_conflict_by_count = evidence_found and evidence_quality_score >= 0.30 and (
            (label == "fake" and support_count > refute_count) or
            (label == "real" and refute_count > support_count)
        )
        # Aggregate score signal: e.g., support_score=0.83 when label=fake is a conflict
        evidence_conflict_by_score = evidence_found and evidence_quality_score >= 0.30 and (
            (label == "fake" and support_score > contradiction_score + 0.15) or
            (label == "real" and contradiction_score > support_score + 0.15)
        )
        mixed_stance = evidence_found and support_count >= 1 and refute_count >= 1

        if rule_a_conflict or rule_b_conflict or evidence_conflict_by_count or evidence_conflict_by_score or mixed_stance:
            # Evidence conflicts with classifier decision
            conflict_flag = True
        else:
            conflict_flag = False

        # ── Human review state ─────────────────────────────────────────────────
        insufficient_confidence = confidence < 0.72
        all_neutral = evidence_found and support_count == 0 and refute_count == 0
        weak_or_sparse_evidence = evidence_found and (evidence_quality_score < 0.40)
        aligned_real = label == "real" and support_score > contradiction_score + 0.12
        aligned_fake = label == "fake" and contradiction_score > support_score + 0.12
        aligned_direction = aligned_real or aligned_fake
        if (
            not evidence_found
            or evidence_quality_score < 0.30
            or insufficient_confidence
            or conflict_flag
            or all_neutral
            or weak_or_sparse_evidence
            or not aligned_direction
        ):
            human_review_state = ReviewState.uncertain
        elif label == "fake":
            human_review_state = ReviewState.fake
        elif label == "real":
            human_review_state = ReviewState.real
        else:
            human_review_state = ReviewState.uncertain

        # ── Decision reason ────────────────────────────────────────────────────
        reason = self._build_reason(
            label=label,
            confidence=confidence,
            evidence_found=evidence_found,
            support_count=support_count,
            refute_count=refute_count,
            neutral_count=neutral_count,
            support_score=support_score,
            contradiction_score=contradiction_score,
            source_credibility_score=source_credibility_score,
            evidence_quality_score=evidence_quality_score,
            conflict_flag=conflict_flag,
            insufficient_confidence=insufficient_confidence,
            all_neutral=all_neutral,
            rule_a_conflict=rule_a_conflict,
            rule_b_conflict=rule_b_conflict,
            sources=sources,
        )

        return {
            "trust_score": float(trust_score),
            "decision_reason": reason,
            "risk_level": risk_level,
            "support_count": support_count,
            "refute_count": refute_count,
            "neutral_count": neutral_count,
            "support_score": float(support_score),
            "contradiction_score": float(contradiction_score),
            "source_credibility_score": float(source_credibility_score),
            "evidence_quality_score": float(evidence_quality_score),
            "conflict_flag": conflict_flag,
            "human_review_state": human_review_state.value,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_reason(
        self,
        label: str,
        confidence: float,
        evidence_found: bool,
        support_count: int,
        refute_count: int,
        neutral_count: int,
        support_score: float,
        contradiction_score: float,
        source_credibility_score: float,
        evidence_quality_score: float,
        conflict_flag: bool,
        insufficient_confidence: bool,
        all_neutral: bool,
        rule_a_conflict: bool,
        rule_b_conflict: bool,
        sources: list[dict[str, Any]],
    ) -> str:
        """Build a human-readable explanation of why the decision was made."""
        total = support_count + refute_count + neutral_count

        if not evidence_found:
            return (
                f"No external evidence found. Relying solely on classifier "
                f"(confidence={confidence:.2f}). Review state forced to UNCERTAIN."
            )

        # Collect supporting source titles
        support_titles = [
            str(s.get("title", "unknown"))
            for s in sources
            if str(s.get("stance", "")).lower() == "support"
        ][:3]
        refute_titles = [
            str(s.get("title", "unknown"))
            for s in sources
            if str(s.get("stance", "")).lower() in {"refute", "contradict"}
        ][:3]
        neutral_titles = [
            str(s.get("title", "unknown"))
            for s in sources
            if str(s.get("stance", "")).lower() == "neutral"
        ][:2]

        parts: list[str] = [
            f"Classifier: {label.upper()} (confidence={confidence:.2f}). "
            f"Evidence: {total} sources retrieved "
            f"(support={support_count}, refute={refute_count}, neutral={neutral_count}). "
            f"Source credibility={source_credibility_score:.2f}."
        ]

        if support_titles:
            parts.append(
                f"Sources supporting the claim: {', '.join(support_titles)}."
            )
        if refute_titles:
            parts.append(
                f"Sources refuting the claim: {', '.join(refute_titles)}."
            )
        if neutral_titles and all_neutral:
            parts.append(
                f"Sources retrieved but not addressing the specific claim: "
                f"{', '.join(neutral_titles)}. These provide background context only."
            )

        if conflict_flag:
            if rule_a_conflict:
                parts.append(
                    "Conflict Rule A triggered: classifier predicts FAKE but evidence strongly supports the claim "
                    "with weak contradiction. Human review required."
                )
            elif rule_b_conflict:
                parts.append(
                    "Conflict Rule B triggered: classifier predicts REAL but evidence strongly refutes the claim "
                    "with weak support. Human review required."
                )
            else:
                parts.append(
                    f"Evidence conflicts with classifier label. "
                    f"Evidence stances disagree with the classifier label. "
                    f"Human review required (support_score={support_score:.2f}, "
                    f"contradiction_score={contradiction_score:.2f})."
                )
        elif all_neutral:
            parts.append(
                "All retrieved sources are NEUTRAL — they discuss the topic but do not "
                "explicitly confirm or refute the specific claim. "
                "Review state set to UNCERTAIN."
            )
        elif insufficient_confidence:
            parts.append(
                f"Classifier confidence ({confidence:.2f}) is below the required threshold. "
                "Review state set to UNCERTAIN."
            )
        elif evidence_quality_score < 0.30:
            parts.append(
                f"Evidence quality is low ({evidence_quality_score:.2f}). "
                "Insufficient signal for a confident decision."
            )
        else:
            if label == "fake" and refute_count >= support_count:
                parts.append(
                    "Evidence aligns with FAKE classification: refuting sources "
                    "corroborate the classifier output."
                )
            elif label == "real" and support_count >= refute_count:
                parts.append(
                    "Evidence aligns with REAL classification: supporting sources "
                    "corroborate the classifier output."
                )

        return " ".join(parts)
