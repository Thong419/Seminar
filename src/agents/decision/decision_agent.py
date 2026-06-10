"""Decision agent that converts classifier and evidence signals into a verdict."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.state import AgentConfig, Verdict


@dataclass(frozen=True, slots=True)
class DecisionResult:
    final_decision: Verdict
    decision_reason: str


class DecisionAgent:
    def decide(
        self,
        predicted_label: str,
        confidence: float,
        evidence_score: float,
        config: AgentConfig,
    ) -> DecisionResult:
        if confidence >= config.confidence_threshold and evidence_score >= 0.75:
            return DecisionResult(
                final_decision=Verdict.confirmed_real if predicted_label == "real" else Verdict.confirmed_fake,
                decision_reason="High confidence and strong evidence alignment.",
            )

        if confidence >= config.confidence_threshold and evidence_score < 0.75:
            return DecisionResult(
                final_decision=Verdict.likely_real if predicted_label == "real" else Verdict.likely_fake,
                decision_reason="High classifier confidence but weaker evidence support.",
            )

        if confidence < config.confidence_threshold and evidence_score >= 0.6:
            return DecisionResult(
                final_decision=Verdict.likely_real if predicted_label == "real" else Verdict.likely_fake,
                decision_reason="Evidence supports the classifier prediction despite lower confidence.",
            )

        return DecisionResult(
            final_decision=Verdict.uncertain,
            decision_reason="Signals are mixed or weak, so the result remains uncertain.",
        )
