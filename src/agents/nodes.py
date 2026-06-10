"""LangGraph node functions for the agentic workflow."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.analysis.evidence_analysis_agent import EvidenceAnalysisAgent
from src.agents.classification.classification_agent import ClassificationAgent
from src.agents.decision.decision_agent import DecisionAgent
from src.agents.explanation.explanation_agent import ExplanationAgent, ExplanationContext
from src.agents.retrieval.retrieval_agent import RetrievalAgent
from src.agents.state import AgentConfig, AgentState, Verdict


@dataclass(frozen=True, slots=True)
class AgentNodes:
    classifier: ClassificationAgent
    retriever: RetrievalAgent
    analyzer: EvidenceAnalysisAgent
    decider: DecisionAgent
    explainer: ExplanationAgent
    config: AgentConfig


def classify(state: AgentState, nodes: AgentNodes) -> AgentState:
    result = nodes.classifier.classify(state["article_text"])
    return {**state, "predicted_label": result.label, "confidence": result.confidence}


def confidence_check(state: AgentState, nodes: AgentNodes) -> str:
    if state.get("confidence", 0.0) >= nodes.config.confidence_threshold:
        return "make_decision"
    return "retrieve_evidence"


def retrieve_evidence(state: AgentState, nodes: AgentNodes) -> AgentState:
    bundle = nodes.retriever.retrieve(state["article_text"])
    return {**state, "retrieved_evidence": bundle.combined()}


def analyze_evidence(state: AgentState, nodes: AgentNodes) -> AgentState:
    analysis = nodes.analyzer.analyze(state["article_text"], state.get("retrieved_evidence", []))
    return {**state, "evidence_score": analysis.evidence_score}


def make_decision(state: AgentState, nodes: AgentNodes) -> AgentState:
    decision = nodes.decider.decide(
        predicted_label=state.get("predicted_label", "uncertain"),
        confidence=state.get("confidence", 0.0),
        evidence_score=state.get("evidence_score", 0.0),
        config=nodes.config,
    )
    return {**state, "final_decision": decision.final_decision}


def generate_explanation(state: AgentState, nodes: AgentNodes) -> AgentState:
    explanation_result = nodes.explainer.generate(
        ExplanationContext(
            article_text=state["article_text"],
            true_label=state.get("true_label"),
            predicted_label=state.get("predicted_label", "uncertain"),
            confidence=state.get("confidence", 0.0),
            evidence=state.get("retrieved_evidence", []),
            evidence_score=state.get("evidence_score", 0.0),
            final_decision=state.get("final_decision", Verdict.uncertain),
        )
    )
    report = explanation_result.report
    return {
        **state,
        "explanation": report.final_explanation,
        "trust_score": report.trust_score,
        "important_tokens": report.important_tokens,
        "evidence_summary": report.evidence_summary,
        "explanation_details": report.as_dict(),
    }
