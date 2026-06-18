"""Agent Controller - orchestrates multiple tools with conditional routing."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.classifier_tool import ClassifierTool
from src.agent.decision_tool import DecisionTool
from src.agent.evidence_tool import EvidenceTool
from src.agent.trace import AgentTrace, ToolTrace
from src.agents.state import EvidenceItem, ReviewState
from src.config.pipeline import ModelConfig
from src.explainability.explainer import ExplainabilityService
from src.inference.predictor import Predictor
from src.monitoring.agent_metrics import get_agent_metrics_tracker


logger = logging.getLogger(__name__)


@dataclass
class AgentControllerConfig:
    """Configuration for agent controller behavior."""
    confidence_threshold: float = 0.85
    enable_evidence_retrieval: bool = True
    enable_tracing: bool = True
    trace_artifact_dir: str = "artifacts/agent_traces"


@dataclass
class AgentResult:
    """Final agent output."""
    label: str
    confidence: float
    trust_score: float
    risk_level: str
    human_review_state: str
    conflict_flag: bool
    decision_reason: str
    explanation: str
    evidence_found: bool
    sources: list[dict[str, Any]]
    important_tokens: list[dict[str, float | str]]
    evidence_summary: str
    claim: str
    queries: list[str]
    support_score: float
    contradiction_score: float
    source_credibility_score: float
    evidence_quality_score: float
    trace: AgentTrace | None = None


class AgentController:
    """Main controller that orchestrates the agentic workflow.

    Flow:
    1. Classifier Tool: predict label/confidence from article
    2. Confidence check: if confidence >= threshold, skip to decision; else retrieve evidence
    3. Evidence Tool: retrieve external evidence for low-confidence cases
    4. Decision Tool: synthesize classifier + evidence → trust_score, decision_reason
    5. Explainability Service: generate explanation with token importance
    6. Return final result
    """

    def __init__(
        self,
        predictor: Predictor,
        model_config: ModelConfig | None = None,
        explainability_config_path: Path | str = "configs/explainability.yaml",
        agent_config: AgentControllerConfig | None = None,
        classifier: ClassifierTool | None = None,
        evidence_tool: EvidenceTool | None = None,
        decision_tool: DecisionTool | None = None,
        explainability_service: ExplainabilityService | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierTool(predictor)
        self.evidence_tool = evidence_tool or EvidenceTool()
        self.decision_tool = decision_tool or DecisionTool()
        self.explainability_service = explainability_service or ExplainabilityService(
            model_config=model_config,
            config_path=Path(explainability_config_path),
        )
        self.config = agent_config or AgentControllerConfig()
        self.predictor = predictor

    def run(
        self,
        article_text: str,
        request_id: str = "agent_request",
    ) -> AgentResult:
        """Execute the agent workflow.

        Returns:
            AgentResult with all outputs and optional execution trace.
        """
        start_time = time.time()
        trace = AgentTrace(request_id=request_id, article_text=article_text) if self.config.enable_tracing else None

        # Step 1: Classify
        logger.info("Step 1: Classify article")
        cls_start = time.time()
        classification = self.classifier.run(article_text)
        cls_time_ms = (time.time() - cls_start) * 1000
        if trace:
            trace.add_tool_trace(ToolTrace(
                tool_name="classifier",
                input_data={"article_text": article_text[:100]},
                output_data=classification,
                execution_time_ms=cls_time_ms,
            ))

        label = classification.get("label", "uncertain")
        confidence = float(classification.get("confidence", 0.0))

        # Step 2: Always retrieve evidence for agentic reasoning.
        evidence = {"evidence_found": False, "sources": [], "summary": "", "claim": "", "queries": []}
        if self.config.enable_evidence_retrieval:
            logger.info("Step 2: Retrieve evidence")
            ev_start = time.time()
            evidence = self.evidence_tool.run(article_text)
            ev_time_ms = (time.time() - ev_start) * 1000
            if trace:
                trace.add_tool_trace(
                    ToolTrace(
                        tool_name="evidence",
                        input_data={"article_text": article_text[:100]},
                        output_data={
                            "evidence_found": evidence.get("evidence_found"),
                            "num_sources": len(evidence.get("sources", [])),
                            "conflict_flag": evidence.get("conflict_flag"),
                            "support_score": evidence.get("support_score"),
                        },
                        execution_time_ms=ev_time_ms,
                    )
                )
        else:
            logger.info("Step 2: Evidence retrieval disabled by configuration")

        # Step 3: Decision
        logger.info("Step 3: Make decision")
        dec_start = time.time()
        decision = self.decision_tool.decide(classification, evidence)
        dec_time_ms = (time.time() - dec_start) * 1000
        if trace:
            trace.add_tool_trace(ToolTrace(
                tool_name="decision",
                input_data={"classification": classification, "evidence_found": evidence.get("evidence_found")},
                output_data=decision,
                execution_time_ms=dec_time_ms,
            ))

        trust_score = decision.get("trust_score", 0.5)
        risk_level = decision.get("risk_level", "unknown")
        decision_reason = decision.get("decision_reason", "")
        human_review_state = str(decision.get("human_review_state", ReviewState.uncertain.value))
        conflict_flag = bool(decision.get("conflict_flag", False))
        support_score = float(decision.get("support_score", 0.0))
        contradiction_score = float(decision.get("contradiction_score", 0.0))
        source_credibility_score = float(decision.get("source_credibility_score", 0.0))
        evidence_quality_score = float(decision.get("evidence_quality_score", 0.0))

        # Step 4: Generate explanation
        logger.info("Step 4: Generate explanation")
        exp_start = time.time()
        try:
            evidence_items = [EvidenceItem.model_validate(item) for item in evidence.get("sources", [])]
            aligned_evidence_score = max(support_score, contradiction_score, evidence_quality_score)
            explanation_report = self.explainability_service.explain(
                article_text=article_text,
                prediction=label,
                confidence=confidence,
                evidence=evidence_items,
                evidence_score=aligned_evidence_score,
                trust_score=trust_score,
            )
            evidence_sentence = (
                f" Evidence support={support_score:.2f}, contradiction={contradiction_score:.2f}, "
                f"source_credibility={source_credibility_score:.2f}, quality={evidence_quality_score:.2f}."
            )
            if human_review_state == ReviewState.uncertain.value:
                evidence_sentence += " Human review state is UNCERTAIN because the evidence is weak, conflicting, or insufficient."
            elif human_review_state == ReviewState.real.value:
                evidence_sentence += " Human review state is REAL because evidence and classifier align."
            elif human_review_state == ReviewState.fake.value:
                evidence_sentence += " Human review state is FAKE because evidence and classifier align."
            explanation_text = f"{explanation_report.final_explanation}{evidence_sentence}"
            important_tokens = [{"token": t.get("token"), "importance": t.get("importance")} for t in explanation_report.important_tokens]
        except Exception as e:
            logger.warning(f"Explainability service failed: {e}")
            explanation_text = (
                f"Prediction: {label} (confidence: {confidence:.2f}). Trust score: {trust_score:.2f}. "
                f"Evidence support={support_score:.2f}, contradiction={contradiction_score:.2f}, "
                f"human review state={human_review_state}."
            )
            important_tokens = []

        exp_time_ms = (time.time() - exp_start) * 1000
        if trace:
            trace.add_tool_trace(ToolTrace(
                tool_name="explainability",
                input_data={"prediction": label, "confidence": confidence},
                output_data={"explanation_length": len(explanation_text), "num_tokens": len(important_tokens), "human_review_state": human_review_state},
                execution_time_ms=exp_time_ms,
            ))

        # Finalize trace
        total_time_ms = (time.time() - start_time) * 1000
        if trace:
            trace.finalize(
                trust_score=trust_score,
                decision=label,
                exec_time_ms=total_time_ms,
            )
            trace_path = trace.save(self.config.trace_artifact_dir)
            logger.info(f"Trace saved to {trace_path}")

        # Build result
        result = AgentResult(
            label=label,
            confidence=confidence,
            trust_score=trust_score,
            risk_level=risk_level,
            human_review_state=human_review_state,
            conflict_flag=conflict_flag,
            decision_reason=decision_reason,
            explanation=explanation_text,
            evidence_found=evidence.get("evidence_found", False),
            sources=evidence.get("sources", []),
            important_tokens=important_tokens,
            evidence_summary=str(evidence.get("summary", "")),
            claim=str(evidence.get("claim", "")),
            queries=list(evidence.get("queries", [])),
            support_score=support_score,
            contradiction_score=contradiction_score,
            source_credibility_score=source_credibility_score,
            evidence_quality_score=evidence_quality_score,
            trace=trace,
        )
        logger.info(f"Agent completed in {total_time_ms:.2f}ms: {label} (confidence={confidence:.2f}, trust={trust_score:.2f})")
        get_agent_metrics_tracker().record(
            evidence_found=result.evidence_found,
            conflict_flag=result.conflict_flag,
            human_review_state=result.human_review_state,
            confidence=result.confidence,
            trust_score=result.trust_score,
            response_time_ms=total_time_ms,
            evidence_source_count=len(result.sources),
        )
        return result
