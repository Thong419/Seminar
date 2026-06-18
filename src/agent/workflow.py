"""LangGraph workflow for the agentic fake-news detection system."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from src.agent.analysis.evidence_analysis_agent import EvidenceAnalysisAgent
from src.agent.classification.classification_agent import ClassificationAgent
from src.agent.decision.decision_agent import DecisionAgent
from src.agent.explanation.explanation_agent import ExplanationAgent
from src.agent.nodes import (
    AgentNodes,
    analyze_evidence,
    classify,
    confidence_check,
    generate_explanation,
    make_decision,
    retrieve_evidence,
)
from src.agent.retrieval.retrieval_agent import RetrievalAgent
from src.agent.state import AgentConfig, AgentState


@dataclass(frozen=True, slots=True)
class AgenticWorkflow:
    classifier: ClassificationAgent
    retriever: RetrievalAgent
    analyzer: EvidenceAnalysisAgent
    decider: DecisionAgent
    explainer: ExplanationAgent
    config: AgentConfig

    def _nodes(self) -> AgentNodes:
        return AgentNodes(
            classifier=self.classifier,
            retriever=self.retriever,
            analyzer=self.analyzer,
            decider=self.decider,
            explainer=self.explainer,
            config=self.config,
        )

    def build_graph(self) -> StateGraph[AgentState]:
        graph = StateGraph(AgentState)
        graph.add_node("classify", lambda state: classify(state, self._nodes()))
        graph.add_node("retrieve_evidence", lambda state: retrieve_evidence(state, self._nodes()))
        graph.add_node("analyze_evidence", lambda state: analyze_evidence(state, self._nodes()))
        graph.add_node("make_decision", lambda state: make_decision(state, self._nodes()))
        graph.add_node("generate_explanation", lambda state: generate_explanation(state, self._nodes()))

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: confidence_check(state, self._nodes()),
            {
                "make_decision": "make_decision",
                "retrieve_evidence": "retrieve_evidence",
            },
        )
        graph.add_edge("retrieve_evidence", "analyze_evidence")
        graph.add_edge("analyze_evidence", "make_decision")
        graph.add_edge("make_decision", "generate_explanation")
        graph.add_edge("generate_explanation", END)
        return graph

    def run(self, article_text: str) -> AgentState:
        compiled = self.build_graph().compile()
        return compiled.invoke({"article_text": article_text})
