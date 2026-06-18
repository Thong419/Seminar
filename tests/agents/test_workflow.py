from pathlib import Path

from src.agent.analysis.evidence_analysis_agent import EvidenceAnalysisAgent
from src.agent.classification.classification_agent import ClassificationAgent, ClassificationResult
from src.agent.decision.decision_agent import DecisionAgent
from src.agent.explanation.explanation_agent import ExplanationAgent
from src.agent.retrieval.retrieval_agent import RetrievalAgent
from src.agent.state import AgentConfig
from src.agent.workflow import AgenticWorkflow
from src.config.pipeline import ModelConfig
from src.retrieval.document_fetcher import EvidenceDocument
from src.retrieval.search_client import SearchResult


class DummyPredictor:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence

    def predict(self, text: str) -> ClassificationResult:
        return ClassificationResult(label=self.label, confidence=self.confidence)


class DummySearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="Reuters verifies claim",
                url="https://example.com/reuters",
                source="Reuters",
                snippet="Reuters verifies the claim.",
                provider_relevance=0.95,
            )
        ]


class DummyFetcher:
    def fetch(self, result: SearchResult) -> EvidenceDocument:
        return EvidenceDocument(
            title=result.title,
            url=result.url,
            source=result.source,
            content=result.snippet,
            trust_score=1.0,
            relevance_score=result.provider_relevance,
        )


class DummySHAPExplainer:
    def explain(self, text: str, top_k_tokens: int = 8):
        class Result:
            tokens = ["scientists", "discover", "cancer", "cure"]
            values = [0.2, 0.4, 0.6, 0.8]

        return Result()


def test_workflow_runs_and_returns_final_explanation() -> None:
    predictor = DummyPredictor(label="fake", confidence=0.82)
    classifier = ClassificationAgent(
        predictor=predictor,
        model_config=ModelConfig(model_output_dir=Path("models/roberta")),
    )
    retriever = RetrievalAgent(
        retrieval_config_path=Path("configs/retrieval.yaml"),
        search_client=DummySearchClient(),
        document_fetcher=DummyFetcher(),
    )
    workflow = AgenticWorkflow(
        classifier=classifier,
        retriever=retriever,
        analyzer=EvidenceAnalysisAgent(),
        decider=DecisionAgent(),
        explainer=ExplanationAgent(shap_explainer=DummySHAPExplainer()),
        config=AgentConfig(confidence_threshold=0.90),
    )

    result = workflow.run("This article contains a claim about an event.")

    assert result["predicted_label"] == "fake"
    assert result["confidence"] == 0.82
    assert "Final verdict" in result["explanation"]
