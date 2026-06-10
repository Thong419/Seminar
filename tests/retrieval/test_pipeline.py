from pathlib import Path

from src.retrieval.document_fetcher import EvidenceDocument
from src.retrieval.pipeline import EvidenceRetrievalPipeline
from src.retrieval.search_client import SearchResult


class DummySearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="Reuters verifies claim",
                url="https://example.com/reuters",
                source="Reuters",
                snippet="Reuters verifies the article claim.",
                provider_relevance=0.9,
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


def test_pipeline_returns_ranked_evidence() -> None:
    pipeline = EvidenceRetrievalPipeline(
        search_client=DummySearchClient(),
        document_fetcher=DummyFetcher(),
        retrieval_config_path=Path("configs/retrieval.yaml"),
    )

    bundle = pipeline.retrieve("Scientists discover miracle cancer cure in early research.")

    assert bundle.claim
    assert bundle.queries
    assert bundle.evidence
    assert bundle.evidence[0].source == "Reuters"
