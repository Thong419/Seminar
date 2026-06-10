"""Retrieval adapter used by the agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents.state import EvidenceItem
from src.retrieval.document_fetcher import DocumentFetcher
from src.retrieval.pipeline import EvidenceRetrievalPipeline
from src.retrieval.search_client import SearchClient, TavilySearchClient


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    evidence: list[EvidenceItem]
    claim: str
    keywords: list[str]
    queries: list[str]

    def combined(self) -> list[EvidenceItem]:
        return self.evidence


class RetrievalAgent:
    def __init__(
        self,
        retrieval_config_path: Path | str = Path("configs/retrieval.yaml"),
        search_client: SearchClient | None = None,
        document_fetcher: DocumentFetcher | None = None,
    ) -> None:
        self.retrieval_config_path = Path(retrieval_config_path)
        self.search_client = search_client or TavilySearchClient(api_url="https://api.tavily.com/search")
        self.document_fetcher = document_fetcher or DocumentFetcher()
        self.pipeline = EvidenceRetrievalPipeline(
            search_client=self.search_client,
            document_fetcher=self.document_fetcher,
            retrieval_config_path=self.retrieval_config_path,
        )

    def retrieve(self, text: str) -> RetrievalBundle:
        bundle = self.pipeline.retrieve(text)
        evidence = [
            EvidenceItem(
                title=item.title,
                source=item.source,
                content=item.content,
                relevance_score=item.relevance_score,
                url=item.url,
            )
            for item in bundle.evidence
        ]
        return RetrievalBundle(
            evidence=evidence,
            claim=bundle.claim,
            keywords=bundle.keywords,
            queries=bundle.queries,
        )
