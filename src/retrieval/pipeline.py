"""End-to-end real evidence retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.retrieval.claim_extractor import extract_claim
from src.retrieval.document_fetcher import DocumentFetcher, EvidenceDocument
from src.retrieval.evidence_ranker import load_rank_config, load_trust_scores, rank_evidence
from src.retrieval.query_generator import generate_search_queries
from src.retrieval.search_client import SearchClient, SearchResult


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    claim: str
    keywords: list[str]
    queries: list[str]
    evidence: list[EvidenceDocument]


class EvidenceRetrievalPipeline:
    def __init__(
        self,
        search_client: SearchClient,
        document_fetcher: DocumentFetcher,
        retrieval_config_path: Path,
    ) -> None:
        self.search_client = search_client
        self.document_fetcher = document_fetcher
        self.retrieval_config_path = retrieval_config_path
        self.rank_config = load_rank_config(retrieval_config_path)
        self.trust_scores = load_trust_scores(retrieval_config_path)

    def retrieve(self, article_text: str) -> RetrievalBundle:
        claim_result = extract_claim(article_text)
        claim = str(claim_result["claim"])
        keywords = list(claim_result["keywords"])
        queries = generate_search_queries(article_text=article_text, claim=claim, keywords=keywords, max_queries=self.rank_config.top_k)

        search_results: list[SearchResult] = []
        for query in queries:
            search_results.extend(self.search_client.search(query, limit=self.rank_config.top_k))

        evidence_documents = [self.document_fetcher.fetch(result) for result in search_results]
        evidence_documents = [
            EvidenceDocument(
                title=item.title,
                url=item.url,
                source=item.source,
                content=item.content,
                trust_score=self.trust_scores.get(item.source, 0.5),
                relevance_score=item.relevance_score,
                query=item.query,
                provider=item.provider,
                source_credibility=getattr(item, "source_credibility", self.trust_scores.get(item.source, 0.5)),
                stance=item.stance,
                matched_terms=item.matched_terms,
            )
            for item in evidence_documents
        ]

        ranked = rank_evidence(claim, evidence_documents, self.trust_scores, self.rank_config)
        return RetrievalBundle(claim=claim, keywords=keywords, queries=queries, evidence=ranked)
