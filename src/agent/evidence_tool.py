"""Evidence retrieval tool.

This version uses the existing retrieval pipeline with HTML search fallbacks,
so the agent always performs evidence retrieval even when no Tavily API key is
available. The returned bundle includes source metadata and conflict signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from src.agents.retrieval.retrieval_agent import RetrievalAgent
from src.agents.state import EvidenceItem
from src.retrieval.document_fetcher import DocumentFetcher
from src.retrieval.search_client import BingHtmlSearchClient, HybridSearchClient, SearchClient, TavilySearchClient, WikipediaHtmlSearchClient


SUPPORT_KEYWORDS = {
    "verified",
    "confirmed",
    "supports",
    "supported",
    "reports",
    "report",
    "exploring",
    "mission",
    "missioned",
    "evidence",
}

CONTRADICTION_KEYWORDS = {
    "false",
    "fake",
    "hoax",
    "misinformation",
    "debunk",
    "refute",
    "refuted",
    "no evidence",
    "not true",
    "unsupported",
}


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    evidence_found: bool
    claim: str
    queries: list[str]
    sources: list[dict[str, Any]]
    summary: str
    support_score: float
    contradiction_score: float
    source_credibility_score: float
    evidence_quality_score: float
    conflict_flag: bool


class EvidenceTool:
    def __init__(
        self,
        max_results: int = 5,
        timeout: float = 8.0,
        retrieval_config_path: Path | str = Path("configs/retrieval.yaml"),
        search_client: SearchClient | None = None,
        document_fetcher: DocumentFetcher | None = None,
        retrieval_agent: RetrievalAgent | None = None,
    ) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self.retrieval_config_path = Path(retrieval_config_path)
        self.search_client = search_client or self._default_search_client()
        self.document_fetcher = document_fetcher or DocumentFetcher(timeout_seconds=int(timeout))
        self.retrieval_agent = retrieval_agent or RetrievalAgent(
            retrieval_config_path=self.retrieval_config_path,
            search_client=self.search_client,
            document_fetcher=self.document_fetcher,
        )

    def _default_search_client(self) -> SearchClient:
        return HybridSearchClient(
            [
                WikipediaHtmlSearchClient(timeout_seconds=int(self.timeout)),
                BingHtmlSearchClient(timeout_seconds=int(self.timeout)),
            ]
        )

    def run(self, article_text: str) -> dict[str, Any]:
        bundle = self.retrieval_agent.retrieve(article_text)
        sources = [self._item_to_dict(item, article_text) for item in bundle.evidence]

        evidence_found = bool(sources)
        support_score, contradiction_score = self._compute_alignment_scores(sources)
        source_credibility_score = self._source_credibility_score(sources)
        evidence_quality_score = self._evidence_quality_score(sources, source_credibility_score)
        conflict_flag = self._detect_conflict(article_text, sources, evidence_quality_score)
        summary = self._build_summary(bundle.claim, bundle.queries, sources, support_score, contradiction_score)

        return {
            "evidence_found": evidence_found,
            "claim": bundle.claim,
            "queries": bundle.queries,
            "sources": sources,
            "summary": summary,
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "source_credibility_score": source_credibility_score,
            "evidence_quality_score": evidence_quality_score,
            "conflict_flag": conflict_flag,
        }

    def _item_to_dict(self, item: EvidenceItem, article_text: str) -> dict[str, Any]:
        stance = self._infer_stance(article_text, item.content)
        matched_terms = self._matched_terms(article_text, item.content)
        relevance_score = float(getattr(item, "relevance_score", 0.0))
        source_credibility = float(getattr(item, "source_credibility", 0.5))
        return {
            "title": item.title,
            "source": item.source,
            "content": item.content,
            "relevance_score": relevance_score,
            "url": item.url,
            "query": getattr(item, "query", None),
            "provider": getattr(item, "provider", None),
            "source_credibility": source_credibility,
            "stance": stance,
            "matched_terms": matched_terms,
        }

    def _infer_stance(self, article_text: str, content: str) -> str:
        lowered = f"{article_text} {content}".lower()
        overlap = len(self._matched_terms(article_text, content))
        if overlap >= 3 or any(keyword in lowered for keyword in SUPPORT_KEYWORDS):
            return "support"
        if any(keyword in lowered for keyword in CONTRADICTION_KEYWORDS) and overlap <= 2:
            return "contradict"
        return "neutral"

    def _matched_terms(self, article_text: str, content: str) -> list[str]:
        strip_chars = " .,:;!?()[]{}\"'`"
        article_tokens = {token.strip(strip_chars).lower() for token in article_text.split() if len(token.strip(strip_chars)) > 2}
        content_tokens = {token.strip(strip_chars).lower() for token in content.split() if len(token.strip(strip_chars)) > 2}
        return sorted(article_tokens & content_tokens)

    def _compute_alignment_scores(self, sources: list[dict[str, Any]]) -> tuple[float, float]:
        if not sources:
            return 0.0, 0.0

        support_scores: list[float] = []
        contradiction_scores: list[float] = []
        for source in sources:
            relevance = float(source.get("relevance_score", 0.0))
            credibility = float(source.get("source_credibility", 0.5))
            stance = str(source.get("stance", "neutral"))
            if stance == "support":
                support_scores.append((0.7 * relevance) + (0.3 * credibility))
                contradiction_scores.append(0.0)
            elif stance == "contradict":
                contradiction_scores.append((0.7 * relevance) + (0.3 * credibility))
                support_scores.append(0.0)
            else:
                support_scores.append(0.35 * relevance)
                contradiction_scores.append(0.35 * relevance)

        support_score = min(1.0, fmean(support_scores) if support_scores else 0.0)
        contradiction_score = min(1.0, fmean(contradiction_scores) if contradiction_scores else 0.0)
        return support_score, contradiction_score

    def _source_credibility_score(self, sources: list[dict[str, Any]]) -> float:
        if not sources:
            return 0.0
        scores = [float(source.get("source_credibility", 0.5)) for source in sources]
        return max(0.0, min(1.0, fmean(scores)))

    def _evidence_quality_score(self, sources: list[dict[str, Any]], source_credibility_score: float) -> float:
        if not sources:
            return 0.0
        average_relevance = fmean(float(source.get("relevance_score", 0.0)) for source in sources)
        coverage = min(1.0, len(sources) / max(1, self.max_results))
        return max(0.0, min(1.0, (0.4 * coverage) + (0.3 * average_relevance) + (0.3 * source_credibility_score)))

    def _detect_conflict(self, article_text: str, sources: list[dict[str, Any]], evidence_quality_score: float) -> bool:
        if evidence_quality_score < 0.35 or not sources:
            return False

        article_lower = article_text.lower()
        contradiction_hit = any(source.get("stance") == "contradict" for source in sources)
        support_hit = any(source.get("stance") == "support" for source in sources)

        if contradiction_hit and not support_hit:
            return True
        if support_hit and any(keyword in article_lower for keyword in ("no evidence", "false", "fake", "hoax", "misinformation")):
            return True
        return False

    def _build_summary(
        self,
        claim: str,
        queries: list[str],
        sources: list[dict[str, Any]],
        support_score: float,
        contradiction_score: float,
    ) -> str:
        if not sources:
            return f"No supporting evidence was retrieved for claim: {claim}. Queries tried: {', '.join(queries[:3])}."

        top_titles = ", ".join(source["title"] for source in sources[:3])
        support_text = "supports" if support_score >= contradiction_score else "contradicts"
        return (
            f"Retrieved {len(sources)} evidence items. Top sources: {top_titles}. "
            f"Evidence {support_text} the claim with support_score={support_score:.2f} and contradiction_score={contradiction_score:.2f}."
        )
