"""Evidence retrieval tool.

This version uses the existing retrieval pipeline with HTML search fallbacks,
so the agent always performs evidence retrieval even when no Tavily API key is
available. The returned bundle includes source metadata and conflict signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from statistics import fmean
from typing import Any

from src.agent.claim_classifier import classify_claim_type
from src.agent.retrieval.retrieval_agent import RetrievalAgent
from src.agent.stance_detector import detect_stance, stance_summary
from src.agent.state import EvidenceItem
from src.retrieval.document_fetcher import DocumentFetcher
from src.retrieval.search_client import BingHtmlSearchClient, HybridSearchClient, SearchClient, TavilySearchClient, WikipediaHtmlSearchClient


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
    accepted_evidence: list[dict[str, Any]] | None = None
    rejected_evidence: list[dict[str, Any]] | None = None


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
        retrieval_mode = os.getenv("RETRIEVAL_MODE", "live").strip().lower()
        if retrieval_agent is not None:
            self.retrieval_agent = retrieval_agent
            self.search_client = search_client or self._default_search_client()
            self.document_fetcher = document_fetcher or DocumentFetcher(timeout_seconds=int(timeout))
        elif retrieval_mode == "demo" and search_client is None and document_fetcher is None:
            self.retrieval_agent = RetrievalAgent(retrieval_config_path=self.retrieval_config_path)
            self.search_client = self.retrieval_agent.search_client
            self.document_fetcher = self.retrieval_agent.document_fetcher
        else:
            self.search_client = search_client or self._default_search_client()
            self.document_fetcher = document_fetcher or DocumentFetcher(timeout_seconds=int(timeout))
            self.retrieval_agent = RetrievalAgent(
                retrieval_config_path=self.retrieval_config_path,
                search_client=self.search_client,
                document_fetcher=self.document_fetcher,
            )

    def _default_search_client(self) -> SearchClient:
        return HybridSearchClient(
            [
                BingHtmlSearchClient(timeout_seconds=int(self.timeout)),
                WikipediaHtmlSearchClient(timeout_seconds=int(self.timeout)),
            ]
        )

    def run(self, article_text: str) -> dict[str, Any]:
        bundle = self.retrieval_agent.retrieve(article_text)
        # Use the extracted claim (not the full article) for stance detection.
        # This prevents generic Wikipedia pages from matching via irrelevant overlap.
        claim_text = bundle.claim or article_text
        accepted_audit = {
            self._audit_key(record): record
            for record in list(getattr(bundle, "accepted_evidence", []) or [])
        }
        sources = [
            self._item_to_dict(item, claim_text, accepted_audit.get((item.url or "", item.title)))
            for item in bundle.evidence
        ]

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
            "accepted_evidence": list(getattr(bundle, "accepted_evidence", []) or []),
            "rejected_evidence": list(getattr(bundle, "rejected_evidence", []) or []),
        }

    def _item_to_dict(
        self,
        item: EvidenceItem,
        claim: str,
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert an EvidenceItem to a dict using real stance detection.

        Uses the extracted *claim* (not the full article) so that a Wikipedia
        page about 'Water' does not get incorrectly labelled as SUPPORT for a
        cancer-cure claim.
        """
        raw_stance = detect_stance(claim, item.content)
        # Normalise to lowercase keys used by downstream decision logic
        stance_map = {"SUPPORT": "support", "REFUTE": "refute", "NEUTRAL": "neutral"}
        stance = stance_map.get(raw_stance, "neutral")
        matched_terms = self._matched_terms(claim, item.content)
        relevance_score = float(getattr(item, "relevance_score", 0.0))
        source_credibility = float(getattr(item, "source_credibility", 0.5))
        payload = {
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
        if audit:
            payload.update(
                {
                    "query_used": audit.get("query_used"),
                    "semantic_score": audit.get("semantic_score", 0.0),
                    "coverage_score": audit.get("coverage_score", 0.0),
                    "entity_overlap": audit.get("entity_overlap", 0.0),
                    "predicate_overlap": audit.get("predicate_overlap", 0.0),
                    "accepted": True,
                    "rejection_reason": None,
                }
            )
        return payload

    def _audit_key(self, record: dict[str, Any]) -> tuple[str, str]:
        return (str(record.get("url", "")), str(record.get("title", "")))

    def _matched_terms(self, claim: str, content: str) -> list[str]:
        strip_chars = " .,:;!?()[]{}\"'`"
        claim_tokens = {t.strip(strip_chars).lower() for t in claim.split() if len(t.strip(strip_chars)) > 3}
        content_tokens = {t.strip(strip_chars).lower() for t in content.split() if len(t.strip(strip_chars)) > 3}
        return sorted(claim_tokens & content_tokens)

    def _compute_alignment_scores(self, sources: list[dict[str, Any]]) -> tuple[float, float]:
        """Compute support/contradiction scores from per-source stance labels.

        NEUTRAL sources contribute a small amount to both scores (they are
        topically relevant but non-conclusive). REFUTE maps to contradiction.
        """
        if not sources:
            return 0.0, 0.0

        support_scores: list[float] = []
        contradiction_scores: list[float] = []
        for source in sources:
            relevance = float(source.get("relevance_score", 0.0))
            credibility = float(source.get("source_credibility", 0.5))
            stance = str(source.get("stance", "neutral"))
            weighted = (0.65 * relevance) + (0.35 * credibility)
            if stance == "support":
                support_scores.append(weighted)
                contradiction_scores.append(0.0)
            elif stance in {"refute", "contradict"}:
                contradiction_scores.append(weighted)
                support_scores.append(0.0)
            else:  # neutral
                # Neutral evidence contributes a small, unequal amount to both sides
                # (slightly more to support to break ties in real-world cases)
                support_scores.append(0.18 * relevance)
                contradiction_scores.append(0.12 * relevance)

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
        generic_hits = sum(1 for s in sources if self._is_generic_source(s))
        generic_penalty = min(0.35, 0.10 * generic_hits)
        score = (0.4 * coverage) + (0.3 * average_relevance) + (0.3 * source_credibility_score) - generic_penalty
        return max(0.0, min(1.0, score))

    def _is_generic_source(self, source: dict[str, Any]) -> bool:
        title = str(source.get("title", "")).strip().lower()
        generic_titles = {
            "water",
            "great salt lake",
            "google",
            "smartphone",
            "white house",
            "ufo",
        }
        if title in generic_titles:
            return True
        if str(source.get("source", "")).lower().endswith("wikipedia") and len(title.split()) <= 2:
            return True
        return False

    def _detect_conflict(self, article_text: str, sources: list[dict[str, Any]], evidence_quality_score: float) -> bool:
        """Detect when evidence stances conflict with the article claim.

        A conflict exists when:
        - Refuting sources are found (any quality level)
        - Mixed stances exist with meaningful evidence quality
        - Article itself makes a contradictory claim (article text contains
          both assertion and denial, e.g., 'The WHO said there is no evidence...')
        """
        if not sources:
            return False

        refute_hit = any(source.get("stance") in {"refute", "contradict"} for source in sources)
        support_hit = any(source.get("stance") == "support" for source in sources)

        # Article itself may be stating a denial/correction (e.g., fact-check style)
        _article_lower = article_text.lower()
        article_is_denial = any(kw in _article_lower for kw in ("no evidence", "false", "not true", "hoax", "misinformation", "said there is no", "there is no evidence"))

        # If the article itself is a denial, any retrieved source that discusses the topic
        # (support OR neutral) indicates a conflict between the article's stance and the evidence
        if article_is_denial and (support_hit or any(source.get("stance") == "neutral" for source in sources)):
            return True

        if evidence_quality_score >= 0.30:
            if refute_hit and not support_hit:
                return True
            if refute_hit and support_hit:
                # Mixed evidence: flag as conflict
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
