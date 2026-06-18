"""End-to-end real evidence retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from src.retrieval.claim_extractor import extract_claim
from src.retrieval.document_fetcher import DocumentFetcher, EvidenceDocument
from src.retrieval.evidence_ranker import load_rank_config, load_retrieval_config, load_trust_scores, rank_evidence
from src.retrieval.query_generator import generate_search_queries
from src.retrieval.relevance_filter import dedupe_documents, score_claim_relevance, semantic_similarity_score
from src.retrieval.search_client import BingHtmlSearchClient, NasaWpSearchClient, SearchClient, SearchResult
from src.retrieval.source_credibility import DOMAIN_PRIORITY_THRESHOLD, score_source_credibility


_AUTHORITATIVE_DOMAINS_BY_CLAIM_TYPE: dict[str, list[str]] = {
    "science_claim": ["nasa.gov", "jpl.nasa.gov", "nature.com", "science.org"],
    "medical_claim": ["nih.gov", "cdc.gov", "who.int", "pubmed.ncbi.nlm.nih.gov"],
    "political_claim": ["reuters.com", "apnews.com"],
    "technology_claim": ["ieee.org", "arstechnica.com", "wired.com"],
    "economic_claim": ["imf.org", "worldbank.org", "bls.gov"],
}

_NOISY_QUERY_TOKENS = {
    "music",
    "song",
    "songs",
    "artist",
    "artists",
    "game",
    "games",
    "rockstar",
    "wikipedia",
}


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    claim: str
    keywords: list[str]
    queries: list[str]
    evidence: list[EvidenceDocument]
    accepted_evidence: list[dict[str, object]] | None = None
    rejected_evidence: list[dict[str, object]] | None = None


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
        retrieval_config = load_retrieval_config(retrieval_config_path)
        search_cfg = retrieval_config.get("search", {}) if isinstance(retrieval_config, dict) else {}
        if isinstance(search_cfg, dict):
            self.max_queries = int(search_cfg.get("max_queries", 4))
        else:
            self.max_queries = 4

    def retrieve(self, article_text: str) -> RetrievalBundle:
        claim_result = extract_claim(article_text)
        claim = str(claim_result["claim"])
        keywords = list(claim_result["keywords"])
        claim_type = str(claim_result.get("claim_type", "general_claim"))
        queries = generate_search_queries(
            article_text=article_text,
            claim=claim,
            keywords=keywords,
            max_queries=min(self.rank_config.top_k, self.max_queries),
            claim_type=claim_type,
            subject=str(claim_result.get("subject", "")),
            predicate=str(claim_result.get("predicate", "")),
            object_=str(claim_result.get("object", "")),
        )

        search_results: list[SearchResult] = []
        for query in queries:
            search_results.extend(self.search_client.search(query, limit=self.rank_config.top_k))

        fallback_queries: list[str] = []
        domains = _AUTHORITATIVE_DOMAINS_BY_CLAIM_TYPE.get(claim_type, [])
        if domains and not _has_domain_match(search_results, domains):
            fallback_queries = _build_authoritative_fallback_queries(claim_result, claim, domains)
            if fallback_queries:
                authoritative_client = BingHtmlSearchClient()
                nasa_client = NasaWpSearchClient() if "nasa.gov" in domains else None
                for query in fallback_queries:
                    if nasa_client is not None:
                        try:
                            search_results.extend(nasa_client.search(query, limit=self.rank_config.top_k))
                        except Exception:
                            pass
                    search_results.extend(authoritative_client.search(query, limit=self.rank_config.top_k))
                queries = _merge_queries(queries, fallback_queries, self.max_queries + len(fallback_queries))

        evidence_documents = [self.document_fetcher.fetch(result) for result in search_results]
        normalized_documents: list[EvidenceDocument] = []
        for item in evidence_documents:
            fetched_credibility = float(getattr(item, "source_credibility", 0.0) or 0.0)
            resolved_credibility = max(fetched_credibility, score_source_credibility(item.source, item.url))
            normalized_documents.append(
                EvidenceDocument(
                    title=item.title,
                    url=item.url,
                    source=item.source,
                    content=item.content,
                    trust_score=self.trust_scores.get(item.source, 0.5),
                    relevance_score=item.relevance_score,
                    query=item.query,
                    provider=item.provider,
                    source_credibility=resolved_credibility,
                    stance=item.stance,
                    matched_terms=item.matched_terms,
                )
            )
        evidence_documents = normalized_documents

        evidence_documents = dedupe_documents(evidence_documents)

        accepted: list[EvidenceDocument] = []
        accepted_audit: list[dict[str, object]] = []
        rejected_audit: list[dict[str, object]] = []
        for item in evidence_documents:
            semantic_score = semantic_similarity_score(claim, item)
            source_credibility = float(getattr(item, "source_credibility", 0.5))
            claim_type_threshold = float(DOMAIN_PRIORITY_THRESHOLD.get(claim_type, 0.60))
            relevance = score_claim_relevance(
                claim_result,
                item,
                semantic_similarity=semantic_score,
                min_claim_coverage=self.rank_config.min_claim_coverage,
                min_entity_overlap=self.rank_config.min_entity_overlap,
                min_semantic_similarity=self.rank_config.min_semantic_similarity,
            )
            is_accepted = relevance.accepted and source_credibility >= claim_type_threshold
            rejection_reason = relevance.rejection_reason
            if relevance.accepted and not is_accepted:
                rejection_reason = "low_source_credibility"
            audit_record = {
                "title": item.title,
                "url": item.url,
                "source": item.source,
                "query_used": item.query,
                "semantic_score": relevance.semantic_similarity_score,
                "coverage_score": relevance.claim_coverage_score,
                "entity_overlap": relevance.entity_overlap_score,
                "predicate_overlap": relevance.action_overlap_score,
                "entity_overlap_score": relevance.entity_overlap_score,
                "action_overlap_score": relevance.action_overlap_score,
                "claim_coverage_score": relevance.claim_coverage_score,
                "generic_penalty": relevance.generic_penalty,
                "adjusted_score": relevance.adjusted_score,
                "source_credibility": source_credibility,
                "claim_type": claim_type,
                "claim_type_threshold": claim_type_threshold,
                "accepted": is_accepted,
                "rejection_reason": rejection_reason,
            }
            if is_accepted:
                accepted.append(item)
                accepted_audit.append(audit_record)
            else:
                rejected_audit.append(audit_record)

        ranked = rank_evidence(claim, accepted, self.trust_scores, self.rank_config)
        return RetrievalBundle(
            claim=claim,
            keywords=keywords,
            queries=queries,
            evidence=ranked,
            accepted_evidence=accepted_audit,
            rejected_evidence=rejected_audit,
        )


def _merge_queries(base: list[str], extra: list[str], max_queries: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for query in [*base, *extra]:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
        if len(merged) >= max_queries:
            break
    return merged


def _host_from_result(result: SearchResult) -> str:
    if result.url:
        parsed = urlparse(result.url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            return host
    source = (result.source or "").strip().lower()
    return source.removeprefix("www.")


def _has_domain_match(results: list[SearchResult], domains: list[str]) -> bool:
    normalized_domains = [domain.lower().removeprefix("www.") for domain in domains]
    for result in results:
        host = _host_from_result(result)
        if any(host == domain or host.endswith(f".{domain}") for domain in normalized_domains):
            return True
    return False


def _tokenize_query_text(text: str) -> list[str]:
    import re

    cleaned = re.sub(r"[\u2018\u2019]", "'", text or "")
    cleaned = re.sub(r"[^A-Za-z0-9\-']+", " ", cleaned)
    tokens: list[str] = []
    for raw in cleaned.lower().replace("-", " ").split():
        token = raw.strip("'\".,:;!?()[]{}")
        if token.endswith("'s"):
            token = token[:-2]
        if token and token not in _NOISY_QUERY_TOKENS:
            tokens.append(token)
    return tokens


def _build_authoritative_fallback_queries(
    claim_result: dict[str, object],
    claim: str,
    domains: list[str],
) -> list[str]:
    subject = str(claim_result.get("subject", ""))
    predicate = str(claim_result.get("predicate", ""))
    object_ = str(claim_result.get("object", ""))
    keywords = [str(item) for item in claim_result.get("keywords", []) if str(item).strip()]

    base_tokens = _tokenize_query_text(" ".join([subject, predicate, object_, " ".join(keywords), claim]))
    compact_tokens: list[str] = []
    seen: set[str] = set()
    for token in base_tokens:
        if token in seen or len(token) < 3:
            continue
        seen.add(token)
        compact_tokens.append(token)

    focus = " ".join(compact_tokens[:7])
    if not focus:
        focus = " ".join(_tokenize_query_text(claim)[:6])

    queries: list[str] = []
    if focus:
        for domain in domains[:3]:
            queries.append(f"{focus} site:{domain}")

    if subject and object_:
        subject_focus = " ".join(_tokenize_query_text(subject)[:3])
        object_focus = " ".join(_tokenize_query_text(object_)[:4])
        if subject_focus and object_focus:
            queries.insert(0, f"\"{subject_focus}\" \"{object_focus}\" site:{domains[0]}")

    return _merge_queries([], queries, 6)
