"""Evidence ranking utilities with semantic similarity and trust scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.document_fetcher import EvidenceDocument


@dataclass(frozen=True, slots=True)
class RankConfig:
    semantic_similarity_weight: float = 0.5
    source_trust_weight: float = 0.3
    provider_relevance_weight: float = 0.2
    min_trust_score: float = 0.5
    min_relevance_score: float = 0.35
    top_k: int = 5


class EvidenceRanker:
    def __init__(self, trust_scores: dict[str, float], config: RankConfig) -> None:
        self.trust_scores = trust_scores
        self.config = config

    def rank(self, claim: str, evidence: list[EvidenceDocument]) -> list[EvidenceDocument]:
        return rank_evidence(claim, evidence, self.trust_scores, self.config)


def load_retrieval_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Retrieval configuration must be a mapping.")
    return data


def load_trust_scores(path: Path) -> dict[str, float]:
    data = load_retrieval_config(path)
    raw_scores = data.get("trust_scores", {})
    if not isinstance(raw_scores, dict):
        raise ValueError("trust_scores must be a mapping.")
    return {str(source): float(score) for source, score in raw_scores.items()}


def load_rank_config(path: Path) -> RankConfig:
    data = load_retrieval_config(path)
    thresholds = data.get("thresholds", {})
    weights = data.get("ranking_weights", {})
    if not isinstance(thresholds, dict) or not isinstance(weights, dict):
        raise ValueError("Retrieval thresholds and weights must be mappings.")
    return RankConfig(
        semantic_similarity_weight=float(weights.get("semantic_similarity", RankConfig.semantic_similarity_weight)),
        source_trust_weight=float(weights.get("source_trust", RankConfig.source_trust_weight)),
        provider_relevance_weight=float(weights.get("provider_relevance", RankConfig.provider_relevance_weight)),
        min_trust_score=float(thresholds.get("min_trust_score", RankConfig.min_trust_score)),
        min_relevance_score=float(thresholds.get("min_relevance_score", RankConfig.min_relevance_score)),
        top_k=int(thresholds.get("top_k", RankConfig.top_k)),
    )


def rank_evidence(
    claim: str,
    evidence: list[EvidenceDocument],
    trust_scores: dict[str, float],
    config: RankConfig,
) -> list[EvidenceDocument]:
    if not evidence:
        return []

    texts = [claim, *[item.content for item in evidence]]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    claim_vector = matrix[0]

    scored_pairs: list[tuple[EvidenceDocument, float]] = []
    for index, item in enumerate(evidence, start=1):
        semantic_similarity = float(cosine_similarity(claim_vector, matrix[index])[0][0])
        trust_score = trust_scores.get(item.source, 0.5)
        combined = _combine_scores(semantic_similarity, trust_score, item.relevance_score, config)
        if trust_score >= config.min_trust_score and item.relevance_score >= config.min_relevance_score:
            scored_pairs.append((item, combined))

    scored_pairs.sort(key=lambda pair: pair[1], reverse=True)
    ranked: list[EvidenceDocument] = []
    for item, score in scored_pairs[: config.top_k]:
        ranked.append(
            EvidenceDocument(
                title=item.title,
                url=item.url,
                source=item.source,
                content=item.content,
                trust_score=trust_scores.get(item.source, 0.5),
                relevance_score=max(0.0, min(1.0, score)),
                query=item.query,
                provider=item.provider,
                source_credibility=trust_scores.get(item.source, item.source_credibility),
                stance=item.stance,
                matched_terms=item.matched_terms,
            )
        )
    return ranked


def _combine_scores(semantic_similarity: float, trust_score: float, relevance_score: float, config: RankConfig) -> float:
    raw_score = (
        semantic_similarity * config.semantic_similarity_weight
        + trust_score * config.source_trust_weight
        + relevance_score * config.provider_relevance_weight
    )
    return max(0.0, min(1.0, raw_score))
