"""Claim-centric relevance filtering and deduplication utilities.

This module hardens retrieval quality by:
1. Deduplicating evidence by URL and title similarity.
2. Scoring evidence against the extracted claim assertion (subject/action/object).
3. Rejecting low-coverage and generic concept pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any
from urllib.parse import urlparse, unquote

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.document_fetcher import EvidenceDocument


_GENERIC_TITLES = {
    "water",
    "great salt lake",
    "google",
    "smartphone",
    "white house",
    "ufo",
    "alcoholic beverage",
    "covid-19 misinformation",
}

_GENERIC_PAGE_HINTS = (
    "wikipedia",
    "encyclopedia",
    "category",
    "disambiguation",
)

_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-']+")


@dataclass(frozen=True, slots=True)
class RelevanceScore:
    entity_overlap_score: float
    action_overlap_score: float
    claim_coverage_score: float
    semantic_similarity_score: float
    generic_penalty: float
    adjusted_score: float
    accepted: bool
    rejection_reason: str | None = None
    matched_subject_terms: list[str] | None = None
    matched_action_terms: list[str] | None = None
    matched_object_terms: list[str] | None = None


def dedupe_documents(documents: list[EvidenceDocument], title_similarity_threshold: float = 0.80) -> list[EvidenceDocument]:
    """Remove duplicate evidence documents by URL and title similarity."""
    deduped: list[EvidenceDocument] = []
    seen_urls: set[str] = set()

    for doc in documents:
        normalized_url = _normalize_url(doc.url)
        normalized_title = _normalize_title(doc.title)

        if normalized_url and normalized_url in seen_urls:
            continue

        is_title_dup = False
        for existing in deduped:
            similarity = SequenceMatcher(
                None,
                normalized_title,
                _normalize_title(existing.title),
            ).ratio()
            if similarity >= title_similarity_threshold:
                is_title_dup = True
                break

        if is_title_dup:
            continue

        if normalized_url:
            seen_urls.add(normalized_url)
        deduped.append(doc)

    return deduped


def score_claim_relevance(
    claim_data: dict[str, Any],
    evidence: EvidenceDocument,
    semantic_similarity: float | None = None,
    min_claim_coverage: float = 0.30,
    min_entity_overlap: float = 0.20,
    min_semantic_similarity: float = 0.08,
) -> RelevanceScore:
    """Score whether evidence addresses the claim assertion.

    Required focus:
    - subject overlap
    - action overlap
    - object overlap

    Reject when claim_coverage_score is below threshold.
    """
    content = _normalize_text(evidence.content)
    title = _normalize_text(evidence.title)
    claim_text = str(claim_data.get("claim") or " ".join(
        part for part in [claim_data.get("subject", ""), claim_data.get("predicate", ""), claim_data.get("object", "")] if part
    ))
    if semantic_similarity is None:
        semantic_similarity = semantic_similarity_score(claim_text, evidence)

    subject_terms = _extract_terms(str(claim_data.get("subject", "")))
    action_terms = _extract_action_terms(str(claim_data.get("predicate", "")))
    object_terms = _extract_terms(str(claim_data.get("object", "")))

    matched_subject = [t for t in subject_terms if t in content]
    matched_action = [t for t in action_terms if t in content]
    matched_object = [t for t in object_terms if t in content]

    subject_overlap = _ratio(len(matched_subject), len(subject_terms))
    action_overlap = _ratio(len(matched_action), len(action_terms))
    object_overlap = _ratio(len(matched_object), len(object_terms))

    # Entity overlap combines subject + object coverage.
    entity_overlap = (0.45 * subject_overlap) + (0.55 * object_overlap)

    # Coverage emphasizes action + object so topic-only pages are penalized.
    claim_coverage = (0.30 * entity_overlap) + (0.40 * action_overlap) + (0.30 * object_overlap)

    generic_penalty = _generic_page_penalty(title=title, url=evidence.url, content=content)

    adjusted = max(0.0, min(1.0, claim_coverage - generic_penalty))
    has_entity_coverage = validate_claim_entity_coverage(
        claim_data,
        evidence,
        min_entity_overlap=min_entity_overlap,
        min_claim_coverage=min_claim_coverage,
    )
    accepted = (
        has_entity_coverage
        and semantic_similarity >= min_semantic_similarity
        and claim_coverage >= min_claim_coverage
        and entity_overlap >= min_entity_overlap
        and generic_penalty < 0.20
    )

    rejection_reason: str | None = None
    if not accepted:
        if generic_penalty >= 0.20:
            rejection_reason = "generic_or_concept_page"
        elif semantic_similarity < min_semantic_similarity:
            rejection_reason = "low_semantic_similarity"
        elif entity_overlap < min_entity_overlap:
            rejection_reason = "low_entity_overlap"
        elif action_overlap < 0.15:
            rejection_reason = "missing_claim_action"
        elif object_overlap < 0.15:
            rejection_reason = "missing_claim_object"
        else:
            rejection_reason = "low_claim_coverage"

    return RelevanceScore(
        entity_overlap_score=round(entity_overlap, 4),
        action_overlap_score=round(action_overlap, 4),
        claim_coverage_score=round(claim_coverage, 4),
        semantic_similarity_score=round(semantic_similarity, 4),
        generic_penalty=round(generic_penalty, 4),
        adjusted_score=round(adjusted, 4),
        accepted=accepted,
        rejection_reason=rejection_reason,
        matched_subject_terms=matched_subject,
        matched_action_terms=matched_action,
        matched_object_terms=matched_object,
    )


def validate_claim_entity_coverage(
    claim_data: dict[str, Any],
    evidence: EvidenceDocument,
    min_entity_overlap: float = 0.20,
    min_claim_coverage: float = 0.30,
) -> bool:
    """Require evidence to cover both claim entities and the asserted action."""
    content = _normalize_text(f"{evidence.title} {evidence.content}")
    subject_terms = _extract_terms(str(claim_data.get("subject", "")))
    object_terms = _extract_terms(str(claim_data.get("object", "")))
    action_terms = _extract_action_terms(str(claim_data.get("predicate", "")))

    subject_overlap = _ratio(sum(1 for t in subject_terms if t in content), len(subject_terms))
    object_overlap = _ratio(sum(1 for t in object_terms if t in content), len(object_terms))
    action_overlap = _ratio(sum(1 for t in action_terms if t in content), len(action_terms))
    entity_overlap = (0.45 * subject_overlap) + (0.55 * object_overlap)
    claim_coverage = (0.30 * entity_overlap) + (0.40 * action_overlap) + (0.30 * object_overlap)

    return entity_overlap >= min_entity_overlap and claim_coverage >= min_claim_coverage


def semantic_similarity_score(claim: str, evidence: EvidenceDocument) -> float:
    """Compute lightweight semantic similarity used by the acceptance gate."""
    evidence_text = _compact_text(f"{evidence.title} {evidence.content}")
    claim_text = _compact_text(claim)
    if not claim_text or not evidence_text:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([claim_text, evidence_text])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def _generic_page_penalty(title: str, url: str, content: str) -> float:
    title_clean = title.strip().lower()

    penalty = 0.0
    if title_clean in _GENERIC_TITLES:
        penalty += 0.35

    if any(hint in title_clean for hint in _GENERIC_PAGE_HINTS):
        penalty += 0.15

    host = _extract_host(url)
    if host.endswith("wikipedia.org"):
        # Penalize concept pages on Wikipedia when title is very short and generic.
        title_tokens = title_clean.split()
        if len(title_tokens) <= 2:
            penalty += 0.20

    # Very short article content usually means weak context/snippet extraction.
    if len(content.split()) < 40:
        penalty += 0.08

    return min(0.55, penalty)


def _extract_terms(text: str) -> list[str]:
    return [t for t in _WORD_PATTERN.findall(_normalize_text(text)) if len(t) > 2]


def _extract_action_terms(predicate: str) -> list[str]:
    base = _extract_terms(predicate)
    expanded: set[str] = set(base)
    for term in base:
        if term.endswith("ed") and len(term) > 4:
            expanded.add(term[:-2])
        if term.endswith("ing") and len(term) > 5:
            expanded.add(term[:-3])
        if term.endswith("es") and len(term) > 4:
            expanded.add(term[:-2])
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
    return sorted(expanded)


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = unquote(parsed.path or "").rstrip("/")
    return f"{host}{path}"


def _extract_host(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _normalize_title(text: str) -> str:
    normalized = _normalize_text(text)
    normalized = normalized.replace(" - wikipedia", "")
    normalized = normalized.replace("(disambiguation)", "")
    return normalized.strip()


def _compact_text(text: str) -> str:
    return " ".join(str(text).split())
