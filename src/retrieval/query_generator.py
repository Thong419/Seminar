"""Search query generation from extracted claims and keywords."""

from __future__ import annotations

from collections.abc import Iterable
import re


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "this", "that", "these", "those",
    "said", "say", "says", "according", "report", "reports", "article",
}

ENTITY_PATTERN = re.compile(r"\b(?:[A-Z]{2,}(?:[-'][A-Z0-9]+)*|[A-Z][a-z]+(?:[-'][A-Z0-9]+)*)\b")


_DOMAIN_SUFFIX_BY_CLAIM_TYPE: dict[str, list[str]] = {
    "medical_claim": [
        "site:nih.gov OR site:who.int OR site:cdc.gov",
        "site:pubmed.ncbi.nlm.nih.gov OR site:cancer.gov",
        "medical evidence",
        "clinical study",
        "health research",
    ],
    "science_claim": [
        "site:nasa.gov OR site:nature.com OR site:science.org",
        "peer reviewed",
        "scientific study",
        "research evidence",
    ],
    "political_claim": [
        "site:reuters.com OR site:apnews.com",
        "fact check",
        "verified",
        "official statement",
    ],
    "technology_claim": [
        "site:wired.com OR site:arstechnica.com OR site:ieee.org",
        "technical analysis",
        "verified source",
    ],
    "economic_claim": [
        "site:federalreserve.gov OR site:imf.org OR site:worldbank.org OR site:bls.gov",
        "economic data",
        "official statistics",
    ],
    "general_claim": [
        "fact check",
        "verified",
        "evidence",
    ],
}


def generate_search_queries(
    article_text: str,
    claim: str | None = None,
    keywords: list[str] | None = None,
    max_queries: int = 5,
    claim_type: str = "general_claim",
    subject: str = "",
    predicate: str = "",
    object_: str = "",
) -> list[str]:
    """Generate domain-aware search queries.

    Args:
        article_text: Full article text.
        claim: Extracted claim sentence.
        keywords: Key terms from claim extraction.
        max_queries: Maximum number of queries to return.
        claim_type: Claim type from classify_claim_type() for domain-biased queries.

    Returns:
        Deduplicated list of search query strings capped at max_queries.
    """
    claim = claim or article_text
    keywords = keywords or []

    claim_tokens = _normalize_tokens(claim)
    keyword_tokens = _normalize_tokens(" ".join(keywords))
    entity_tokens = _extract_entity_tokens(article_text) or _extract_entity_tokens(claim)
    tokens = _dedupe([token for token in claim_tokens if token not in STOPWORDS] + keyword_tokens)
    claim_focus_tokens = _claim_focus_tokens(claim, subject=subject, predicate=predicate, object_=object_)
    queries: list[str] = []

    # Domain-priority variables (used later for authoritative source queries)
    domain_suffixes = _DOMAIN_SUFFIX_BY_CLAIM_TYPE.get(claim_type, _DOMAIN_SUFFIX_BY_CLAIM_TYPE["general_claim"])
    core = " ".join(claim_focus_tokens[: min(len(claim_focus_tokens), 6)] or tokens[: min(len(tokens), 5)])
    entity_core = " ".join(entity_tokens[:3]) if entity_tokens else ""

    # Domain-priority queries must appear first to survive max_queries truncation.
    if core and domain_suffixes:
        queries.append(f"{core} {domain_suffixes[0]}")

    # Keep one broad query as secondary fallback.
    if claim_focus_tokens:
        queries.append(" ".join(claim_focus_tokens[: min(len(claim_focus_tokens), 7)]))
    elif tokens:
        queries.append(" ".join(tokens[: min(len(tokens), 6)]))

    if len(claim_focus_tokens) >= 3:
        queries.append(" ".join(claim_focus_tokens[:4]) + " fact check")
        queries.append(" ".join(claim_focus_tokens[:4]) + " research study")
        queries.append(" ".join(claim_focus_tokens[:4]) + " evidence")
    elif len(tokens) >= 3:
        queries.append(" ".join(tokens[:3]) + " fact check")
        queries.append(" ".join(tokens[:3]) + " research study")
        queries.append(" ".join(tokens[:3]) + " evidence")

    if core and len(domain_suffixes) > 1:
        queries.append(f"{core} {domain_suffixes[1]}")

    # Entity-only queries are de-prioritized to avoid generic pages.
    if entity_tokens and claim_focus_tokens:
        queries.append(" ".join(claim_focus_tokens[:3]) + " " + " ".join(entity_tokens[:2]))

    if keyword_tokens:
        queries.append(" ".join(keyword_tokens[: min(len(keyword_tokens), 5)]))

    fallback = _compact_whitespace(article_text[:160])
    if fallback:
        queries.append(fallback)

    return _dedupe(queries)[:max_queries]


def _normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[\u2018\u2019]", "'", text)
    cleaned = re.sub(r"[^A-Za-z0-9\-']+", " ", cleaned)
    normalized: list[str] = []
    for raw_token in cleaned.lower().replace("-", " ").split():
        token = raw_token.strip("'\".,:;!?()[]{}")
        if token.endswith("'s"):
            token = token[:-2]
        if token.endswith("s'"):
            token = token[:-1]
        token = token.strip("'\".,:;!?()[]{}")
        if token:
            normalized.append(token)
    return normalized


def _extract_entity_tokens(text: str) -> list[str]:
    tokens = []
    for match in ENTITY_PATTERN.findall(text or ""):
        token = match.strip("'\".,:;!?()[]{}")
        token = token.replace("'s", "")
        token = token.replace("’s", "")
        if token and token.lower() not in STOPWORDS:
            tokens.append(token)
    return _dedupe(tokens)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def _compact_whitespace(text: str) -> str:
    return " ".join(text.split())


_LEADING_NOISE_TOKENS = {
    "washington",
    "reuters",
    "associated",
    "press",
    "breaking",
    "exclusive",
}

_GENERIC_FOCUS_TOKENS = {
    "president",
    "government",
    "scientists",
    "scientist",
    "officials",
    "chief",
    "strategist",
    "announced",
    "starting",
    "worldwide",
    "routine",
    "change",
    "part",
    "wednesday",
}


def _claim_focus_tokens(claim: str, subject: str = "", predicate: str = "", object_: str = "") -> list[str]:
    """Return tokens that better represent claim assertion than entities alone."""
    subject_tokens = [t for t in _normalize_tokens(subject) if t not in STOPWORDS and t not in _LEADING_NOISE_TOKENS]
    predicate_tokens = [t for t in _normalize_tokens(predicate) if t not in STOPWORDS]
    object_tokens = [t for t in _normalize_tokens(object_) if t not in STOPWORDS]
    subject_specific = [t for t in subject_tokens if t not in _GENERIC_FOCUS_TOKENS]
    object_specific = [t for t in object_tokens if t not in _GENERIC_FOCUS_TOKENS]
    if subject_tokens or predicate_tokens or object_tokens:
        focused = object_specific[:5] + predicate_tokens[:2] + subject_specific[:3] + object_tokens[:2] + subject_tokens[:2]
        return _dedupe(focused)

    tokens = [t for t in _normalize_tokens(claim) if t not in STOPWORDS]
    while tokens and tokens[0] in _LEADING_NOISE_TOKENS:
        tokens.pop(0)
    return _dedupe(tokens)
