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


def generate_search_queries(
    article_text: str,
    claim: str | None = None,
    keywords: list[str] | None = None,
    max_queries: int = 4,
) -> list[str]:
    claim = claim or article_text
    keywords = keywords or []

    claim_tokens = _normalize_tokens(claim)
    keyword_tokens = _normalize_tokens(" ".join(keywords))
    entity_tokens = _extract_entity_tokens(article_text) or _extract_entity_tokens(claim)
    tokens = _dedupe([token for token in claim_tokens if token not in STOPWORDS] + keyword_tokens)
    queries: list[str] = []

    if entity_tokens:
        queries.append(" ".join(entity_tokens[: min(len(entity_tokens), 5)]))
    if entity_tokens and len(entity_tokens) >= 2:
        queries.append(" ".join(entity_tokens[:3]) + " fact check")
    if entity_tokens and len(entity_tokens) >= 2:
        queries.append(" ".join(entity_tokens[:3]) + " verified")
    if entity_tokens and len(entity_tokens) >= 2:
        queries.append(" ".join(entity_tokens[:3]) + " research study")

    if tokens:
        queries.append(" ".join(tokens[: min(len(tokens), 6)]))
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[:3]) + " fact check")
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[:3]) + " research study")
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[:3]) + " evidence")
    if len(tokens) >= 4:
        queries.append(" ".join(tokens[:4]) + " source")
    if keyword_tokens:
        queries.append(" ".join(keyword_tokens[: min(len(keyword_tokens), 5)]))

    fallback = _compact_whitespace(article_text[:160])
    if fallback:
        queries.append(fallback)

    return _dedupe(queries)[:max_queries]


def _normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[\u2018\u2019]", "'", text)
    cleaned = re.sub(r"[^A-Za-z0-9\-']+", " ", cleaned)
    return [
        token.strip("'\".,:;!?()[]{}")
        for token in cleaned.lower().replace("-", " ").split()
        if token.strip("'\".,:;!?()[]{}")
    ]


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
