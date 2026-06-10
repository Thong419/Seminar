"""Search query generation from extracted claims and keywords."""

from __future__ import annotations

from collections.abc import Iterable


def generate_search_queries(
    article_text: str,
    claim: str | None = None,
    keywords: list[str] | None = None,
    max_queries: int = 4,
) -> list[str]:
    claim = claim or article_text
    keywords = keywords or []

    tokens = _dedupe([token for token in _normalize_tokens(claim)] + keywords)
    queries: list[str] = []

    if tokens:
        queries.append(" ".join(tokens[: min(len(tokens), 6)]))
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[:3]) + " research study")
    if tokens:
        queries.append(" ".join(tokens[: min(len(tokens), 5)]) + " fact check")
    if len(tokens) >= 2:
        queries.append(" ".join(tokens[:2]) + " evidence")
    if len(tokens) >= 4:
        queries.append(" ".join(tokens[:4]) + " source")

    fallback = article_text[:120].strip()
    if fallback:
        queries.append(fallback)

    return _dedupe(queries)[:max_queries]


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in text.lower().replace("-", " ").split() if token]


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
