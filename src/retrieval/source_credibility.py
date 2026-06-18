"""Domain-based source credibility scoring."""

from __future__ import annotations

from urllib.parse import urlparse


KNOWN_HIGH_CREDIBILITY_DOMAINS = {
    "nasa.gov": 1.0,
    "jpl.nasa.gov": 1.0,
    "nih.gov": 1.0,
    "cdc.gov": 1.0,
    "who.int": 0.95,
    "apnews.com": 0.95,
    "reuters.com": 0.95,
    "bbc.com": 0.92,
    "nytimes.com": 0.92,
    "theguardian.com": 0.9,
    "wikipedia.org": 0.55,
}


def score_source_credibility(source: str | None = None, url: str | None = None) -> float:
    """Score source credibility on a 0.0-1.0 scale using domain heuristics."""

    normalized_source = (source or "").strip().lower()
    normalized_host = _extract_host(url)

    if normalized_host in KNOWN_HIGH_CREDIBILITY_DOMAINS:
        return KNOWN_HIGH_CREDIBILITY_DOMAINS[normalized_host]
    if normalized_host.endswith(".gov"):
        return 0.98
    if normalized_host.endswith(".edu"):
        return 0.92
    if normalized_host.endswith(".int"):
        return 0.9
    if normalized_host.endswith("wikipedia.org"):
        return 0.55

    if normalized_source in {"reuters", "reuters.com"}:
        return 0.95
    if normalized_source in {"ap news", "apnews", "apnews.com"}:
        return 0.95
    if normalized_source in {"wikipedia", "wikipedia.org"}:
        return 0.55
    if normalized_source.endswith(".gov"):
        return 0.98

    if normalized_host:
        return 0.45
    if normalized_source:
        return 0.4
    return 0.35


def _extract_host(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host
