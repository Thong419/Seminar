"""Domain-based source credibility scoring."""

from __future__ import annotations

from urllib.parse import urlparse


KNOWN_HIGH_CREDIBILITY_DOMAINS: dict[str, float] = {
    # Government / international health
    "nasa.gov": 1.0,
    "jpl.nasa.gov": 1.0,
    "nih.gov": 1.0,
    "cdc.gov": 1.0,
    "cancer.gov": 0.98,
    "pubmed.ncbi.nlm.nih.gov": 0.98,
    "nlm.nih.gov": 0.98,
    "who.int": 0.97,
    "bls.gov": 0.97,
    "federalreserve.gov": 0.97,
    "congress.gov": 0.97,
    "whitehouse.gov": 0.97,
    "esa.int": 0.95,
    "imf.org": 0.95,
    "worldbank.org": 0.95,
    "noaa.gov": 0.95,
    "fda.gov": 0.95,
    # Wire services (highest journalistic credibility)
    "apnews.com": 0.95,
    "reuters.com": 0.95,
    # Fact-checkers
    "snopes.com": 0.92,
    "politifact.com": 0.92,
    "factcheck.org": 0.92,
    "fullfact.org": 0.92,
    "africacheck.org": 0.88,
    # High-quality science journals
    "nature.com": 0.97,
    "science.org": 0.97,
    "pnas.org": 0.95,
    "nejm.org": 0.97,
    "thelancet.com": 0.95,
    "bmj.com": 0.95,
    "jamanetwork.com": 0.95,
    "scientificamerican.com": 0.90,
    "newscientist.com": 0.88,
    "sciencedaily.com": 0.85,
    "phys.org": 0.82,
    # Medical reference
    "mayoclinic.org": 0.92,
    "webmd.com": 0.78,
    "healthline.com": 0.75,
    # Reputable press
    "bbc.com": 0.92,
    "bbc.co.uk": 0.92,
    "nytimes.com": 0.90,
    "theguardian.com": 0.90,
    "washingtonpost.com": 0.90,
    "economist.com": 0.92,
    "ft.com": 0.92,
    "wsj.com": 0.90,
    "bloomberg.com": 0.88,
    "time.com": 0.86,
    # Tech press
    "wired.com": 0.88,
    "arstechnica.com": 0.88,
    "ieee.org": 0.95,
    "acm.org": 0.95,
    "technologyreview.com": 0.90,
    "theverge.com": 0.82,
    "zdnet.com": 0.78,
    # Wikipedia is background only
    "wikipedia.org": 0.50,
}

# Minimum credibility thresholds per claim type for evidence ranking
DOMAIN_PRIORITY_THRESHOLD: dict[str, float] = {
    "medical_claim": 0.80,
    "science_claim": 0.80,
    "political_claim": 0.75,
    "technology_claim": 0.70,
    "economic_claim": 0.75,
    "general_claim": 0.60,
}


def score_source_credibility(source: str | None = None, url: str | None = None) -> float:
    """Score source credibility on a 0.0–1.0 scale using domain heuristics."""

    normalized_source = (source or "").strip().lower()
    normalized_host = _extract_host(url)

    if normalized_host in KNOWN_HIGH_CREDIBILITY_DOMAINS:
        return KNOWN_HIGH_CREDIBILITY_DOMAINS[normalized_host]
    for known_domain, score in KNOWN_HIGH_CREDIBILITY_DOMAINS.items():
        if normalized_host.endswith(f".{known_domain}"):
            return score

    # TLD-based fallback
    if normalized_host.endswith(".gov"):
        return 0.97
    if normalized_host.endswith(".edu"):
        return 0.90
    if normalized_host.endswith(".int"):
        return 0.88

    # Source name fallback
    if normalized_source in {"reuters", "reuters.com"}:
        return 0.95
    if normalized_source in {"ap news", "apnews", "apnews.com"}:
        return 0.95
    if normalized_source in {"wikipedia", "wikipedia.org"}:
        return 0.50

    if normalized_host:
        return 0.42
    if normalized_source:
        return 0.38
    return 0.32


def is_high_credibility_for_claim_type(url: str | None, claim_type: str) -> bool:
    """Return True if the source meets the minimum threshold for the claim type."""
    score = score_source_credibility(url=url)
    threshold = DOMAIN_PRIORITY_THRESHOLD.get(claim_type, 0.60)
    return score >= threshold


def _extract_host(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host
