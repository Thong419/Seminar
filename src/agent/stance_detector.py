"""Stance detection: classify evidence as SUPPORT, REFUTE, or NEUTRAL against a claim.

This replaces the naive token-overlap heuristic in evidence_tool.py.
The key insight is:
    - A Wikipedia page about "Water" must NOT support a cancer-cure claim just
      because "water" appears in both texts.
    - SUPPORT requires (a) the content addresses the claim's core assertion AND
      (b) explicit confirmation language is present.
    - REFUTE requires (a) coverage of the claim topic AND (b) explicit denial /
      debunking language.
    - NEUTRAL is the default for topically related but non-conclusive content.

Usage:
    >>> from src.agent.stance_detector import detect_stance
    >>> detect_stance("Warm salt water cures cancer", "Water is H2O.")
    'NEUTRAL'
    >>> detect_stance("Warm salt water cures cancer", "There is no evidence that salt water cures cancer. Medical authorities state this is false.")
    'REFUTE'
"""

from __future__ import annotations

import re
from typing import Literal

StanceLabel = Literal["SUPPORT", "REFUTE", "NEUTRAL"]

# ─── Stance signal patterns ────────────────────────────────────────────────────

_SUPPORT_PATTERNS: list[str] = [
    r"\bconfirm(s|ed|ing)?\b",
    r"\bverif(y|ied|ies|ying)\b",
    r"\bprove[sdn]?\b",
    r"\bproven\b",
    r"\bdemonstrat(e[sd]?|ing)\b",
    r"\bvalidat(e[sd]?|ing)\b",
    r"\bstudy (found|shows?|confirm|suggest)\b",
    r"\bresearch (found|shows?|confirm|suggest|supports?|indicates?)\b",
    r"\b(research|evidence|data|studies) (supports?|shows?|suggests?|indicates?|demonstrates?)\b",
    r"\bscientists? (found|confirm|show|report|suggest)\b",
    r"\b(is|are|was|were) (true|accurate|correct|factual)\b",
    r"\bhas been (confirmed|verified|proven|established)\b",
    r"\bwell.?established\b",
    r"\bevidence (supports?|shows?|indicates?|suggests?|found)\b",
    r"\beffective(ly)?\b",
    r"\bworks?\b",
    r"\bsuccessful(ly)?\b",
    r"\bbeneficial\b",
    r"\brecommend(s|ed)?\b",
    r"\bapproved by\b",
    r"\blinked to\b",
    r"\bassociated with\b",
    r"\bsupport(s|ed) (the|this|an?) (claim|finding|evidence|association|link)\b",
]

_REFUTE_PATTERNS: list[str] = [
    r"\bno evidence\b",
    r"\bnot true\b",
    r"\bfalse claim\b",
    r"\bmisinformation\b",
    r"\bdebunk(s|ed|ing)?\b",
    r"\bfact.?check\b",
    r"\brefut(e[sd]?|ing)\b",
    r"\bdisprov(e[sdn]?|ing)\b",
    r"\bcontradicts?\b",
    r"\bhoax\b",
    r"\bfake news\b",
    r"\bunsubstantiated\b",
    r"\blacks? (evidence|support|scientific backing)\b",
    r"\bnever been (proven|confirmed|shown|demonstrated)\b",
    r"\bcannot (cure|treat|prevent|heal)\b",
    r"\bdoes not (cure|treat|prevent|work|help)\b",
    r"\bis not (effective|proven|supported|evidence.based)\b",
    r"\bno scientific (basis|evidence|support|consensus)\b",
    r"\bpseudoscience\b",
    r"\bquackery\b",
    r"\bwithout (scientific )?evidence\b",
    r"\bnot supported by\b",
    r"\bmisleading\b",
    r"\buntrue\b",
    r"\bincorrect\b",
    r"\binaccurate\b",
    r"\bspreading (false|misinformation)\b",
    r"\bexpert[s]? (warn|dismiss|reject|disagree)\b",
    r"\bno (proven|known) (treatment|cure|effect)\b",
    r"\bhas not been (proven|confirmed|established)\b",
    r"\bcritically ill\b",
    r"\bdanger(ous)?\b",
    r"\bharm(ful)?\b",
]

# Terms that are stopwords in the claim context
_ASSERTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "that", "this", "these", "those",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "it", "its", "they", "their", "he", "she", "we", "you", "i",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "can", "all", "every", "each", "any", "some", "no",
        "said", "say", "says", "claim", "claims", "according",
        "confirm", "that", "new", "report", "study", "show",
        "also", "which", "who", "what", "when", "where", "how",
        "not", "only", "just", "more", "most", "very", "than",
        "there", "here", "then", "now", "about", "over", "such",
    }
)


# Strong debunking terms that immediately indicate REFUTE when the claim is covered,
# regardless of incidental support language in the same document.
_STRONG_REFUTE_PATTERNS: list[str] = [
    r"\bpseudoscience\b",
    r"\bhoax\b",
    r"\bfake.?news\b",
    r"\bdebunk(s|ed|ing)?\b",
    r"\bmisinformation\b",
    r"\bquackery\b",
    r"\bno scientific (basis|evidence|support|consensus)\b",
    r"\bunsubstantiated\b",
    r"\bpropaganda\b",
    r"\bconspiracy.?theory\b",
]

# ─── Public API ────────────────────────────────────────────────────────────────


def detect_stance(claim: str, evidence_content: str) -> StanceLabel:
    """Determine whether *evidence_content* SUPPORTS, REFUTEs, or is NEUTRAL
    with respect to *claim*.

    Algorithm:
    1. Extract the core assertion terms from the claim (meaningful nouns / verbs).
    2. Compute how many of those terms appear in the evidence content (coverage).
    3. If coverage < threshold → NEUTRAL (content doesn't discuss the claim at all).
    4. Count support and refute signal patterns in the evidence.
    5. Return the dominant stance; default to NEUTRAL on tie or low signals.

    This correctly handles:
    - "Water" Wikipedia page vs cancer-cure claim  → NEUTRAL  (low coverage)
    - "Great Salt Lake" vs cancer-cure claim        → NEUTRAL  (no assertion terms)
    - Debunking article for the same claim          → REFUTE   (explicit patterns)
    """
    core_terms = _extract_core_assertion_terms(claim)
    content_lower = evidence_content.lower()

    coverage = _assertion_coverage(core_terms, content_lower)

    # Content doesn't even address the claim's subject → NEUTRAL
    if coverage < 0.20:
        return "NEUTRAL"

    # Strong debunking signal: if the document contains clear debunking language
    # AND covers the claim's topic, treat as REFUTE regardless of incidental
    # positive language (e.g., "confirmed by satellite" about real science
    # appearing in the same article that also debunks a flat-earth claim).
    strong_refute = _count_patterns(content_lower, _STRONG_REFUTE_PATTERNS)
    if strong_refute >= 1:
        return "REFUTE"

    support_score = _count_patterns(content_lower, _SUPPORT_PATTERNS)
    refute_score = _count_patterns(content_lower, _REFUTE_PATTERNS)

    if refute_score > support_score and refute_score >= 1:
        return "REFUTE"
    if support_score > refute_score and support_score >= 1 and coverage >= 0.25:
        return "SUPPORT"
    return "NEUTRAL"


def detect_stance_batch(
    claim: str, evidence_items: list[dict]
) -> list[StanceLabel]:
    """Apply detect_stance to a list of evidence item dicts (must have 'content' key)."""
    return [detect_stance(claim, item.get("content", "")) for item in evidence_items]


def stance_summary(stances: list[StanceLabel]) -> dict[str, int]:
    """Count occurrences of each stance label."""
    return {
        "SUPPORT": stances.count("SUPPORT"),
        "REFUTE": stances.count("REFUTE"),
        "NEUTRAL": stances.count("NEUTRAL"),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _extract_core_assertion_terms(claim: str) -> list[str]:
    """Extract semantically meaningful tokens from the claim, excluding stopwords."""
    tokens = re.findall(r"[a-z]+", claim.lower())
    return [t for t in tokens if t not in _ASSERTION_STOPWORDS and len(t) > 3]


def _assertion_coverage(core_terms: list[str], content_lower: str) -> float:
    """Return fraction of core claim terms that appear in *content_lower*."""
    if not core_terms:
        return 0.0
    hits = sum(1 for term in core_terms if term in content_lower)
    return hits / len(core_terms)


def _count_patterns(text: str, patterns: list[str]) -> int:
    """Count how many patterns from the list match in *text*."""
    return sum(1 for p in patterns if re.search(p, text))
