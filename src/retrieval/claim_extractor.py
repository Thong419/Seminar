"""Structured claim extraction for evidence retrieval.

Extracts:
    - claim: most informative sentence
    - keywords: content-bearing tokens
    - subject: likely subject of the claim
    - predicate: main action/relation
    - object_: target of the action
    - temporal: time expressions (e.g. "every morning", "within 3 days")
    - quantities: numeric expressions (e.g. "3 days", "100%")
    - claim_type: domain classification (medical, science, political, …)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "his", "her", "their", "its", "our", "your", "my",
    "mr", "mrs", "ms", "dr",
}

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")
ENTITY_PATTERN = re.compile(r"\b(?:[A-Z]{2,}(?:[-'][A-Z0-9]+)*|[A-Z][a-z]+(?:[-'][A-Z0-9]+)*)\b")

# Temporal expression patterns
_TEMPORAL_PATTERN = re.compile(
    r"\b("
    r"every (morning|evening|night|day|week|month|year)|"
    r"daily|weekly|monthly|annually|"
    r"within \d+ (hour|day|week|month|year)s?|"
    r"in \d+ (hour|day|week|month|year)s?|"
    r"after \d+ (hour|day|week|month|year)s?|"
    r"(this|last|next) (year|month|week|day)|"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r")\b",
    re.IGNORECASE,
)

# Quantity patterns
_QUANTITY_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(percent|%|million|billion|thousand|kg|km|miles?|days?|weeks?|months?|years?|hours?|"
    r"times?|doses?|mg|ml|units?|cases?|patients?|people|countries|nations)\b",
    re.IGNORECASE,
)

# Action verbs suggesting a predicate
_ACTION_PATTERN = re.compile(
    r"\b(cure[sd]?|treat(s|ed)?|prevent[s]?|cause[sd]?|confirm(s|ed)?|prove[sd]?|show(s|ed)?|"
    r"increase[sd]?|decrease[sd]?|reduce[sd]?|discover(ed|s)?|claim(s|ed)?|announce[sd]?|"
    r"warn(s|ed)?|ban(ned|s)?|approve[sd]?|reject(s|ed)?|deny|denies|denied|"
    r"remove[sd]?|collect(s|ed)?|gather(s|ed)?|declare[sd]?|"
    r"launch(es|ed)?|release[sd]?|develop(s|ed)?|create[sd]?|build[s]?|built|"
    r"kill(s|ed)?|harm(s|ed)?|help(s|ed)?|boost(s|ed)?|link(s|ed)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    claim: str
    keywords: list[str]
    subject: str = ""
    predicate: str = ""
    object_: str = ""
    temporal: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)
    claim_type: str = "general_claim"

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "keywords": self.keywords,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "temporal": self.temporal,
            "quantities": self.quantities,
            "claim_type": self.claim_type,
        }


def extract_claim(article_text: str) -> dict:
    """Extract structured claim components from article text.

    Returns a dict with keys: claim, keywords, subject, predicate, object,
    temporal, quantities, claim_type.
    """
    sentences = [s.strip() for s in SENTENCE_PATTERN.split(article_text) if s.strip()]
    if not sentences:
        return ClaimExtractionResult(claim=article_text.strip(), keywords=[]).as_dict()

    claim = max(sentences, key=_sentence_score)
    keywords = _extract_keywords(claim)
    temporal = [m.group(0) for m in _TEMPORAL_PATTERN.finditer(claim)]
    quantities = [m.group(0) for m in _QUANTITY_PATTERN.finditer(claim)]
    subject, predicate, object_ = _extract_spo(claim)

    # lazy import to avoid circular dependency
    from src.agent.claim_classifier import classify_claim_type
    claim_type = classify_claim_type(article_text)

    return ClaimExtractionResult(
        claim=claim,
        keywords=keywords,
        subject=subject,
        predicate=predicate,
        object_=object_,
        temporal=temporal,
        quantities=quantities,
        claim_type=claim_type,
    ).as_dict()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _sentence_score(sentence: str) -> int:
    tokens = WORD_PATTERN.findall(sentence.lower())
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    action_matches = len(_ACTION_PATTERN.findall(sentence))
    has_quantity = 1 if _QUANTITY_PATTERN.search(sentence) else 0
    entity_bonus = min(4, len([m.group(0) for m in ENTITY_PATTERN.finditer(sentence)]))
    return len(content_tokens) + min(len(sentence), 120) // 40 + 5 * action_matches + 2 * entity_bonus + 2 * has_quantity


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [t.lower() for t in WORD_PATTERN.findall(text)]
    keywords: list[str] = []
    for token in tokens:
        if token in STOPWORDS or token in keywords or len(token) < 3:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _extract_spo(sentence: str) -> tuple[str, str, str]:
    """Very lightweight SPO extraction: first noun phrase, first verb, rest."""
    words = WORD_PATTERN.findall(sentence)
    if not words:
        return "", "", ""

    predicate_match = _ACTION_PATTERN.search(sentence)
    if not predicate_match:
        return words[0] if words else "", "", ""

    predicate = predicate_match.group(0)
    pred_pos = sentence.lower().find(predicate.lower())

    subject_part = sentence[:pred_pos].strip()
    object_part = sentence[pred_pos + len(predicate):].strip()

    subject_tokens = [t for t in WORD_PATTERN.findall(subject_part) if t.lower() not in STOPWORDS]
    object_tokens = [t for t in WORD_PATTERN.findall(object_part) if t.lower() not in STOPWORDS]

    subject = " ".join(subject_tokens[:4])
    object_ = " ".join(object_tokens[:6])
    return subject, predicate, object_
