"""Lightweight claim extraction for evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass
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
}

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    claim: str
    keywords: list[str]

    def as_dict(self) -> dict[str, list[str] | str]:
        return {"claim": self.claim, "keywords": self.keywords}


def extract_claim(article_text: str) -> dict[str, list[str] | str]:
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(article_text) if sentence.strip()]
    if not sentences:
        return {"claim": article_text.strip(), "keywords": []}

    claim = max(sentences, key=_sentence_score)
    keywords = _extract_keywords(claim)
    return ClaimExtractionResult(claim=claim, keywords=keywords).as_dict()


def _sentence_score(sentence: str) -> int:
    tokens = WORD_PATTERN.findall(sentence.lower())
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    return len(content_tokens) + min(len(sentence), 120) // 30


def _extract_keywords(text: str, limit: int = 6) -> list[str]:
    tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
    keywords: list[str] = []
    for token in tokens:
        if token in STOPWORDS or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords
