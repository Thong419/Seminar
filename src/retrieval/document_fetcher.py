"""Document fetching and snippet extraction from search results."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re

import requests
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except Exception:  # pragma: no cover - optional dependency fallback
    Article = None  # type: ignore[assignment]

from src.retrieval.search_client import SearchResult


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    title: str
    url: str
    source: str
    content: str
    trust_score: float
    relevance_score: float


class DocumentFetcher:
    def __init__(self, timeout_seconds: int = 15, user_agent: str = "FakeNewsDetectionBot/1.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def fetch(self, result: SearchResult) -> EvidenceDocument:
        content = self._extract_content(result.url, result.snippet)
        return EvidenceDocument(
            title=result.title,
            url=result.url,
            source=result.source,
            content=content,
            trust_score=0.0,
            relevance_score=result.provider_relevance,
        )

    def _extract_content(self, url: str, fallback_snippet: str) -> str:
        if not url:
            return fallback_snippet.strip()

        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()

        if Article is not None:
            try:
                article = Article(url)
                article.download(input_html=response.text)
                article.parse()
                text = article.text.strip()
                if text:
                    return _compact_whitespace(text)
            except Exception:
                pass

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = _compact_whitespace(unescape(soup.get_text(" ")))
        return text or fallback_snippet.strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
