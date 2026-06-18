"""Provider abstraction for search APIs used by the evidence retrieval layer."""

from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, urlparse
from typing import Protocol
import os

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    source: str
    snippet: str
    provider_relevance: float = 0.5
    query: str | None = None
    provider: str | None = None


class SearchClient(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search the configured provider and return structured results."""


class WikipediaHtmlSearchClient:
    def __init__(self, timeout_seconds: int = 15, user_agent: str = "FakeNewsDetectionBot/1.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []

        response = requests.get(
            "https://en.wikipedia.org/w/index.php",
            params={"search": query},
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.user_agent},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for rank, item in enumerate(soup.select("li.mw-search-result")[:limit]):
            anchor = item.select_one(".mw-search-result-heading a")
            snippet = item.select_one(".searchresult")
            if not anchor or not anchor.get("href"):
                continue

            title = _compact_whitespace(anchor.get_text(" ", strip=True))
            url = anchor.get("href", "")
            if url.startswith("/"):
                url = f"https://en.wikipedia.org{url}"

            snippet_text = _compact_whitespace(unescape(snippet.get_text(" ", strip=True) if snippet else ""))
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    source="Wikipedia",
                    snippet=snippet_text,
                    provider_relevance=max(0.15, 1.0 - (rank * 0.12)),
                    query=query,
                    provider="wikipedia_html",
                )
            )
        return results


class BingHtmlSearchClient:
    def __init__(self, timeout_seconds: int = 15, user_agent: str = "FakeNewsDetectionBot/1.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query},
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.user_agent},
        )
        response.raise_for_status()

        response.encoding = response.encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for rank, item in enumerate(soup.select("li.b_algo")[:limit]):
            anchor = item.select_one("h2 a")
            snippet = item.select_one(".b_caption p")
            if not anchor or not anchor.get("href"):
                continue

            url = _unwrap_bing_redirect(anchor.get("href", ""))
            title = _compact_whitespace(anchor.get_text(" ", strip=True))
            snippet_text = _compact_whitespace(unescape(snippet.get_text(" ", strip=True) if snippet else ""))
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    source=_source_from_url(url),
                    snippet=snippet_text,
                    provider_relevance=max(0.15, 1.0 - (rank * 0.12)),
                    query=query,
                    provider="bing_html",
                )
            )
        return results


class HybridSearchClient:
    def __init__(self, clients: list[SearchClient]) -> None:
        self.clients = clients

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for client in self.clients:
            try:
                results = client.search(query, limit=limit)
            except Exception:
                continue
            for result in results:
                key = result.url or result.title.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(result)
                if len(merged) >= limit:
                    return merged
        return merged


class TavilySearchClient:
    def __init__(self, api_url: str, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self.api_url = api_url
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required for the Tavily search client.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
        }
        response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:limit]:
            results.append(
                SearchResult(
                    title=str(item.get("title", "Untitled")),
                    url=str(item.get("url", "")),
                    source=str(item.get("source", item.get("title", "unknown"))),
                    snippet=str(item.get("content", item.get("snippet", ""))),
                    provider_relevance=float(item.get("score", 0.5)),
                )
            )
        return results


class SerpApiSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError("SerpAPI support will be added later.")


class BingSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError("Bing support will be added later.")


class GoogleSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError("Google support will be added later.")


def _unwrap_bing_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/a"):
        query = parse_qs(parsed.query)
        payload = query.get("u", [""])[0]
        decoded = _decode_bing_payload(payload)
        if decoded.startswith("http"):
            return decoded
    return url


def _decode_bing_payload(payload: str) -> str:
    if not payload:
        return ""
    if payload.startswith("a1"):
        payload = payload[2:]
    padded = payload + ("=" * (-len(payload) % 4))
    try:
        return b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _source_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        return host.split(":")[0]
    return "unknown"


def _compact_whitespace(text: str) -> str:
    return " ".join(text.split())
