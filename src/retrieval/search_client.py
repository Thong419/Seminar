"""Provider abstraction for search APIs used by the evidence retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import os

import requests


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    source: str
    snippet: str
    provider_relevance: float = 0.5


class SearchClient(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search the configured provider and return structured results."""


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
