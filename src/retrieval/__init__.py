"""Real evidence retrieval package."""

from src.retrieval.claim_extractor import ClaimExtractionResult, extract_claim
from src.retrieval.document_fetcher import DocumentFetcher, EvidenceDocument
from src.retrieval.evidence_ranker import RankConfig, EvidenceRanker, rank_evidence
from src.retrieval.pipeline import EvidenceRetrievalPipeline, RetrievalBundle
from src.retrieval.query_generator import generate_search_queries
from src.retrieval.search_client import (
    BingSearchClient,
    GoogleSearchClient,
    SearchClient,
    SearchResult,
    SerpApiSearchClient,
    TavilySearchClient,
)
