from __future__ import annotations

from pathlib import Path

from src.agent.classifier_tool import ClassifierTool
from src.agent.decision_tool import DecisionTool
from src.agent.evidence_tool import EvidenceTool
from src.agent.retrieval.retrieval_agent import RetrievalAgent
from src.inference.predictor import Prediction
from src.retrieval.claim_extractor import extract_claim
from src.retrieval.demo_provider import DemoDocumentFetcher, DemoSearchClient
from src.retrieval.pipeline import EvidenceRetrievalPipeline
from src.retrieval.query_generator import generate_search_queries


TRUMP_BANNON_ARTICLE = (
    "WASHINGTON (Reuters) - U.S. President Donald Trump removed his chief strategist Steve Bannon "
    "from the National Security Council on Wednesday. The White House said the change was part of a "
    "routine restructuring to improve coordination of foreign policy and national security decisions."
)

PERSEVERANCE_ARTICLE = "NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars."

CANCER_HOAX_ARTICLE = (
    "Scientists confirm that drinking warm salt water every morning can permanently cure all types of "
    "cancer within 3 days without any medical treatment."
)

SMARTPHONE_HOAX_ARTICLE = (
    "The government secretly announced that smartphones will be banned worldwide starting next month "
    "to reduce human dependency on technology."
)

class FakePredictor:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence

    def predict(self, text: str) -> Prediction:
        return Prediction(label=self.label, confidence=self.confidence)


def run_demo_pipeline(article_text: str, expected_label: str, confidence: float = 0.98) -> dict[str, object]:
    search_client = DemoSearchClient()
    fetcher = DemoDocumentFetcher()
    retrieval_path = Path("configs/retrieval.yaml")

    claim_result = extract_claim(article_text)
    generated_queries = generate_search_queries(
        article_text=article_text,
        claim=str(claim_result["claim"]),
        keywords=list(claim_result["keywords"]),
        claim_type=str(claim_result.get("claim_type", "general_claim")),
        max_queries=4,
    )

    pipeline = EvidenceRetrievalPipeline(
        search_client=search_client,
        document_fetcher=fetcher,
        retrieval_config_path=retrieval_path,
    )
    retrieval_bundle = pipeline.retrieve(article_text)

    retrieval_agent = RetrievalAgent(
        retrieval_config_path=retrieval_path,
        search_client=search_client,
        document_fetcher=fetcher,
    )
    evidence_tool = EvidenceTool(
        retrieval_config_path=retrieval_path,
        search_client=search_client,
        document_fetcher=fetcher,
        retrieval_agent=retrieval_agent,
    )
    classifier = ClassifierTool(FakePredictor(expected_label, confidence))
    decision_tool = DecisionTool()

    classification = classifier.run(article_text)
    evidence = evidence_tool.run(article_text)
    decision = decision_tool.decide(classification, evidence)

    return {
        "claim_result": claim_result,
        "queries": generated_queries,
        "retrieval_bundle": retrieval_bundle,
        "classification": classification,
        "evidence": evidence,
        "decision": decision,
    }


def _joined_sources_text(sources: list[dict[str, object]]) -> str:
    return " ".join(
        f"{item.get('title', '')} {item.get('content', '')}" for item in sources
    ).lower()


def _titles(items: list[dict[str, object]]) -> list[str]:
    return [str(item.get("title", "")) for item in items]


def _assert_debug_fields(items: list[dict[str, object]]) -> None:
    required = {"query_used", "semantic_score", "coverage_score", "entity_overlap", "predicate_overlap", "accepted", "rejection_reason"}
    for item in items:
        assert required.issubset(item.keys())


def test_demo_claim_real_trump_bannon_pipeline() -> None:
    outcome = run_demo_pipeline(TRUMP_BANNON_ARTICLE, expected_label="real")

    classification = outcome["classification"]
    evidence = outcome["evidence"]
    decision = outcome["decision"]
    accepted = evidence["accepted_evidence"]
    rejected = evidence["rejected_evidence"]
    sources = evidence["sources"]
    evidence_text = _joined_sources_text(sources)

    assert classification["label"] == "real"
    assert evidence["evidence_found"] is True
    assert decision["human_review_state"] == "REAL"
    assert evidence["support_score"] > evidence["contradiction_score"]
    assert "steve bannon" in evidence_text
    assert "national security council" in evidence_text
    assert "white house" in evidence_text
    assert "trump" in evidence_text
    assert "White House" not in _titles(sources)
    assert "Donald Trump" not in _titles(sources)
    assert "White House" in _titles(rejected)
    assert "Donald Trump" in _titles(rejected)
    _assert_debug_fields(accepted)
    _assert_debug_fields(rejected)


def test_demo_claim_real_perseverance_pipeline() -> None:
    outcome = run_demo_pipeline(PERSEVERANCE_ARTICLE, expected_label="real")

    classification = outcome["classification"]
    evidence = outcome["evidence"]
    decision = outcome["decision"]
    sources = evidence["sources"]
    accepted = evidence["accepted_evidence"]
    evidence_text = _joined_sources_text(sources)
    source_hosts = " ".join(str(item.get("source", "")).lower() for item in sources)

    assert classification["label"] == "real"
    assert evidence["evidence_found"] is True
    assert decision["human_review_state"] == "REAL"
    assert "perseverance" in evidence_text
    assert "jezero" in evidence_text
    assert "nasa" in evidence_text
    assert "sample" in evidence_text
    assert "nasa.gov" in source_hosts or any("nasa.gov" in str(item.get("url", "")).lower() for item in sources)
    assert evidence["support_score"] > evidence["contradiction_score"]
    assert "Mars" not in _titles(sources)
    _assert_debug_fields(accepted)


def test_demo_claim_fake_cancer_hoax_pipeline() -> None:
    outcome = run_demo_pipeline(CANCER_HOAX_ARTICLE, expected_label="fake")

    classification = outcome["classification"]
    evidence = outcome["evidence"]
    decision = outcome["decision"]
    accepted = evidence["accepted_evidence"]
    rejected = evidence["rejected_evidence"]
    sources = evidence["sources"]
    evidence_text = _joined_sources_text(sources)

    assert classification["label"] == "fake"
    assert evidence["evidence_found"] is True
    assert decision["human_review_state"] == "FAKE"
    assert evidence["contradiction_score"] > evidence["support_score"]
    assert "cancer" in evidence_text
    assert "medical treatment" in evidence_text
    assert "misinformation" in evidence_text
    assert "Great Salt Lake" not in _titles(sources)
    assert "Great Salt Lake" in _titles(rejected)
    assert all("utah" not in str(item.get("content", "")).lower() for item in sources)
    assert all("climate change" not in str(item.get("content", "")).lower() for item in sources)
    _assert_debug_fields(accepted)
    _assert_debug_fields(rejected)


def test_demo_claim_fake_smartphone_ban_pipeline() -> None:
    outcome = run_demo_pipeline(SMARTPHONE_HOAX_ARTICLE, expected_label="fake")

    classification = outcome["classification"]
    evidence = outcome["evidence"]
    decision = outcome["decision"]
    accepted = evidence["accepted_evidence"]
    rejected = evidence["rejected_evidence"]
    sources = evidence["sources"]
    evidence_text = _joined_sources_text(sources)

    assert classification["label"] == "fake"
    assert evidence["evidence_found"] is True
    assert decision["human_review_state"] == "FAKE"
    assert evidence["contradiction_score"] > evidence["support_score"]
    assert "fact-check" in evidence_text or "fact-checkers" in evidence_text
    assert "official" in evidence_text or "government policy" in evidence_text
    assert "Smartphone" not in _titles(sources)
    assert "Huawei" not in _titles(sources)
    assert "Nokia" not in _titles(sources)
    assert "WhatsApp" not in _titles(sources)
    assert {"Smartphone", "Huawei", "Nokia", "WhatsApp"}.issubset(set(_titles(rejected)))
    _assert_debug_fields(accepted)
    _assert_debug_fields(rejected)
