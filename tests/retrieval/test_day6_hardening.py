from __future__ import annotations

from src.agent.decision_tool import DecisionTool
from src.agent.state import ReviewState
from src.retrieval.document_fetcher import EvidenceDocument
from src.retrieval.relevance_filter import dedupe_documents, score_claim_relevance


def _doc(title: str, url: str, content: str) -> EvidenceDocument:
    return EvidenceDocument(
        title=title,
        url=url,
        source="wikipedia.org",
        content=content,
        trust_score=0.5,
        relevance_score=0.8,
    )


def _source(stance: str, credibility: float = 0.9, relevance: float = 0.8) -> dict[str, object]:
    return {
        "title": f"{stance} source",
        "source": "test",
        "content": "test content",
        "stance": stance,
        "source_credibility": credibility,
        "relevance_score": relevance,
    }


def test_day6_dedupe_documents_by_url_and_similar_title() -> None:
    docs = [
        _doc("Water", "https://en.wikipedia.org/wiki/Water", "Water is H2O"),
        _doc("Water", "https://en.wikipedia.org/wiki/Water/", "Water duplicate by URL"),
        _doc("Water - Wikipedia", "https://en.wikipedia.org/wiki/Water_(chemistry)", "Water duplicate by title"),
    ]

    deduped = dedupe_documents(docs)

    assert len(deduped) == 1


def test_day6_relevance_rejects_generic_water_page_for_specific_claim() -> None:
    claim_data = {
        "subject": "drinking warm salt water",
        "predicate": "cures",
        "object": "all types of cancer",
    }
    water_doc = _doc(
        "Water",
        "https://en.wikipedia.org/wiki/Water",
        "Water is a chemical compound with formula H2O and is essential for life.",
    )

    relevance = score_claim_relevance(claim_data, water_doc)

    assert relevance.accepted is False
    assert relevance.claim_coverage_score < 0.30
    assert relevance.rejection_reason in {"generic_or_concept_page", "missing_claim_action", "missing_claim_object"}


def test_day6_relevance_accepts_assertion_focused_refute_page() -> None:
    claim_data = {
        "subject": "drinking warm salt water",
        "predicate": "cures",
        "object": "all types of cancer",
    }
    refute_doc = _doc(
        "No evidence salt water cures cancer",
        "https://example.org/medical-fact-check",
        (
            "There is no scientific evidence that drinking warm salt water cures cancer. "
            "Medical authorities refute this claim and state there is no proven cure from salt water."
        ),
    )

    relevance = score_claim_relevance(claim_data, refute_doc)

    assert relevance.accepted is True
    assert relevance.action_overlap_score > 0.2
    assert relevance.claim_coverage_score >= 0.30


def test_day6_decision_rule_a_sets_uncertain() -> None:
    tool = DecisionTool()
    classification = {"label": "fake", "confidence": 0.92}
    evidence = {
        "evidence_found": True,
        "sources": [_source("support"), _source("support")],
        "support_score": 0.82,
        "contradiction_score": 0.10,
        "source_credibility_score": 0.88,
        "evidence_quality_score": 0.78,
        "conflict_flag": False,
    }

    result = tool.decide(classification, evidence)

    assert result["conflict_flag"] is True
    assert result["human_review_state"] == ReviewState.uncertain.value
    assert "Rule A" in result["decision_reason"]


def test_day6_decision_rule_b_sets_uncertain() -> None:
    tool = DecisionTool()
    classification = {"label": "real", "confidence": 0.93}
    evidence = {
        "evidence_found": True,
        "sources": [_source("refute"), _source("refute")],
        "support_score": 0.12,
        "contradiction_score": 0.79,
        "source_credibility_score": 0.90,
        "evidence_quality_score": 0.80,
        "conflict_flag": False,
    }

    result = tool.decide(classification, evidence)

    assert result["conflict_flag"] is True
    assert result["human_review_state"] == ReviewState.uncertain.value
    assert "Rule B" in result["decision_reason"]


def test_day6_decision_aligned_fake_keeps_fake_state() -> None:
    tool = DecisionTool()
    classification = {"label": "fake", "confidence": 0.95}
    evidence = {
        "evidence_found": True,
        "sources": [_source("refute"), _source("refute")],
        "support_score": 0.08,
        "contradiction_score": 0.83,
        "source_credibility_score": 0.92,
        "evidence_quality_score": 0.84,
        "conflict_flag": False,
    }

    result = tool.decide(classification, evidence)

    assert result["conflict_flag"] is False
    assert result["human_review_state"] == ReviewState.fake.value
