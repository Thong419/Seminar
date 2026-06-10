from src.retrieval.document_fetcher import EvidenceDocument
from src.retrieval.evidence_ranker import RankConfig, rank_evidence


def test_rank_evidence_prioritizes_trusted_relevant_sources() -> None:
    evidence = [
        EvidenceDocument(
            title="Low trust blog",
            url="https://example.com/1",
            source="Unknown Blog",
            content="This claim appears unverified and speculative.",
            trust_score=0.2,
            relevance_score=0.7,
        ),
        EvidenceDocument(
            title="Reuters fact report",
            url="https://example.com/2",
            source="Reuters",
            content="Reuters reports a verified analysis contradicting the claim.",
            trust_score=1.0,
            relevance_score=0.9,
        ),
    ]

    ranked = rank_evidence(
        claim="Scientists discover miracle cancer cure",
        evidence=evidence,
        trust_scores={"Reuters": 1.0, "Unknown Blog": 0.2},
        config=RankConfig(top_k=2),
    )

    assert ranked[0].source == "Reuters"
    assert 0.0 <= ranked[0].relevance_score <= 1.0
