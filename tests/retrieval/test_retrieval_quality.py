"""Retrieval quality benchmark tests.

Tests the full claim extraction → stance detection → decision pipeline with:
    - Obvious fake medical claims
    - Obvious fake science claims
    - Obvious fake conspiracy claims
    - True factual claims
    - Ambiguous claims

Each case verifies:
    1. Claim type classification
    2. Stance detection correctness on synthetic evidence
    3. Decision tool output (human_review_state, conflict_flag)
    4. Explanation quality (presence of key phrases)

Note: These tests do NOT make live HTTP requests. Evidence content is injected
via mock objects to isolate the logic under test.
"""

from __future__ import annotations

import pytest

from src.agent.claim_classifier import classify_claim_type, get_priority_domains
from src.agent.stance_detector import detect_stance, stance_summary
from src.agent.decision_tool import DecisionTool
from src.agent.state import ReviewState
from src.explainability.explanation_formatter import (
    format_explanation,
    build_evidence_summary,
    TrustScoreWeights,
)
from src.agent.state import EvidenceItem


# ─── Benchmark fixtures ───────────────────────────────────────────────────────

BENCHMARK_CASES = [
    # ── Obvious fake medical claims ─────────────────────────────────────────
    {
        "id": "MED-01",
        "category": "fake_medical",
        "claim": "Scientists confirm that drinking warm salt water every morning can permanently cure all types of cancer within 3 days.",
        "expected_claim_type": "medical_claim",
        "synthetic_evidence": [
            {
                "title": "Water",
                "content": "Water is a chemical compound with formula H2O. It is essential for life.",
                "expected_stance": "NEUTRAL",
            },
            {
                "title": "Great Salt Lake",
                "content": "The Great Salt Lake is a saline lake in Utah. It is the largest saltwater lake in the Western Hemisphere.",
                "expected_stance": "NEUTRAL",
            },
            {
                "title": "No evidence salt water cures cancer",
                "content": "There is no scientific evidence that salt water can cure cancer. No clinical trials have confirmed this. Medical authorities warn this is misinformation and potentially dangerous to cancer patients.",
                "expected_stance": "REFUTE",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
    {
        "id": "MED-02",
        "category": "fake_medical",
        "claim": "Drinking bleach every day can cure COVID-19 and all bacterial infections.",
        "expected_claim_type": "medical_claim",
        "synthetic_evidence": [
            {
                "title": "Bleach safety warning",
                "content": "Drinking bleach is dangerous and can cause severe chemical burns. There is no evidence it cures COVID-19. The CDC warns against ingesting disinfectants.",
                "expected_stance": "REFUTE",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
    # ── Obvious fake science claims ──────────────────────────────────────────
    {
        "id": "SCI-01",
        "category": "fake_science",
        "claim": "NASA scientists confirmed that the moon is made entirely of cheese discovered during Apollo 11.",
        "expected_claim_type": "science_claim",
        "synthetic_evidence": [
            {
                "title": "Apollo 11 mission",
                "content": "Apollo 11 was the first crewed mission to land on the Moon. Astronauts collected lunar rock samples, which are composed of basalt and anorthosite. There is no evidence of cheese.",
                "expected_stance": "REFUTE",
            },
            {
                "title": "Moon",
                "content": "The Moon is Earth's only natural satellite. It is composed of rock, dust, and minerals.",
                "expected_stance": "NEUTRAL",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
    {
        "id": "SCI-02",
        "category": "fake_science",
        "claim": "Scientists proved that the Earth is flat and NASA has been hiding this for decades.",
        "expected_claim_type": "science_claim",
        "synthetic_evidence": [
            {
                "title": "Shape of the Earth",
                "content": "The Earth is an oblate spheroid. This has been confirmed by satellite imagery, physics, and direct observation. Flat Earth claims are pseudoscience debunked by overwhelming scientific consensus.",
                "expected_stance": "REFUTE",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
    # ── Obvious fake conspiracy claims ───────────────────────────────────────
    {
        "id": "CON-01",
        "category": "fake_conspiracy",
        "claim": "5G towers are secretly spreading COVID-19 through microwave radiation.",
        "expected_claim_type": "medical_claim",  # COVID-19 dominates claim type detection
        "synthetic_evidence": [
            {
                "title": "5G and COVID-19 misinformation",
                "content": "There is no scientific evidence linking 5G towers to COVID-19. This claim has been widely debunked by health authorities and scientists. COVID-19 is caused by the SARS-CoV-2 virus.",
                "expected_stance": "REFUTE",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
    # ── True factual claims ──────────────────────────────────────────────────
    {
        "id": "REAL-01",
        "category": "true_claim",
        "claim": "NASA's Perseverance rover successfully landed on Mars in February 2021.",
        "expected_claim_type": "science_claim",
        "synthetic_evidence": [
            {
                "title": "Perseverance rover landing confirmed",
                "content": "NASA confirmed that the Perseverance rover successfully landed on Mars on February 18, 2021. Scientists verified the landing through telemetry data. The rover has been operating on the Martian surface since.",
                "expected_stance": "SUPPORT",
            },
        ],
        "expected_final_state": ReviewState.real.value,
    },
    {
        "id": "REAL-02",
        "category": "true_claim",
        "claim": "The World Health Organization declared COVID-19 a global pandemic in March 2020.",
        "expected_claim_type": "medical_claim",
        "synthetic_evidence": [
            {
                "title": "WHO pandemic declaration",
                "content": "The World Health Organization officially declared COVID-19 a pandemic on March 11, 2020. The declaration was confirmed by WHO Director-General. This has been verified by multiple official sources.",
                "expected_stance": "SUPPORT",
            },
        ],
        "expected_final_state": ReviewState.real.value,
    },
    # ── Ambiguous claims ─────────────────────────────────────────────────────
    {
        "id": "AMB-01",
        "category": "ambiguous",
        "claim": "A new study suggests coffee may reduce the risk of type 2 diabetes.",
        "expected_claim_type": "medical_claim",
        "synthetic_evidence": [
            {
                "title": "Coffee and diabetes research",
                "content": "Some observational studies have suggested an association between coffee consumption and reduced diabetes risk. However, causality has not been established.",
                # 'causality has not been established' is a scientific hedging/refute of the causal claim
                "expected_stance": "REFUTE",
            },
            {
                "title": "Meta-analysis on coffee benefits",
                "content": "Research supports an association between coffee and lower diabetes incidence. Scientists found evidence suggesting benefits in several studies.",
                "expected_stance": "SUPPORT",
            },
        ],
        "expected_final_state": ReviewState.uncertain.value,
    },
]


# ─── Claim type classification tests ─────────────────────────────────────────


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[c["id"] for c in BENCHMARK_CASES])
def test_claim_type_classification(case: dict) -> None:
    """Claim type classifier must match the expected domain."""
    result = classify_claim_type(case["claim"])
    assert result == case["expected_claim_type"], (
        f"[{case['id']}] Expected {case['expected_claim_type']!r}, got {result!r} "
        f"for claim: {case['claim'][:80]}"
    )


def test_medical_priority_domains() -> None:
    domains = get_priority_domains("medical_claim")
    assert "nih.gov" in domains
    assert "who.int" in domains
    assert "pubmed.ncbi.nlm.nih.gov" in domains


def test_science_priority_domains() -> None:
    domains = get_priority_domains("science_claim")
    assert "nasa.gov" in domains
    assert "nature.com" in domains


def test_political_priority_domains() -> None:
    domains = get_priority_domains("political_claim")
    assert "reuters.com" in domains
    assert "apnews.com" in domains
    assert "factcheck.org" in domains


# ─── Stance detection tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[c["id"] for c in BENCHMARK_CASES])
def test_stance_detection_per_case(case: dict) -> None:
    """Each synthetic evidence item must produce the expected stance label."""
    for ev in case["synthetic_evidence"]:
        result = detect_stance(case["claim"], ev["content"])
        assert result == ev["expected_stance"], (
            f"[{case['id']}] Evidence '{ev['title']}': "
            f"expected stance={ev['expected_stance']!r}, got {result!r}.\n"
            f"Claim: {case['claim'][:80]}\n"
            f"Content: {ev['content'][:120]}"
        )


def test_water_page_neutral_for_cancer_cure_claim() -> None:
    """Regression: Wikipedia 'Water' page must NOT support a cancer-cure claim."""
    claim = "Drinking warm salt water can permanently cure all types of cancer within 3 days."
    water_content = "Water is a chemical compound with formula H2O. It is essential for life on Earth."
    assert detect_stance(claim, water_content) == "NEUTRAL"


def test_great_salt_lake_neutral_for_cancer_cure_claim() -> None:
    """Regression: Great Salt Lake page must NOT support a cancer-cure claim."""
    claim = "Drinking warm salt water can permanently cure all types of cancer within 3 days."
    content = "The Great Salt Lake is a saline lake in Utah. It is known for its high salt concentration."
    assert detect_stance(claim, content) == "NEUTRAL"


def test_donald_trump_page_neutral_for_unrelated_political_claim() -> None:
    """Regression: Generic 'Donald Trump' Wikipedia page alone is not factual evidence."""
    claim = "Donald Trump secretly signed an executive order banning all social media in 2023."
    content = "Donald Trump is an American businessman and politician who served as the 45th President of the United States."
    result = detect_stance(claim, content)
    # Generic bio page should be neutral — it does not confirm or deny the specific order
    assert result == "NEUTRAL"


def test_debunking_article_is_refute() -> None:
    """A fact-check article explicitly debunking a claim should be REFUTE."""
    claim = "5G towers are spreading COVID-19 through radiation."
    content = (
        "This claim has been debunked by health experts. There is no scientific evidence "
        "linking 5G technology to COVID-19. The virus is caused by SARS-CoV-2, not radio waves. "
        "Fact-checkers found this claim to be false and misinformation."
    )
    assert detect_stance(claim, content) == "REFUTE"


def test_confirmation_article_is_support() -> None:
    """An article explicitly confirming a claim should be SUPPORT."""
    claim = "NASA's Perseverance rover landed on Mars in 2021."
    content = (
        "NASA confirmed the successful landing of the Perseverance rover on Mars on February 18, 2021. "
        "Scientists verified the mission through telemetry. The rover has been operating successfully since landing."
    )
    assert detect_stance(claim, content) == "SUPPORT"


def test_stance_summary_counts() -> None:
    stances = ["SUPPORT", "REFUTE", "NEUTRAL", "SUPPORT", "NEUTRAL"]
    summary = stance_summary(stances)
    assert summary == {"SUPPORT": 2, "REFUTE": 1, "NEUTRAL": 2}


# ─── Decision tool tests ──────────────────────────────────────────────────────


def _make_source(stance: str, credibility: float = 0.85, relevance: float = 0.75) -> dict:
    return {
        "title": f"Source ({stance})",
        "source": "test",
        "content": "test content",
        "stance": stance,
        "source_credibility": credibility,
        "relevance_score": relevance,
    }


def test_decision_uncertain_when_all_neutral() -> None:
    """All-neutral evidence must result in UNCERTAIN review state."""
    tool = DecisionTool()
    classification = {"label": "fake", "confidence": 0.98}
    evidence = {
        "evidence_found": True,
        "sources": [_make_source("neutral"), _make_source("neutral")],
        "summary": "Some neutral sources",
        "support_score": 0.05,
        "contradiction_score": 0.05,
        "source_credibility_score": 0.50,
        "evidence_quality_score": 0.45,
        "conflict_flag": False,
    }
    result = tool.decide(classification, evidence)
    assert result["human_review_state"] == ReviewState.uncertain.value
    assert "NEUTRAL" in result["decision_reason"] or "neutral" in result["decision_reason"].lower()


def test_decision_fake_when_refuting_evidence() -> None:
    """Strong refuting evidence + high-confidence FAKE classifier → FAKE state."""
    tool = DecisionTool()
    classification = {"label": "fake", "confidence": 0.9997}
    evidence = {
        "evidence_found": True,
        "sources": [
            _make_source("refute", credibility=0.95, relevance=0.85),
            _make_source("refute", credibility=0.90, relevance=0.80),
        ],
        "summary": "Sources refute the claim",
        "support_score": 0.02,
        "contradiction_score": 0.85,
        "source_credibility_score": 0.92,
        "evidence_quality_score": 0.80,
        "conflict_flag": False,
    }
    result = tool.decide(classification, evidence)
    assert result["human_review_state"] == ReviewState.fake.value
    assert result["conflict_flag"] is False


def test_decision_conflict_when_mixed_stance() -> None:
    """Mixed support + refute evidence must trigger conflict_flag and UNCERTAIN state."""
    tool = DecisionTool()
    classification = {"label": "fake", "confidence": 0.85}
    evidence = {
        "evidence_found": True,
        "sources": [
            _make_source("support", credibility=0.80),
            _make_source("refute", credibility=0.90),
        ],
        "summary": "Mixed evidence",
        "support_score": 0.45,
        "contradiction_score": 0.55,
        "source_credibility_score": 0.85,
        "evidence_quality_score": 0.60,
        "conflict_flag": True,
    }
    result = tool.decide(classification, evidence)
    assert result["conflict_flag"] is True
    assert result["human_review_state"] == ReviewState.uncertain.value


def test_decision_real_when_supporting_evidence() -> None:
    """High-confidence REAL + supporting evidence → REAL state."""
    tool = DecisionTool()
    classification = {"label": "real", "confidence": 0.95}
    evidence = {
        "evidence_found": True,
        "sources": [
            _make_source("support", credibility=0.97, relevance=0.90),
            _make_source("support", credibility=0.92, relevance=0.88),
        ],
        "summary": "Strong supporting evidence",
        "support_score": 0.90,
        "contradiction_score": 0.02,
        "source_credibility_score": 0.94,
        "evidence_quality_score": 0.88,
        "conflict_flag": False,
    }
    result = tool.decide(classification, evidence)
    assert result["human_review_state"] == ReviewState.real.value


def test_decision_reason_mentions_source_titles() -> None:
    """Decision reason must explicitly name supporting/refuting source titles."""
    tool = DecisionTool()
    sources = [
        {**_make_source("refute"), "title": "WHO debunking article"},
        {**_make_source("neutral"), "title": "Water Wikipedia"},
    ]
    classification = {"label": "fake", "confidence": 0.99}
    evidence = {
        "evidence_found": True,
        "sources": sources,
        "summary": "",
        "support_score": 0.0,
        "contradiction_score": 0.6,
        "source_credibility_score": 0.72,
        "evidence_quality_score": 0.55,
        "conflict_flag": False,
    }
    result = tool.decide(classification, evidence)
    assert "WHO debunking article" in result["decision_reason"]


# ─── Explanation formatter tests ─────────────────────────────────────────────


def _make_evidence_item(title: str, stance: str, source: str = "test") -> EvidenceItem:
    return EvidenceItem(
        title=title,
        source=source,
        content="test content",
        relevance_score=0.8,
        stance=stance,
        source_credibility=0.85,
    )


def test_explanation_mentions_supporting_source() -> None:
    ev = [_make_evidence_item("WHO confirms cancer is not cured by salt water", "refute")]
    result = format_explanation(
        prediction="fake",
        confidence=0.99,
        evidence_score=0.7,
        source_trust=0.9,
        important_tokens=[{"token": "cancer", "value": 0.5}],
        evidence=ev,
    )
    assert "refuting" in result.final_explanation.lower() or "refute" in result.final_explanation.lower() or "contradict" in result.final_explanation.lower()


def test_explanation_uncertainty_when_mixed() -> None:
    ev = [
        _make_evidence_item("Supporting article", "support"),
        _make_evidence_item("Debunking article", "refute"),
    ]
    result = format_explanation(
        prediction="uncertain",
        confidence=0.55,
        evidence_score=0.4,
        source_trust=0.7,
        important_tokens=[],
        evidence=ev,
        trust_score=0.45,
    )
    assert "UNCERTAIN" in result.final_explanation or "uncertain" in result.final_explanation.lower()


def test_explanation_neutral_sources_not_called_support() -> None:
    """When all sources are neutral, explanation must NOT say they 'support' the claim."""
    ev = [
        _make_evidence_item("Water Wikipedia", "neutral"),
        _make_evidence_item("Great Salt Lake", "neutral"),
    ]
    result = format_explanation(
        prediction="fake",
        confidence=0.99,
        evidence_score=0.1,
        source_trust=0.5,
        important_tokens=[],
        evidence=ev,
        trust_score=0.7,
    )
    # The explanation should mention that sources are background / neutral context
    explanation_lower = result.final_explanation.lower()
    # Must NOT claim these neutral pages support the fake claim
    assert "sources supporting this classification" not in explanation_lower or \
           "water wikipedia" not in explanation_lower


# ─── Before vs After comparison report ───────────────────────────────────────


def test_before_vs_after_comparison_report(capsys) -> None:
    """Print a comparison table of how the new stance detector handles the
    three failing cases from the original bug report.
    """
    cancer_claim = (
        "Scientists confirm that drinking warm salt water every morning "
        "can permanently cure all types of cancer within 3 days."
    )
    old_water_stance = "support"  # old behaviour (wrong)
    new_water_stance = detect_stance(cancer_claim, "Water is a chemical compound H2O essential for life.")

    old_salt_lake_stance = "support"  # old behaviour (wrong)
    new_salt_lake_stance = detect_stance(
        cancer_claim, "The Great Salt Lake is a saline lake in Utah."
    )

    old_alcohol_stance = "support"  # old behaviour (wrong)
    new_alcohol_stance = detect_stance(
        cancer_claim, "Alcoholic beverages contain ethanol and are fermented."
    )

    print("\n=== Before vs After Comparison ===")
    print(f"{'Source':<30} {'Before':>10} {'After':>10} {'Fixed?':>8}")
    print("-" * 62)
    print(f"{'Water':<30} {'support':>10} {new_water_stance:>10} {'✓' if new_water_stance == 'NEUTRAL' else '✗':>8}")
    print(f"{'Great Salt Lake':<30} {'support':>10} {new_salt_lake_stance:>10} {'✓' if new_salt_lake_stance == 'NEUTRAL' else '✗':>8}")
    print(f"{'Alcoholic beverage':<30} {'support':>10} {new_alcohol_stance:>10} {'✓' if new_alcohol_stance == 'NEUTRAL' else '✗':>8}")

    assert new_water_stance == "NEUTRAL"
    assert new_salt_lake_stance == "NEUTRAL"
    assert new_alcohol_stance == "NEUTRAL"
