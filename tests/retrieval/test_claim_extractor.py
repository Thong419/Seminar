from src.retrieval.claim_extractor import extract_claim


def test_extract_claim_returns_key_sentence_and_keywords() -> None:
    article = (
        "Scientists discover miracle cancer cure in early research. "
        "The study suggests a potential treatment breakthrough."
    )

    result = extract_claim(article)

    assert "cancer cure" in result["claim"].lower()
    assert isinstance(result["keywords"], list)
    assert result["keywords"]
