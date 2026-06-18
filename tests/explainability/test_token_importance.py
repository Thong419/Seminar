from src.explainability.token_importance import extract_token_importance


def test_extract_token_importance_sorts_descending() -> None:
    tokens = ["news", "fake", "claim", "the"]
    values = [0.1, 0.8, -0.4, 0.2]

    result = extract_token_importance(tokens, values, top_k=3)

    assert result[0]["token"] == "fake"
    assert result[0]["importance"] == 0.8
    assert len(result) == 3


def test_extract_token_importance_truncates_mismatched_lengths() -> None:
    tokens = ["[CLS]", "news", "fake", "claim"]
    values = [0.0, 0.7, -0.2]

    result = extract_token_importance(tokens, values, top_k=10)

    assert all(item["token"] != "[CLS]" for item in result)
    assert len(result) == 2
