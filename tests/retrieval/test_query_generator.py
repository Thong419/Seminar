from src.retrieval.query_generator import generate_search_queries


def test_generate_search_queries_creates_multiple_variants() -> None:
    queries = generate_search_queries(
        article_text="Scientists discover miracle cancer cure",
        claim="Scientists discover miracle cancer cure",
        keywords=["scientists", "cancer", "cure"],
        max_queries=4,
    )

    assert len(queries) == 4
    assert any("fact check" in query for query in queries)
    assert any("research study" in query for query in queries)
