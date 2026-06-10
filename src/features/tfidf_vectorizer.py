"""TF-IDF vectorizer factory for the baseline classifier."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.pipeline import TFIDFConfig


def build_tfidf_vectorizer(config: TFIDFConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
    )