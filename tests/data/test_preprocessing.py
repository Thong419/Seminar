from src.config.pipeline import PreprocessingConfig
from src.data.preprocessing import clean_dataframe, preprocess_text


def test_preprocess_text_applies_core_steps() -> None:
    config = PreprocessingConfig()
    result = preprocess_text("<p>Hello   WORLD! Visit https://example.com</p>", config)
    assert result == "hello world visit"


def test_clean_dataframe_removes_duplicates_and_missing_values() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "text": ["One", "One", None, "Two"],
            "label": ["fake", "fake", "real", "real"],
        }
    )
    cleaned, summary = clean_dataframe(frame, "text", "label", PreprocessingConfig())
    assert cleaned.shape[0] == 2
    assert summary.rows_before == 4
    assert summary.rows_after == 2