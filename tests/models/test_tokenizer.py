from src.config.pipeline import ModelConfig
from src.models.transformer_model import build_tokenizer


def test_tokenizer_builds_for_configured_model() -> None:
    import src.models.transformer_model as transformer_model

    class DummyTokenizer:
        pass

    original = transformer_model.AutoTokenizer.from_pretrained
    transformer_model.AutoTokenizer.from_pretrained = lambda *args, **kwargs: DummyTokenizer()
    try:
        tokenizer = build_tokenizer(ModelConfig(name="roberta-base", tokenizer_name="roberta-base"))
        assert isinstance(tokenizer, DummyTokenizer)
    finally:
        transformer_model.AutoTokenizer.from_pretrained = original
