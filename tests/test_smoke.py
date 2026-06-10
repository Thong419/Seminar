from src.api.main import app


def test_app_title() -> None:
    assert app.title == "Fake News & Misinformation Detection System"