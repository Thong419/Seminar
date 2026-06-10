from __future__ import annotations

from src.privacy.pii_detector import detect_pii
from src.privacy.pii_masker import mask_pii


def test_detect_pii_finds_multiple_entity_types() -> None:
    text = "Email john@example.com, call +1 (555) 123-4567, visit https://example.com, 12 Main Street."

    entities = detect_pii(text)

    assert {entity.entity_type for entity in entities} >= {"email", "phone", "url", "address"}


def test_mask_pii_replaces_sensitive_entities() -> None:
    text = "Contact john@example.com or visit https://example.com at 12 Main Street."

    masked = mask_pii(text)

    assert "john@example.com" not in masked.text
    assert "https://example.com" not in masked.text
    assert "[EMAIL]" in masked.text
    assert "[URL]" in masked.text
    assert "[ADDRESS]" in masked.text
