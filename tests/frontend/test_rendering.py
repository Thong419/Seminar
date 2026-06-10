from __future__ import annotations

from frontend.components.prediction_card import prediction_badge_color, prediction_badge_label
from frontend.trust_score_view import classify_trust_score, trust_progress


def test_prediction_badge_helpers() -> None:
    assert prediction_badge_label("fake") == "FAKE"
    assert prediction_badge_label("likely_real") == "LIKELY REAL"
    assert prediction_badge_color("real") == "#0f766e"


def test_trust_score_helpers() -> None:
    assert classify_trust_score(91) == "High Trust"
    assert classify_trust_score(70) == "Medium Trust"
    assert classify_trust_score(12) == "Low Trust"
    assert trust_progress(88) == 0.88
