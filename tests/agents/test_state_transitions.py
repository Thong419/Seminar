from src.agents.nodes import confidence_check
from src.agents.state import AgentConfig, AgentState


def test_confidence_check_skips_retrieval_above_threshold() -> None:
    state: AgentState = {"article_text": "x", "confidence": 0.95}
    nodes = type("Nodes", (), {"config": AgentConfig(confidence_threshold=0.90)})()

    assert confidence_check(state, nodes) == "make_decision"


def test_confidence_check_requests_retrieval_below_threshold() -> None:
    state: AgentState = {"article_text": "x", "confidence": 0.55}
    nodes = type("Nodes", (), {"config": AgentConfig(confidence_threshold=0.90)})()

    assert confidence_check(state, nodes) == "retrieve_evidence"
