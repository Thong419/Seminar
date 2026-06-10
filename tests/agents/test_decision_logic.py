from src.agents.decision.decision_agent import DecisionAgent
from src.agents.state import AgentConfig, Verdict


def test_decision_agent_returns_confirmed_fake_on_strong_fake_signals() -> None:
    agent = DecisionAgent()
    result = agent.decide(
        predicted_label="fake",
        confidence=0.96,
        evidence_score=0.85,
        config=AgentConfig(confidence_threshold=0.90),
    )

    assert result.final_decision == Verdict.confirmed_fake


def test_decision_agent_returns_uncertain_on_weak_signals() -> None:
    agent = DecisionAgent()
    result = agent.decide(
        predicted_label="real",
        confidence=0.42,
        evidence_score=0.10,
        config=AgentConfig(confidence_threshold=0.90),
    )

    assert result.final_decision == Verdict.uncertain
