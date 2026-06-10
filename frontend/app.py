"""Streamlit application for the fake news detection backend."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.api_client import APIClient, APIError, ModelInfo
from frontend.components.prediction_card import prediction_badge_color, prediction_badge_label
from frontend.config import FrontendConfig, get_frontend_config, load_json, load_yaml
from frontend.evidence_view import render_evidence_cards
from frontend.explanation_view import build_explanation_lines
from frontend.trust_score_view import classify_trust_score, trust_progress


EXAMPLE_ARTICLE = (
    "A viral post claims a miracle cure can reverse chronic illness overnight, "
    "but no reputable medical source has verified the claim."
)


def load_model_info(config: FrontendConfig) -> ModelInfo:
    model_yaml = load_yaml(config.model_config_path)
    dataset_yaml = load_yaml(config.dataset_config_path)
    metrics = load_json(config.metrics_artifact_path)

    model_section = model_yaml.get("model", {}) if isinstance(model_yaml, dict) else {}
    dataset_name = str(dataset_yaml.get("name", "unknown")) if isinstance(dataset_yaml, dict) else "unknown"

    def _metric(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    return ModelInfo(
        model_name=str(model_section.get("name", "roberta-base")) if isinstance(model_section, dict) else "roberta-base",
        dataset=dataset_name,
        accuracy=_metric("accuracy"),
        precision=_metric("precision"),
        recall=_metric("recall"),
        f1=_metric("f1"),
    )


def _metric_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _render_badge(prediction: str) -> None:
    label = prediction_badge_label(prediction)
    color = prediction_badge_color(prediction)
    st.markdown(
        f"<div style='display:inline-block;padding:0.45rem 0.85rem;border-radius:999px;background:{color};color:white;font-weight:700;letter-spacing:0.08em;'>{label}</div>",
        unsafe_allow_html=True,
    )


def _render_architecture_diagram() -> None:
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:0.6rem;align-items:stretch;max-width:520px;">
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">Article</div>
            <div style="text-align:center;font-size:1.25rem;">↓</div>
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">RoBERTa</div>
            <div style="text-align:center;font-size:1.25rem;">↓</div>
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">Confidence Check</div>
            <div style="text-align:center;font-size:1.25rem;">↓</div>
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">Evidence Retrieval</div>
            <div style="text-align:center;font-size:1.25rem;">↓</div>
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">Decision</div>
            <div style="text-align:center;font-size:1.25rem;">↓</div>
            <div style="padding:0.9rem 1rem;border:1px solid #d6d3d1;border-radius:14px;background:#fff;">Explanation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    config = get_frontend_config()
    api_client = APIClient(config)

    st.set_page_config(page_title=config.app_title, page_icon="📰", layout="wide")
    st.markdown(
        """
        <style>
            .main { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
            .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            .panel {
                background: rgba(255,255,255,0.86);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.05);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(config.app_title)
    st.caption("A production-style frontend for prediction, evidence, and explanation.")

    analyze_tab, model_tab, architecture_tab = st.tabs([
        "Analyze Article",
        "Model Information",
        "System Architecture",
    ])

    with analyze_tab:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.session_state.setdefault("article_text", "")
        article_text = st.text_area(
            "Article text",
            height=260,
            placeholder="Paste a news article or claim here...",
            key="article_text",
        )
        controls_left, controls_right = st.columns([1, 1])
        with controls_left:
            if st.button("Load example article", use_container_width=True):
                st.session_state["article_text"] = EXAMPLE_ARTICLE
                st.rerun()
        with controls_right:
            analyze_clicked = st.button("Analyze article", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        text_to_analyze = article_text.strip()

        if analyze_clicked:
            if not text_to_analyze:
                st.error("Please enter article text before analyzing.")
            else:
                try:
                    with st.spinner("Calling backend analysis workflow..."):
                        result = api_client.analyze(text_to_analyze)
                    st.session_state["analysis_result"] = result
                except APIError as exc:
                    if exc.code == "timeout":
                        st.error("The backend took too long to respond. Try again or increase the timeout.")
                    elif exc.code == "backend_unavailable":
                        st.error("The FastAPI backend is unavailable. Start the API service and retry.")
                    elif exc.code == "validation_error":
                        st.error("The submitted article text was rejected by the backend validation layer.")
                    else:
                        st.error(f"Analysis failed: {exc.message}")

        result = st.session_state.get("analysis_result")
        if result:
            st.subheader("Results")
            left, middle, right, decision = st.columns(4)
            with left:
                st.metric("Prediction", str(result.get("prediction", "uncertain")).upper())
            with middle:
                confidence = float(result.get("confidence", 0.0))
                st.metric("Confidence", f"{confidence:.2%}")
            with right:
                trust_score = int(result.get("trust_score", 0))
                st.metric("Trust Score", f"{trust_score}/100")
            with decision:
                st.metric("Final Decision", str(result.get("explanation_details", {}).get("final_decision", result.get("prediction", "uncertain")).replace("_", " ")).upper())

            st.markdown("### Prediction Badge")
            _render_badge(str(result.get("prediction", "uncertain")))

            trust_score = int(result.get("trust_score", 0))
            st.markdown("### Trust Score")
            st.progress(trust_progress(trust_score))
            st.write(f"**{trust_score}/100** · {classify_trust_score(trust_score)}")

            st.markdown("### Important Tokens")
            token_rows = result.get("important_tokens", []) or []
            st.dataframe(token_rows, use_container_width=True, hide_index=True)

            st.markdown("### Evidence")
            for item in render_evidence_cards(result.get("evidence", []) or []):
                with st.container(border=True):
                    st.markdown(f"**{item.get('title', 'Untitled evidence')}**")
                    st.caption(f"Source: {item.get('source', 'Unknown')} · Trust: {item.get('trust_score', 'N/A')} · Relevance: {item.get('relevance_score', 'N/A')}")
                    url = item.get("url")
                    if url:
                        st.link_button("Open source", str(url))
                    st.write(str(item.get("content", "")))

            st.markdown("### Explanation")
            evidence_summary, final_explanation = build_explanation_lines(
                str(result.get("evidence_summary", "")),
                str(result.get("final_explanation", "")),
            )
            st.write(f"**Evidence summary:** {evidence_summary}")
            st.write(f"**Final explanation:** {final_explanation}")

    with model_tab:
        model_info = load_model_info(config)
        st.subheader("Model Information")
        a, b, c, d, e = st.columns(5)
        a.metric("Model Name", model_info.model_name)
        b.metric("Dataset", model_info.dataset)
        c.metric("Accuracy", _metric_value(model_info.accuracy))
        d.metric("Precision", _metric_value(model_info.precision))
        e.metric("Recall", _metric_value(model_info.recall))
        st.metric("F1 Score", _metric_value(model_info.f1))
        st.info("Metrics are loaded from artifacts when available. If no evaluation artifact exists yet, values are shown as N/A.")

    with architecture_tab:
        st.subheader("System Architecture")
        _render_architecture_diagram()
        st.caption("Article → RoBERTa → Confidence Check → Evidence Retrieval → Decision → Explanation")


if __name__ == "__main__":
    main()
