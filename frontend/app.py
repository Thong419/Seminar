"""Streamlit application for the fake news detection backend."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.api_client import APIClient, APIError, ModelInfo
from frontend.agent_result_view import render_agent_result
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


def _clip_text(text: str, max_chars: int = 320) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


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


def _render_kv_list(title: str, values: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.caption("None")
        return
    for value in values:
        st.markdown(f"- {value}")


def _render_trace_timeline(trace: dict[str, object] | None, steps: list[dict[str, object]], total_execution_time: str) -> None:
    st.markdown("### Trace Section")
    st.caption("Agent Execution Timeline")
    if trace is None:
        st.info("No trace available.")
        return

    if not steps:
        st.info("Trace captured but no tool steps were recorded.")
        st.write(f"**Total execution time:** {total_execution_time}")
        return

    for step in steps:
        tool_name = str(step.get("tool_name", "Tool"))
        duration_ms = float(step.get("duration_ms", 0.0))
        summary = str(step.get("summary", ""))
        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"**{int(step.get('step', 0))}. {tool_name}**")
                st.write(summary)
            with right:
                st.metric("Duration", f"{duration_ms:.0f} ms")
    st.write(f"**Total execution time:** {total_execution_time}")


def _render_evidence_cards(items: list[dict[str, object]]) -> None:
    st.markdown("### Evidence Cards")
    for item in items:
        title = str(item.get("title", "Untitled evidence"))
        provider = str(item.get("source", item.get("provider", "Unknown")))
        url = str(item.get("url", "")).strip()
        trust_value = item.get("source_credibility", item.get("trust_score", "N/A"))
        relevance_value = item.get("relevance_score", "N/A")
        content = str(item.get("content", ""))
        preview = _clip_text(content, max_chars=340)

        with st.container(border=True):
            st.markdown(f"**{title}**")
            meta_left, meta_right = st.columns(2)
            with meta_left:
                st.write(f"**Provider:** {provider}")
                st.write(f"**URL:** {url if url else 'N/A'}")
            with meta_right:
                if isinstance(trust_value, (int, float)):
                    st.write(f"**Trust Score:** {float(trust_value):.2%}")
                else:
                    st.write(f"**Trust Score:** {trust_value}")
                if isinstance(relevance_value, (int, float)):
                    st.write(f"**Relevance:** {float(relevance_value):.2%}")
                else:
                    st.write(f"**Relevance:** {relevance_value}")

            st.write(f"**Preview:** {preview}")

            if content and len(" ".join(content.split())) > len(preview):
                with st.expander("See More"):
                    st.write(content)
            elif content:
                with st.expander("See More"):
                    st.write(content)


def _render_rejected_evidence(items: list[dict[str, object]]) -> None:
    st.markdown("### Rejected Evidence")
    if not items:
        st.caption("No rejected evidence records.")
        return

    for item in items[:12]:
        title = str(item.get("title", "Untitled evidence"))
        source = str(item.get("source", "Unknown"))
        url = str(item.get("url", "")).strip()
        reason = str(item.get("rejection_reason", "unknown"))
        semantic = item.get("semantic_score", item.get("adjusted_score", 0.0))
        coverage = item.get("coverage_score", item.get("claim_coverage_score", 0.0))
        credibility = item.get("source_credibility", 0.0)

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"Source: {source}")
            if url:
                st.write(url)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.write("**Reason**")
                st.write(reason)
            with c2:
                st.write("**Semantic**")
                st.write(f"{float(semantic):.3f}" if isinstance(semantic, (int, float)) else str(semantic))
            with c3:
                st.write("**Coverage**")
                st.write(f"{float(coverage):.3f}" if isinstance(coverage, (int, float)) else str(coverage))
            with c4:
                st.write("**Credibility**")
                st.write(f"{float(credibility):.2f}" if isinstance(credibility, (int, float)) else str(credibility))

            query_used = str(item.get("query_used", "")).strip()
            if query_used:
                st.caption(f"Query: {query_used}")


def main() -> None:
    config = get_frontend_config()
    api_client = APIClient(config)

    st.set_page_config(page_title=config.app_title, page_icon="📰", layout="wide")
    screenshot_mode = st.toggle("Defense Screenshot Mode", value=False, help="Compact, clean layout for report and slide screenshots.")

    top_padding = "1.0rem" if screenshot_mode else "2.0rem"
    bottom_padding = "1.0rem" if screenshot_mode else "2.0rem"
    section_gap = "0.45rem" if screenshot_mode else "0.9rem"

    st.markdown(
        f"""
        <style>
            html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {{
                font-family: "Segoe UI", "Noto Sans", "Inter", sans-serif;
            }}
            .main {{
                background: linear-gradient(180deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
            }}
            .block-container {{
                padding-top: {top_padding};
                padding-bottom: {bottom_padding};
            }}
            .panel {{
                background: var(--secondary-background-color);
                color: var(--text-color);
                border: 1px solid rgba(128, 128, 128, 0.35);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: {section_gap};
            }}
            [data-testid="stMetricValue"] {{
                color: var(--text-color);
            }}
            [data-testid="stMetricLabel"] {{
                color: var(--text-color);
            }}
            .stCaption {{
                color: var(--text-color);
                opacity: 0.85;
            }}
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
                    with st.spinner("Calling backend agent workflow..."):
                        result = api_client.agent(text_to_analyze)
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
            agent_summary = render_agent_result(result)
            st.subheader("Agent Result Card")
            result_left, result_middle, result_right = st.columns(3)
            with result_left:
                st.metric("Prediction", agent_summary["Prediction"])
                st.metric("Confidence", agent_summary["Confidence"])
            with result_middle:
                st.metric("Trust Score", agent_summary["Trust Score"])
                st.metric("Human Review State", agent_summary["Human Review State"])
            with result_right:
                st.metric("Conflict Flag", agent_summary["Conflict Flag"])
                st.metric("Decision", str(result.get("human_review_state", result.get("prediction", "uncertain"))).replace("_", " ").upper())

            st.markdown("### Decision Reason")
            st.write(agent_summary["Decision Reason"])

            trust_score_100 = int(round(float(result.get("trust_score", 0.0)) * 100.0))
            st.progress(trust_progress(trust_score_100))
            st.caption(f"Trust interpretation: {classify_trust_score(trust_score_100)}")

            st.markdown("### Evidence Section")
            evidence_left, evidence_right = st.columns(2)
            with evidence_left:
                _render_kv_list("Source Providers", agent_summary["Source Providers"])
                st.markdown("**Support Score**")
                st.write(f"{float(result.get('support_score', 0.0)):.2%}")
                st.markdown("**Contradiction Score**")
                st.write(f"{float(result.get('contradiction_score', 0.0)):.2%}")
                st.markdown("**Evidence Quality**")
                st.write(f"{float(result.get('evidence_quality_score', 0.0)):.2%}")
            with evidence_right:
                _render_kv_list("Top Evidence Titles", agent_summary["Evidence Titles"])
                st.markdown("**Evidence Summary**")
                st.write(agent_summary["Evidence Summary"])

            _render_evidence_cards(render_evidence_cards(result.get("sources", []) or []))
            _render_rejected_evidence(list(result.get("rejected_evidence", []) or []))

            _render_trace_timeline(result.get("trace") if isinstance(result.get("trace"), dict) else None, agent_summary["Trace Steps"], agent_summary["Total Execution Time"])

            st.markdown("### Explanation Section")
            evidence_summary, final_explanation = build_explanation_lines(
                str(result.get("evidence_summary", "")),
                str(result.get("explanation", result.get("final_explanation", ""))),
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
