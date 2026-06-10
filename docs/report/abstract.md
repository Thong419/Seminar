# Abstract

This project presents an end-to-end fake news and misinformation detection system that combines transformer-based classification, evidence retrieval, explainability, monitoring, and responsible AI safeguards. The system is designed to support practical decision-making by returning not only a prediction but also confidence, trust score, important tokens, supporting evidence, and a final explanation.

The implementation includes a TF-IDF plus Logistic Regression baseline, a RoBERTa-based transformer model, agentic orchestration for evidence gathering and explanation, and operational layers for drift detection, privacy masking, robustness evaluation, and fairness analysis. The final system is exposed through a FastAPI backend and a Streamlit interface for interactive review.
