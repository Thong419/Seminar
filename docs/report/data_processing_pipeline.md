# Data Processing Pipeline

The data pipeline prepares raw text for both classical and transformer-based modeling. The processing stages are:

```mermaid
flowchart TD
    R[Raw Data] --> V[Validation]
    V --> P[Preprocessing]
    P --> S[Train/Validation/Test Split]
    S --> B[Baseline Features]
    S --> T[Transformer Tokenization]
```

## Reporting Points

- Explain text normalization steps.
- Describe how label validation and split reproducibility are handled.
- Mention feature extraction for TF-IDF and tokenization for RoBERTa.
