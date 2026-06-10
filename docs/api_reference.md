# API Reference

## GET /health

Returns the service health state.

### Response

```json
{ "status": "healthy" }
```

## GET /version

Returns the deployed model name and version.

### Response

```json
{ "model_name": "roberta-base", "model_version": "1.0.0" }
```

## POST /predict

Predicts fake or real using the predictor interface only.

### Request

```json
{ "text": "article text" }
```

### Response

```json
{ "prediction": "fake", "confidence": 0.91 }
```

## POST /analyze

Runs classification, retrieval, evidence analysis, decision making, and explainability.

### Request

```json
{ "text": "article text" }
```

### Response

```json
{
  "prediction": "fake",
  "confidence": 0.91,
  "trust_score": 88,
  "important_tokens": [{ "token": "fake", "importance": 0.81 }],
  "evidence": [],
  "evidence_summary": "Retrieved 1 evidence item from 1 sources.",
  "final_explanation": "The article was classified as fake with confidence 0.91."
}
```

## Error Handling

- Validation errors return `validation_error` payloads.
- Missing model errors return `missing_model` payloads.
- Retrieval failures return `retrieval_failure` payloads.
- Unexpected failures return `unexpected_error` payloads.
