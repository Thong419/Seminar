# Continual Learning Strategy

## Data Collection Strategy

- Log every prediction and analysis request with timestamp, prediction, confidence, trust score, and article length.
- Retain batches of recently observed articles and their monitoring summaries for drift analysis.
- Keep separate reference windows for training distributions and production windows for live traffic.

## Retraining Workflow

1. Collect and summarize monitoring outputs on a fixed cadence.
2. Compare production distributions against reference training baselines.
3. Trigger retraining when drift, health degradation, or schedule-based policies are met.
4. Retrain the RoBERTa classifier, validate against the held-out set, and register the best model.
5. Promote the candidate only after it clears accuracy, F1, and confidence stability checks.

## Monitoring Metrics

- Data drift: article length shift, vocabulary divergence, token frequency variation.
- Prediction drift: fake/real ratio, decision distribution, trust score distribution.
- Confidence drift: mean confidence, confidence histogram, confidence trend slope.
- Performance degradation: drops in accuracy, precision, recall, and F1 relative to baseline.

## Drift Mitigation Strategy

- Use conservative thresholds to separate GREEN, YELLOW, and RED health states.
- Prefer drift-triggered retraining when confidence or prediction distributions move abruptly.
- Use monthly retraining as a backstop even when drift is not extreme.
- Review the latest monitoring summaries before model promotion and rollback on regressions.
