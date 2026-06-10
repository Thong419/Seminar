# Defense Questions and Answers

## 1. Why did you choose RoBERTa?
RoBERTa offers strong contextual language understanding and usually outperforms classical bag-of-words models on nuanced text classification tasks.

## 2. Why did you add an agentic AI layer?
The agentic layer lets the system classify, retrieve evidence, analyze evidence, and explain the outcome in a modular and reviewable way.

## 3. Why is retrieval necessary?
Retrieval helps the system back predictions with supporting or contradictory evidence instead of returning a label alone.

## 4. How is explainability implemented?
The system combines important tokens, trust scores, and evidence summaries into a unified explanation.

## 5. How do you monitor model health?
The system tracks prediction logs, data drift, prediction drift, confidence drift, and performance degradation.

## 6. What happens if monitoring indicates drift?
The retraining manager generates a recommendation for drift-triggered or scheduled retraining.

## 7. How do you protect privacy?
PII is detected and masked before prediction and logging.

## 8. What are the main ethical risks?
False positives, false negatives, over-reliance, and misuse against legitimate content or speakers.

## 9. Why not rely on the classifier alone?
A single prediction is not enough for high-stakes misinformation workflows; evidence and explanations are needed.

## 10. How do you handle satire?
Satire is a known failure case because the surface language can resemble misinformation.

## 11. How do you handle ambiguous articles?
The system can return uncertain or likely labels, and the explanation highlights weak evidence.

## 12. What if evidence retrieval fails?
The system still returns a structured prediction and logs the failure for review.

## 13. Why use TF-IDF + Logistic Regression as a baseline?
It is fast, interpretable, and provides a clear performance floor.

## 14. Why compare against a baseline?
The baseline confirms that RoBERTa provides a meaningful improvement.

## 15. How was hyperparameter tuning done?
The transformer configuration was searched and selected based on validation performance.

## 16. Why is trust score needed?
It helps the user distinguish strong, medium, and weak confidence cases.

## 17. How does the UI help the user?
The UI surfaces prediction, confidence, trust score, evidence, and explanation in one place.

## 18. What is the system’s biggest limitation?
It still depends on the quality of external evidence and the limitations of the training data.

## 19. How did you evaluate robustness?
With synthetic perturbations such as spelling noise, uppercase text, repeated punctuation, and shortened articles.

## 20. How did you evaluate fairness?
By analyzing source, topic, and label imbalance.

## 21. Could the model be biased toward certain sources?
Yes. Source bias is a real risk and must be monitored.

## 22. How do you reduce harm from false positives?
Provide explanations, confidence, and human oversight recommendations.

## 23. How do you reduce harm from false negatives?
Use monitoring, retrieval, and conservative escalation for uncertain cases.

## 24. What is the role of monitoring artifacts?
They support debugging, drift review, and periodic model governance.

## 25. What would you improve next?
Better calibration, stronger retrieval quality, and richer fairness audits.

## 26. Why is agentic AI better than a single classifier here?
Because misinformation detection benefits from evidence gathering and explanation, not just prediction.

## 27. How do you ensure deployment readiness?
By providing backend, frontend, monitoring, privacy, and documentation layers with health checks.

## 28. What is the fallback when internet retrieval fails?
The system still returns the classification and explanation with limited evidence.

## 29. How do you keep the system maintainable?
By separating concerns into config, API, agents, monitoring, privacy, robustness, and fairness modules.

## 30. What is the main value of this project?
It demonstrates a production-oriented misinformation detection workflow that is not just accurate but also explainable, monitorable, and responsible.
