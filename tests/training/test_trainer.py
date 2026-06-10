from pathlib import Path

import pandas as pd

from src.config.pipeline import DatasetConfig, EvaluationConfig, PreprocessingConfig, TFIDFConfig, LogisticRegressionConfig, TrainingConfig
from src.training.trainer import BaselineTrainer


def test_baseline_trainer_runs_end_to_end(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "text": [
                "Fake story about politics",
                "Real report from an established source",
                "Another fake rumor",
                "Verified real article",
            ],
            "label": ["fake", "real", "fake", "real"],
        }
    )

    dataset_config = DatasetConfig(
        source_path=Path("unused.csv"),
        preprocessing=PreprocessingConfig(),
    )
    training_config = TrainingConfig(
        model_output_dir=tmp_path / "models",
        mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
        tfidf=TFIDFConfig(max_features=100, min_df=1),
        logistic_regression=LogisticRegressionConfig(max_iter=200, class_weight=None),
    )
    evaluation_config = EvaluationConfig(output_dir=tmp_path / "evaluation")

    trainer = BaselineTrainer(
        dataset_config=dataset_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        enable_mlflow=False,
    )
    trainer.fit(frame)
    metrics = trainer.evaluate(frame)

    assert metrics.accuracy >= 0.5
    assert (tmp_path / "models" / "baseline_model.joblib").parent.exists()