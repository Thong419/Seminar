"""YAML-backed configuration models for the data and baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")

    return content


def _as_path(value: Any, default: str) -> Path:
    if value is None:
        return Path(default)
    return Path(str(value))


def _tuple_from_sequence(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("Expected a two-item sequence.")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_whitespace: bool = True
    remove_special_characters: bool = True
    special_characters_pattern: str = r"[^a-zA-Z0-9\s]"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str = "fakenewsnet"
    version: str = "unknown"
    source_type: str = "local"
    source_path: Path = Path("data/raw/fakenewsnet.csv")
    file_format: str = "csv"
    text_column: str = "text"
    label_column: str = "label"
    allowed_labels: tuple[str, str] = ("fake", "real")
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True, slots=True)
class TFIDFConfig:
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfig:
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "liblinear"
    class_weight: str | None = "balanced"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    experiment_name: str = "fake_news_baseline"
    model_output_dir: Path = Path("models/baseline")
    mlflow_tracking_uri: str = "http://localhost:5000"
    dataset_version: str = "unknown"
    random_state: int = 42
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    logistic_regression: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    output_dir: Path = Path("artifacts/evaluation")
    save_confusion_matrix: bool = True
    save_metrics_json: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "roberta-base"
    max_length: int = 512
    num_labels: int = 2
    model_output_dir: Path = Path("models/roberta")
    label_names: tuple[str, str] = ("fake", "real")
    tokenizer_name: str = "roberta-base"
    use_fast_tokenizer: bool = True


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    gradient_accumulation_steps: int = 1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    truncation: bool = True
    padding: str = "dynamic"


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = _load_yaml(path)
    preprocessing = raw.get("preprocessing", {})
    return DatasetConfig(
        name=str(raw.get("name", DatasetConfig.name)),
        version=str(raw.get("version", DatasetConfig.version)),
        source_type=str(raw.get("source_type", DatasetConfig.source_type)),
        source_path=_as_path(raw.get("source_path"), str(DatasetConfig.source_path)),
        file_format=str(raw.get("file_format", DatasetConfig.file_format)),
        text_column=str(raw.get("text_column", DatasetConfig.text_column)),
        label_column=str(raw.get("label_column", DatasetConfig.label_column)),
        allowed_labels=tuple(raw.get("allowed_labels", list(DatasetConfig.allowed_labels))),
        preprocessing=PreprocessingConfig(
            lowercase=bool(preprocessing.get("lowercase", True)),
            remove_urls=bool(preprocessing.get("remove_urls", True)),
            remove_html=bool(preprocessing.get("remove_html", True)),
            remove_whitespace=bool(preprocessing.get("remove_whitespace", True)),
            remove_special_characters=bool(
                preprocessing.get("remove_special_characters", True)
            ),
            special_characters_pattern=str(
                preprocessing.get(
                    "special_characters_pattern", PreprocessingConfig.special_characters_pattern
                )
            ),
        ),
    )


def load_split_config(path: Path) -> SplitConfig:
    raw = _load_yaml(path)
    return SplitConfig(
        train_size=float(raw.get("train_size", SplitConfig.train_size)),
        validation_size=float(raw.get("validation_size", SplitConfig.validation_size)),
        test_size=float(raw.get("test_size", SplitConfig.test_size)),
        random_state=int(raw.get("random_state", SplitConfig.random_state)),
        stratify=bool(raw.get("stratify", SplitConfig.stratify)),
    )


def load_split_config_from_dataset(path: Path) -> SplitConfig:
    """Load split configuration from a dataset YAML file that nests split settings."""

    raw = _load_yaml(path)
    split = raw.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("Dataset split configuration must be a mapping.")

    return SplitConfig(
        train_size=float(split.get("train_size", SplitConfig.train_size)),
        validation_size=float(split.get("validation_size", SplitConfig.validation_size)),
        test_size=float(split.get("test_size", SplitConfig.test_size)),
        random_state=int(split.get("random_state", SplitConfig.random_state)),
        stratify=bool(split.get("stratify", SplitConfig.stratify)),
    )


def load_tfidf_config(raw: Mapping[str, Any] | None) -> TFIDFConfig:
    raw = raw or {}
    return TFIDFConfig(
        max_features=int(raw.get("max_features", TFIDFConfig.max_features)),
        ngram_range=_tuple_from_sequence(raw.get("ngram_range"), TFIDFConfig.ngram_range),
        min_df=int(raw.get("min_df", TFIDFConfig.min_df)),
        max_df=float(raw.get("max_df", TFIDFConfig.max_df)),
        sublinear_tf=bool(raw.get("sublinear_tf", TFIDFConfig.sublinear_tf)),
    )


def load_logistic_regression_config(raw: Mapping[str, Any] | None) -> LogisticRegressionConfig:
    raw = raw or {}
    class_weight = raw.get("class_weight", LogisticRegressionConfig.class_weight)
    return LogisticRegressionConfig(
        C=float(raw.get("C", LogisticRegressionConfig.C)),
        max_iter=int(raw.get("max_iter", LogisticRegressionConfig.max_iter)),
        solver=str(raw.get("solver", LogisticRegressionConfig.solver)),
        class_weight=None if class_weight in (None, "null", "None") else str(class_weight),
    )


def load_training_config(path: Path) -> TrainingConfig:
    raw = _load_yaml(path)
    return TrainingConfig(
        experiment_name=str(raw.get("experiment_name", TrainingConfig.experiment_name)),
        model_output_dir=_as_path(
            raw.get("model_output_dir"), str(TrainingConfig.model_output_dir)
        ),
        mlflow_tracking_uri=str(
            raw.get("mlflow_tracking_uri", TrainingConfig.mlflow_tracking_uri)
        ),
        dataset_version=str(raw.get("dataset_version", TrainingConfig.dataset_version)),
        random_state=int(raw.get("random_state", TrainingConfig.random_state)),
        tfidf=load_tfidf_config(raw.get("tfidf")),
        logistic_regression=load_logistic_regression_config(
            raw.get("logistic_regression")
        ),
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = _load_yaml(path)
    return EvaluationConfig(
        output_dir=_as_path(raw.get("output_dir"), str(EvaluationConfig.output_dir)),
        save_confusion_matrix=bool(
            raw.get("save_confusion_matrix", EvaluationConfig.save_confusion_matrix)
        ),
        save_metrics_json=bool(raw.get("save_metrics_json", EvaluationConfig.save_metrics_json)),
    )


def load_model_config(path: Path) -> tuple[ModelConfig, TransformerTrainingConfig, TokenizationConfig]:
    raw = _load_yaml(path)
    model = raw.get("model", {})
    training = raw.get("training", {})
    tokenization = raw.get("tokenization", {})

    if not isinstance(model, dict) or not isinstance(training, dict) or not isinstance(tokenization, dict):
        raise ValueError("Model configuration sections must be mappings.")

    label_names = tuple(model.get("label_names", list(ModelConfig.label_names)))
    if len(label_names) != 2:
        raise ValueError("Exactly two label names are required for binary classification.")
    label_names = (str(label_names[0]), str(label_names[1]))

    return (
        ModelConfig(
            name=str(model.get("name", ModelConfig.name)),
            max_length=int(model.get("max_length", ModelConfig.max_length)),
            num_labels=int(model.get("num_labels", ModelConfig.num_labels)),
            model_output_dir=_as_path(model.get("model_output_dir"), str(ModelConfig.model_output_dir)),
            label_names=label_names,
            tokenizer_name=str(model.get("tokenizer_name", ModelConfig.tokenizer_name)),
            use_fast_tokenizer=bool(model.get("use_fast_tokenizer", ModelConfig.use_fast_tokenizer)),
        ),
        TransformerTrainingConfig(
            batch_size=int(training.get("batch_size", TransformerTrainingConfig.batch_size)),
            learning_rate=float(training.get("learning_rate", TransformerTrainingConfig.learning_rate)),
            epochs=int(training.get("epochs", TransformerTrainingConfig.epochs)),
            weight_decay=float(training.get("weight_decay", TransformerTrainingConfig.weight_decay)),
            warmup_ratio=float(training.get("warmup_ratio", TransformerTrainingConfig.warmup_ratio)),
            evaluation_strategy=str(training.get("evaluation_strategy", TransformerTrainingConfig.evaluation_strategy)),
            save_strategy=str(training.get("save_strategy", TransformerTrainingConfig.save_strategy)),
            load_best_model_at_end=bool(training.get("load_best_model_at_end", TransformerTrainingConfig.load_best_model_at_end)),
            metric_for_best_model=str(training.get("metric_for_best_model", TransformerTrainingConfig.metric_for_best_model)),
            greater_is_better=bool(training.get("greater_is_better", TransformerTrainingConfig.greater_is_better)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", TransformerTrainingConfig.gradient_accumulation_steps)),
            seed=int(training.get("seed", TransformerTrainingConfig.seed)),
        ),
        TokenizationConfig(
            truncation=bool(tokenization.get("truncation", TokenizationConfig.truncation)),
            padding=str(tokenization.get("padding", TokenizationConfig.padding)),
        ),
    )


def load_comparison_output_path() -> Path:
    return Path("artifacts/evaluation/model_comparison.csv")


def as_serializable_dict(config: object) -> dict[str, Any]:
    """Convert a dataclass config into a JSON/YAML-friendly dictionary."""

    return asdict(config)