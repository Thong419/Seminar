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
    source_path: Path = field(default_factory=lambda: Path("data/raw/fakenewsnet.csv"))
    file_format: str = "csv"
    text_column: str = "text"
    label_column: str = "label"
    allowed_labels: tuple[str, ...] = field(default_factory=lambda: ("fake", "real"))
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
    defaults = DatasetConfig()
    preprocessing_defaults = PreprocessingConfig()

    preprocessing = raw.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        raise ValueError("Dataset preprocessing configuration must be a mapping.")

    raw_allowed_labels = raw.get("allowed_labels")
    if raw_allowed_labels is None:
        allowed_labels = defaults.allowed_labels
    elif isinstance(raw_allowed_labels, (list, tuple)):
        allowed_labels = tuple(str(label) for label in raw_allowed_labels)
        if not allowed_labels:
            raise ValueError("allowed_labels must contain at least one label.")
    else:
        raise ValueError("allowed_labels must be a sequence of labels.")

    return DatasetConfig(
        name=str(raw.get("name", defaults.name)),
        version=str(raw.get("version", defaults.version)),
        source_type=str(raw.get("source_type", defaults.source_type)),
        source_path=_as_path(raw.get("source_path"), str(defaults.source_path)),
        file_format=str(raw.get("file_format", defaults.file_format)),
        text_column=str(raw.get("text_column", defaults.text_column)),
        label_column=str(raw.get("label_column", defaults.label_column)),
        allowed_labels=allowed_labels,
        preprocessing=PreprocessingConfig(
            lowercase=bool(preprocessing.get("lowercase", preprocessing_defaults.lowercase)),
            remove_urls=bool(preprocessing.get("remove_urls", preprocessing_defaults.remove_urls)),
            remove_html=bool(preprocessing.get("remove_html", preprocessing_defaults.remove_html)),
            remove_whitespace=bool(preprocessing.get("remove_whitespace", preprocessing_defaults.remove_whitespace)),
            remove_special_characters=bool(
                preprocessing.get(
                    "remove_special_characters",
                    preprocessing_defaults.remove_special_characters,
                )
            ),
            special_characters_pattern=str(
                preprocessing.get(
                    "special_characters_pattern",
                    preprocessing_defaults.special_characters_pattern,
                )
            ),
        ),
    )


def load_split_config(path: Path) -> SplitConfig:
    raw = _load_yaml(path)
    defaults = SplitConfig()
    return SplitConfig(
        train_size=float(raw.get("train_size", defaults.train_size)),
        validation_size=float(raw.get("validation_size", defaults.validation_size)),
        test_size=float(raw.get("test_size", defaults.test_size)),
        random_state=int(raw.get("random_state", defaults.random_state)),
        stratify=bool(raw.get("stratify", defaults.stratify)),
    )


def load_split_config_from_dataset(path: Path) -> SplitConfig:
    """Load split configuration from a dataset YAML file that nests split settings."""

    raw = _load_yaml(path)
    defaults = SplitConfig()
    split = raw.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("Dataset split configuration must be a mapping.")

    return SplitConfig(
        train_size=float(split.get("train_size", defaults.train_size)),
        validation_size=float(split.get("validation_size", defaults.validation_size)),
        test_size=float(split.get("test_size", defaults.test_size)),
        random_state=int(split.get("random_state", defaults.random_state)),
        stratify=bool(split.get("stratify", defaults.stratify)),
    )


def load_tfidf_config(raw: Mapping[str, Any] | None) -> TFIDFConfig:
    raw = raw or {}
    defaults = TFIDFConfig()
    return TFIDFConfig(
        max_features=int(raw.get("max_features", defaults.max_features)),
        ngram_range=_tuple_from_sequence(raw.get("ngram_range"), defaults.ngram_range),
        min_df=int(raw.get("min_df", defaults.min_df)),
        max_df=float(raw.get("max_df", defaults.max_df)),
        sublinear_tf=bool(raw.get("sublinear_tf", defaults.sublinear_tf)),
    )


def load_logistic_regression_config(raw: Mapping[str, Any] | None) -> LogisticRegressionConfig:
    raw = raw or {}
    defaults = LogisticRegressionConfig()
    class_weight = raw.get("class_weight", defaults.class_weight)
    return LogisticRegressionConfig(
        C=float(raw.get("C", defaults.C)),
        max_iter=int(raw.get("max_iter", defaults.max_iter)),
        solver=str(raw.get("solver", defaults.solver)),
        class_weight=None if class_weight in (None, "null", "None") else str(class_weight),
    )


def load_training_config(path: Path) -> TrainingConfig:
    raw = _load_yaml(path)
    defaults = TrainingConfig()
    return TrainingConfig(
        experiment_name=str(raw.get("experiment_name", defaults.experiment_name)),
        model_output_dir=_as_path(
            raw.get("model_output_dir"), str(defaults.model_output_dir)
        ),
        mlflow_tracking_uri=str(
            raw.get("mlflow_tracking_uri", defaults.mlflow_tracking_uri)
        ),
        dataset_version=str(raw.get("dataset_version", defaults.dataset_version)),
        random_state=int(raw.get("random_state", defaults.random_state)),
        tfidf=load_tfidf_config(raw.get("tfidf")),
        logistic_regression=load_logistic_regression_config(
            raw.get("logistic_regression")
        ),
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = _load_yaml(path)
    defaults = EvaluationConfig()
    return EvaluationConfig(
        output_dir=_as_path(raw.get("output_dir"), str(defaults.output_dir)),
        save_confusion_matrix=bool(
            raw.get("save_confusion_matrix", defaults.save_confusion_matrix)
        ),
        save_metrics_json=bool(raw.get("save_metrics_json", defaults.save_metrics_json)),
    )


def load_model_config(path: Path) -> tuple[ModelConfig, TransformerTrainingConfig, TokenizationConfig]:
    raw = _load_yaml(path)
    model_defaults = ModelConfig()
    training_defaults = TransformerTrainingConfig()
    tokenization_defaults = TokenizationConfig()
    model = raw.get("model", {})
    training = raw.get("training", {})
    tokenization = raw.get("tokenization", {})

    if not isinstance(model, dict) or not isinstance(training, dict) or not isinstance(tokenization, dict):
        raise ValueError("Model configuration sections must be mappings.")

    label_names = tuple(model.get("label_names", list(model_defaults.label_names)))
    if len(label_names) != 2:
        raise ValueError("Exactly two label names are required for binary classification.")
    label_names = (str(label_names[0]), str(label_names[1]))

    return (
        ModelConfig(
            name=str(model.get("name", model_defaults.name)),
            max_length=int(model.get("max_length", model_defaults.max_length)),
            num_labels=int(model.get("num_labels", model_defaults.num_labels)),
            model_output_dir=_as_path(model.get("model_output_dir"), str(model_defaults.model_output_dir)),
            label_names=label_names,
            tokenizer_name=str(model.get("tokenizer_name", model_defaults.tokenizer_name)),
            use_fast_tokenizer=bool(model.get("use_fast_tokenizer", model_defaults.use_fast_tokenizer)),
        ),
        TransformerTrainingConfig(
            batch_size=int(training.get("batch_size", training_defaults.batch_size)),
            learning_rate=float(training.get("learning_rate", training_defaults.learning_rate)),
            epochs=int(training.get("epochs", training_defaults.epochs)),
            weight_decay=float(training.get("weight_decay", training_defaults.weight_decay)),
            warmup_ratio=float(training.get("warmup_ratio", training_defaults.warmup_ratio)),
            evaluation_strategy=str(training.get("evaluation_strategy", training_defaults.evaluation_strategy)),
            save_strategy=str(training.get("save_strategy", training_defaults.save_strategy)),
            load_best_model_at_end=bool(training.get("load_best_model_at_end", training_defaults.load_best_model_at_end)),
            metric_for_best_model=str(training.get("metric_for_best_model", training_defaults.metric_for_best_model)),
            greater_is_better=bool(training.get("greater_is_better", training_defaults.greater_is_better)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", training_defaults.gradient_accumulation_steps)),
            seed=int(training.get("seed", training_defaults.seed)),
        ),
        TokenizationConfig(
            truncation=bool(tokenization.get("truncation", tokenization_defaults.truncation)),
            padding=str(tokenization.get("padding", tokenization_defaults.padding)),
        ),
    )


def load_comparison_output_path() -> Path:
    return Path("artifacts/evaluation/model_comparison.csv")


def as_serializable_dict(config: object) -> dict[str, Any]:
    """Convert a dataclass config into a JSON/YAML-friendly dictionary."""

    return asdict(config)