from pathlib import Path

import pytest

from src.config.pipeline import (
    DatasetConfig,
    ModelConfig,
    PreprocessingConfig,
    SplitConfig,
    load_dataset_config,
    load_model_config,
    load_split_config_from_dataset,
)


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_dataset_config_with_allowed_labels(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "dataset.yaml",
                "\n".join(
                        [
                                "name: isot",
                                'version: "1.1"',
                                "source_type: local",
                                "source_path: data/raw/custom.csv",
                                "file_format: csv",
                                "text_column: text",
                                "label_column: label",
                                "allowed_labels:",
                                "  - fake",
                                "  - real",
                                "preprocessing:",
                                "  lowercase: false",
                                "  remove_urls: true",
                                "  remove_html: true",
                                "  remove_whitespace: false",
                                "  remove_special_characters: false",
                                "  special_characters_pattern: '[^a-zA-Z0-9\\s]'",
                        ]
                ),
    )

    config = load_dataset_config(config_path)

    assert config.name == "isot"
    assert config.allowed_labels == ("fake", "real")
    assert config.preprocessing.lowercase is False
    assert config.preprocessing.remove_whitespace is False


def test_load_dataset_config_without_allowed_labels_uses_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
name: isot
version: "1.1"
source_type: local
source_path: data/raw/custom.csv
file_format: csv
text_column: text
label_column: label
""".strip(),
    )

    config = load_dataset_config(config_path)

    defaults = DatasetConfig()
    preprocessing_defaults = PreprocessingConfig()

    assert config.allowed_labels == defaults.allowed_labels
    assert config.preprocessing == preprocessing_defaults


def test_load_split_config_from_dataset_without_split_uses_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
name: isot
version: "1.1"
source_type: local
source_path: data/raw/custom.csv
file_format: csv
text_column: text
label_column: label
""".strip(),
    )

    split_config = load_split_config_from_dataset(config_path)

    assert split_config == SplitConfig()


def test_load_dataset_config_rejects_invalid_allowed_labels_type(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "dataset.yaml",
        """
name: isot
version: "1.1"
source_type: local
source_path: data/raw/custom.csv
file_format: csv
text_column: text
label_column: label
allowed_labels: fake
""".strip(),
    )

    with pytest.raises(ValueError, match="allowed_labels must be a sequence"):
        load_dataset_config(config_path)


def test_load_model_config_with_label_names(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "model.yaml",
        "\n".join(
            [
                "model:",
                "  name: roberta-base",
                "  max_length: 256",
                "  num_labels: 2",
                "  model_output_dir: models/roberta",
                "  label_names:",
                "    - fake",
                "    - real",
                "  tokenizer_name: roberta-base",
                "  use_fast_tokenizer: true",
                "training:",
                "  batch_size: 8",
                "tokenization:",
                "  truncation: true",
                "  padding: dynamic",
            ]
        ),
    )

    model_config, training_config, tokenization_config = load_model_config(config_path)

    assert model_config.label_names == ("fake", "real")
    assert model_config.max_length == 256
    assert training_config.batch_size == 8
    assert tokenization_config.padding == "dynamic"


def test_load_model_config_without_label_names_uses_defaults(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "model.yaml",
        "\n".join(
            [
                "model:",
                "  name: roberta-base",
                "  num_labels: 2",
                "training:",
                "  batch_size: 16",
                "tokenization:",
                "  truncation: true",
                "  padding: dynamic",
            ]
        ),
    )

    model_config, _, _ = load_model_config(config_path)

    assert model_config.label_names == ModelConfig().label_names
