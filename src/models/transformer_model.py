"""HuggingFace transformer model factory and label mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.config.pipeline import ModelConfig


@dataclass(frozen=True, slots=True)
class TransformerArtifacts:
    model_name: str
    tokenizer_name: str


def build_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        model_config.tokenizer_name,
        use_fast=model_config.use_fast_tokenizer,
    )


def build_sequence_classification_model(model_config: ModelConfig) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        model_config.name,
        num_labels=model_config.num_labels,
        id2label={index: label for index, label in enumerate(model_config.label_names)},
        label2id={label: index for index, label in enumerate(model_config.label_names)},
    )
    return AutoModelForSequenceClassification.from_pretrained(model_config.name, config=config)


def build_transformer_artifacts(model_config: ModelConfig) -> TransformerArtifacts:
    return TransformerArtifacts(
        model_name=model_config.name,
        tokenizer_name=model_config.tokenizer_name,
    )


def build_model_directory(model_config: ModelConfig) -> str:
    return str(model_config.model_output_dir)