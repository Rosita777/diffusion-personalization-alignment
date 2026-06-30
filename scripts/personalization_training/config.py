from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.personalization_training.target_alignment import LFLateAlignmentConfig


ALLOWED_CONDITIONS = {"vanilla", "dadt_lf_late", "dadt_residual_gate", "dadt_cfg_residual_gate"}


@dataclass(frozen=True)
class SubjectConfig:
    subject_id: str
    class_name: str
    instance_prompt: str
    class_prompt: str
    image_paths: list[Path]


@dataclass(frozen=True)
class TrainingConfig:
    condition: str
    max_train_steps: int
    learning_rate: float
    lora_rank: int
    seed: int
    train_batch_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    prompts: list[str]
    class_prompts: list[str]
    num_images_per_prompt: int


@dataclass(frozen=True)
class Stage2AConfig:
    model_id: str
    output_dir: Path
    resolution: int
    subjects: list[SubjectConfig]
    training: TrainingConfig
    alignment: LFLateAlignmentConfig
    evaluation: EvaluationConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _require_mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping '{key}' in {source}")
    return value


def _require_list(data: dict[str, Any], key: str, source: Path) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing non-empty list '{key}' in {source}")
    return value


def _path_list(items: list[Any]) -> list[Path]:
    paths = [Path(str(item)) for item in items]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _load_subjects(raw: dict[str, Any], source: Path) -> list[SubjectConfig]:
    subjects = []
    for item in _require_list(raw, "subjects", source):
        if not isinstance(item, dict):
            raise ValueError(f"Subject entries must be mappings in {source}")
        image_items = _require_list(item, "image_paths", source)
        subjects.append(
            SubjectConfig(
                subject_id=str(item["subject_id"]),
                class_name=str(item["class_name"]),
                instance_prompt=str(item["instance_prompt"]),
                class_prompt=str(item["class_prompt"]),
                image_paths=_path_list(image_items),
            )
        )
    return subjects


def _load_training(raw: dict[str, Any], source: Path) -> TrainingConfig:
    training = _require_mapping(raw, "training", source)
    condition = str(training["condition"])
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"Unsupported training condition: {condition}")
    max_train_steps = int(training["max_train_steps"])
    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    lora_rank = int(training["lora_rank"])
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    train_batch_size = int(training.get("train_batch_size", 1))
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    return TrainingConfig(
        condition=condition,
        max_train_steps=max_train_steps,
        learning_rate=float(training["learning_rate"]),
        lora_rank=lora_rank,
        seed=int(training.get("seed", 0)),
        train_batch_size=train_batch_size,
    )


def _load_alignment(raw: dict[str, Any], source: Path) -> LFLateAlignmentConfig:
    alignment = _require_mapping(raw, "alignment", source)
    return LFLateAlignmentConfig(
        alpha=float(alignment["alpha"]),
        late_timestep_threshold=int(alignment["late_timestep_threshold"]),
        low_radius=int(alignment["low_radius"]),
        mid_radius=int(alignment["mid_radius"]),
        residual_gate_quantile=float(alignment.get("residual_gate_quantile", 0.75)),
        residual_gate_keep=float(alignment.get("residual_gate_keep", 0.5)),
    )


def _load_evaluation(raw: dict[str, Any], source: Path) -> EvaluationConfig:
    evaluation = _require_mapping(raw, "evaluation", source)
    return EvaluationConfig(
        prompts=[str(item) for item in _require_list(evaluation, "prompts", source)],
        class_prompts=[str(item) for item in _require_list(evaluation, "class_prompts", source)],
        num_images_per_prompt=int(evaluation.get("num_images_per_prompt", 1)),
    )


def load_training_config(path: str | Path) -> Stage2AConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    return Stage2AConfig(
        model_id=str(raw["model_id"]),
        output_dir=Path(str(raw["output_dir"])),
        resolution=int(raw.get("resolution", 512)),
        subjects=_load_subjects(raw, config_path),
        training=_load_training(raw, config_path),
        alignment=_load_alignment(raw, config_path),
        evaluation=_load_evaluation(raw, config_path),
    )
