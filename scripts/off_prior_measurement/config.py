from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SubjectSpec:
    subject_id: str
    hf_subset: str
    class_name: str
    class_prompt: str
    class_context_prompt: str
    hard_control_prompt: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model_id: str
    prediction_type: str
    device: str
    dtype: str
    resolution: int
    dataset_repo: str
    subject_manifest: Path
    cache_dir: Path
    output_dir: Path
    debug_output_dir: Path
    timesteps: list[int]
    noise_seeds: list[int]
    conditionings: list[str]
    control_images_per_subject: int
    batch_size: int
    save_debug_tensors: bool
    subjects: list[SubjectSpec]
    dataset_source: str = "huggingface"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _require_keys(data: dict[str, Any], keys: list[str], source: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing keys in {source}: {missing}")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    required = [
        "experiment_name",
        "model_id",
        "prediction_type",
        "device",
        "dtype",
        "resolution",
        "dataset_repo",
        "subject_manifest",
        "cache_dir",
        "output_dir",
        "debug_output_dir",
        "timesteps",
        "noise_seeds",
        "conditionings",
        "control_images_per_subject",
        "batch_size",
        "save_debug_tensors",
    ]
    _require_keys(raw, required, config_path)

    subject_manifest = Path(raw["subject_manifest"])
    subjects_raw = _read_yaml(subject_manifest).get("subjects", [])
    subjects = [SubjectSpec(**subject) for subject in subjects_raw]
    if not subjects:
        raise ValueError(f"No subjects configured in {subject_manifest}")

    return ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        model_id=str(raw["model_id"]),
        prediction_type=str(raw["prediction_type"]),
        device=str(raw["device"]),
        dtype=str(raw["dtype"]),
        resolution=int(raw["resolution"]),
        dataset_repo=str(raw["dataset_repo"]),
        subject_manifest=subject_manifest,
        cache_dir=Path(raw["cache_dir"]),
        output_dir=Path(raw["output_dir"]),
        debug_output_dir=Path(raw["debug_output_dir"]),
        timesteps=[int(timestep) for timestep in raw["timesteps"]],
        noise_seeds=[int(seed) for seed in raw["noise_seeds"]],
        conditionings=["null" if item is None else str(item) for item in raw["conditionings"]],
        control_images_per_subject=int(raw["control_images_per_subject"]),
        batch_size=int(raw["batch_size"]),
        save_debug_tensors=bool(raw["save_debug_tensors"]),
        subjects=subjects,
        dataset_source=str(raw.get("dataset_source", "huggingface")),
    )
