from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.config import ExperimentConfig, SubjectSpec, load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def conditioning_prompt(subject: SubjectSpec, conditioning_key: str) -> str:
    if conditioning_key == "null":
        return ""
    if conditioning_key == "class":
        return subject.class_prompt
    if conditioning_key == "class_context":
        return subject.class_context_prompt
    raise ValueError(f"Unsupported conditioning key: {conditioning_key}")


def download_dreambooth_subjects(config: ExperimentConfig) -> Path:
    from huggingface_hub import snapshot_download

    dataset_root = config.cache_dir / "dreambooth_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.dataset_repo,
        repo_type="dataset",
        local_dir=dataset_root,
        allow_patterns=[f"dataset/{subject.hf_subset}/*" for subject in config.subjects],
    )
    return dataset_root / "dataset"


def _iter_subject_images(dataset_root: Path, subject: SubjectSpec) -> list[Path]:
    subject_dir = dataset_root / subject.hf_subset
    if not subject_dir.exists():
        raise FileNotFoundError(f"DreamBooth subject directory not found: {subject_dir}")
    images = sorted(path for path in subject_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found for subject: {subject.subject_id}")
    return images


def build_reference_manifest(
    dataset_root: Path,
    subjects: list[SubjectSpec],
    conditionings: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for subject in subjects:
        for image_path in _iter_subject_images(dataset_root, subject):
            image_id = image_path.stem
            for conditioning_key in conditionings:
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "source_group": "dreambooth_reference",
                        "reference_regime": "standard",
                        "class_name": subject.class_name,
                        "class_prompt": subject.class_prompt,
                        "class_context_prompt": subject.class_context_prompt,
                        "conditioning_key": conditioning_key,
                        "conditioning_prompt": conditioning_prompt(subject, conditioning_key),
                    }
                )
    return pd.DataFrame(rows)


def write_combined_manifest(config_path: str | Path) -> Path:
    from scripts.off_prior_measurement.generate_controls import build_control_manifest
    from scripts.off_prior_measurement.roundtrip_controls import build_roundtrip_manifest

    config = load_config(config_path)
    dataset_root = download_dreambooth_subjects(config)
    reference = build_reference_manifest(dataset_root, config.subjects, config.conditionings)

    manifest_dir = config.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest_path = manifest_dir / "reference_manifest.csv"
    reference.to_csv(reference_manifest_path, index=False)

    generated_root = config.cache_dir / "generated_controls"
    controls = build_control_manifest(config.subjects, generated_root, config.conditionings)
    roundtrip_root = config.cache_dir / "vae_roundtrip_controls"
    roundtrip = build_roundtrip_manifest(reference_manifest_path, roundtrip_root)

    combined = pd.concat([reference, controls, roundtrip], ignore_index=True)
    path = manifest_dir / "combined_manifest.csv"
    combined.to_csv(path, index=False)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = write_combined_manifest(args.config)
    print(path)


if __name__ == "__main__":
    main()
