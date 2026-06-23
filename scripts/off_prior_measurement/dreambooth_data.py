from __future__ import annotations

import base64
import http.client
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import pandas as pd

from scripts.off_prior_measurement.config import ExperimentConfig, SubjectSpec, load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GITHUB_DREAMBOOTH_API = "https://api.github.com/repos/google/dreambooth/contents/dataset"
GITHUB_DREAMBOOTH_ZIP = "https://codeload.github.com/google/dreambooth/zip/refs/heads/main"
GITHUB_HEADERS = {
    "User-Agent": "diffusion-personalization-target-alignment",
    "Accept": "application/vnd.github+json",
}


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
    if config.dataset_source == "github_api":
        _download_dreambooth_subjects_from_github_api(
            dataset_root / "dataset",
            config.subjects,
            image_limit=config.reference_images_per_subject,
        )
        return dataset_root / "dataset"
    if config.dataset_source == "github":
        _download_dreambooth_subjects_from_github(
            dataset_root / "dataset",
            config.subjects,
            image_limit=config.reference_images_per_subject,
        )
        return dataset_root / "dataset"
    if config.dataset_source != "huggingface":
        raise ValueError(f"Unsupported dataset_source: {config.dataset_source}")
    try:
        snapshot_download(
            repo_id=config.dataset_repo,
            repo_type="dataset",
            local_dir=dataset_root,
            allow_patterns=[f"dataset/{subject.hf_subset}/*" for subject in config.subjects],
        )
    except Exception as exc:
        if config.dataset_repo != "google/dreambooth":
            raise
        print(f"Hugging Face DreamBooth download failed; falling back to GitHub: {exc}")
        _download_dreambooth_subjects_from_github(
            dataset_root / "dataset",
            config.subjects,
            image_limit=config.reference_images_per_subject,
        )
    return dataset_root / "dataset"


def _download_dreambooth_subjects_from_github(
    dataset_root: Path,
    subjects: list[SubjectSpec],
    image_limit: int | None = None,
) -> None:
    try:
        _download_dreambooth_subjects_from_github_zip(dataset_root, subjects, image_limit=image_limit)
        return
    except Exception as exc:
        print(f"GitHub zip DreamBooth download failed; falling back to GitHub API files: {exc}")

    for subject in subjects:
        subject_dir = dataset_root / subject.hf_subset
        subject_dir.mkdir(parents=True, exist_ok=True)
        api_url = f"{GITHUB_DREAMBOOTH_API}/{subject.hf_subset}?ref=main"
        with _open_url(api_url, timeout=60) as response:
            entries = json.loads(response.read().decode("utf-8"))
        image_entries = _limited_image_entries(entries, image_limit)
        for entry in image_entries:
            name = str(entry.get("name", ""))
            destination = subject_dir / name
            if destination.exists() and destination.stat().st_size > 0:
                continue
            download_url = entry.get("download_url")
            if not download_url:
                continue
            with _open_url(str(download_url), timeout=120) as response:
                destination.write_bytes(response.read())


def _download_dreambooth_subjects_from_github_api(
    dataset_root: Path,
    subjects: list[SubjectSpec],
    image_limit: int | None = None,
) -> None:
    for subject in subjects:
        subject_dir = dataset_root / subject.hf_subset
        subject_dir.mkdir(parents=True, exist_ok=True)
        api_url = f"{GITHUB_DREAMBOOTH_API}/{subject.hf_subset}?ref=main"
        with _open_url(api_url, timeout=60) as response:
            entries = json.loads(response.read().decode("utf-8"))
        image_entries = _limited_image_entries(entries, image_limit)
        for entry in image_entries:
            name = str(entry.get("name", ""))
            destination = subject_dir / name
            if destination.exists() and destination.stat().st_size > 0:
                continue
            file_url = entry.get("url")
            if not file_url:
                continue
            with _open_url(str(file_url), timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = _github_api_file_content(payload, subject.hf_subset, name)
            destination.write_bytes(content)


def _limited_image_entries(entries: list[dict[str, object]], image_limit: int | None) -> list[dict[str, object]]:
    image_entries = sorted(
        (
            entry
            for entry in entries
            if entry.get("type") == "file" and Path(str(entry.get("name", ""))).suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda entry: str(entry.get("name", "")),
    )
    if image_limit is None:
        return image_entries
    return image_entries[:image_limit]


def _github_api_file_content(payload: dict[str, object], subset: str, name: str) -> bytes:
    if payload.get("encoding") == "base64" and payload.get("content"):
        return base64.b64decode(str(payload["content"]).replace("\n", ""))
    git_url = payload.get("git_url")
    if git_url:
        blob = _read_json_url(str(git_url), timeout=120)
        if blob.get("encoding") == "base64" and blob.get("content"):
            return base64.b64decode(str(blob["content"]).replace("\n", ""))
    raise ValueError(f"GitHub API did not return base64 content for {subset}/{name}")


def _read_json_url(url: str, timeout: int, attempts: int = 3) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with _open_url(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except http.client.IncompleteRead as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read JSON URL: {url}")


def _download_dreambooth_subjects_from_github_zip(
    dataset_root: Path,
    subjects: list[SubjectSpec],
    image_limit: int | None = None,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_root.parent / "dreambooth_github_main.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        tmp_path = zip_path.with_suffix(".zip.tmp")
        with _open_url(GITHUB_DREAMBOOTH_ZIP, timeout=300, accept="application/zip") as response:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp_path.replace(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for subject in subjects:
            prefix = f"dreambooth-main/dataset/{subject.hf_subset}/"
            members = sorted(
                name
                for name in names
                if name.startswith(prefix) and Path(name).suffix.lower() in IMAGE_SUFFIXES
            )
            if image_limit is not None:
                members = members[:image_limit]
            if not members:
                raise FileNotFoundError(f"No GitHub zip images found for subject: {subject.hf_subset}")
            subject_dir = dataset_root / subject.hf_subset
            subject_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                destination = subject_dir / Path(member).name
                if destination.exists() and destination.stat().st_size > 0:
                    continue
                destination.write_bytes(archive.read(member))


def _open_url(url: str, timeout: int, accept: str = "application/vnd.github+json"):
    headers = dict(GITHUB_HEADERS)
    headers["Accept"] = accept
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _iter_subject_images(dataset_root: Path, subject: SubjectSpec, image_limit: int | None = None) -> list[Path]:
    subject_dir = dataset_root / subject.hf_subset
    if not subject_dir.exists():
        raise FileNotFoundError(f"DreamBooth subject directory not found: {subject_dir}")
    images = sorted(path for path in subject_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if image_limit is not None:
        images = images[:image_limit]
    if not images:
        raise FileNotFoundError(f"No images found for subject: {subject.subject_id}")
    return images


def build_reference_manifest(
    dataset_root: Path,
    subjects: list[SubjectSpec],
    conditionings: list[str],
    image_limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for subject in subjects:
        for image_path in _iter_subject_images(dataset_root, subject, image_limit=image_limit):
            image_id = image_path.stem
            for conditioning_key in conditionings:
                rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "source_group": "dreambooth_reference",
                        "reference_regime": "standard_reference",
                        "hardness_axis": "none",
                        "source_standard_image": "",
                        "variant_id": "",
                        "transform_parameters": "{}",
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
    from scripts.off_prior_measurement.hard_references import build_hard_reference_manifest
    from scripts.off_prior_measurement.roundtrip_controls import build_roundtrip_manifest

    config = load_config(config_path)
    dataset_root = download_dreambooth_subjects(config)
    reference = build_reference_manifest(
        dataset_root,
        config.subjects,
        config.conditionings,
        image_limit=config.reference_images_per_subject,
    )

    manifest_dir = config.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest_path = manifest_dir / "reference_manifest.csv"
    reference.to_csv(reference_manifest_path, index=False)

    hard_reference_root = config.cache_dir / "hard_references"
    hard_reference = build_hard_reference_manifest(
        reference_manifest_path=reference_manifest_path,
        hard_root=hard_reference_root,
        variants=config.hard_reference_variants or [],
    )
    if config.hard_reference_limit_per_subject is not None and not hard_reference.empty:
        hard_reference = (
            hard_reference.sort_values(["subject_id", "image_id", "variant_id", "conditioning_key"])
            .groupby(["subject_id", "variant_id", "conditioning_key"], as_index=False)
            .head(config.hard_reference_limit_per_subject)
            .reset_index(drop=True)
        )
    hard_reference_manifest_path = manifest_dir / "hard_reference_manifest.csv"
    hard_reference.to_csv(hard_reference_manifest_path, index=False)

    generated_root = config.cache_dir / "generated_controls"
    controls = build_control_manifest(config.subjects, generated_root, config.conditionings)
    roundtrip_root = config.cache_dir / "vae_roundtrip_controls"
    roundtrip = build_roundtrip_manifest(reference_manifest_path, roundtrip_root)

    combined = pd.concat([reference, hard_reference, controls, roundtrip], ignore_index=True)
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
