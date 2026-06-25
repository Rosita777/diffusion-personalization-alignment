# Stage 1.4 Target-Gap Source Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a Stage 1.4 pipeline that separates VAE/projection artifacts, ordinary real-image domain gap, and DreamBooth subject-specific target gap before any personalization fine-tuning.

**Architecture:** Keep the existing `scripts/off_prior_measurement/` pipeline intact and add a source-decomposition branch beside it. The new branch builds a four-source manifest, measures each image paired with its VAE roundtrip image, computes vector-level residual projection metrics online, then summarizes source gaps and writes a Go / Pivot conclusion.

**Tech Stack:** Python 3.10 in `/home/deepseek_VG/.conda/envs/dyme`, pandas, NumPy, SciPy DCT, Pillow, PyYAML, matplotlib, pytest, existing SD 1.5 backend code and CSV string-preservation helpers.

---

## Scope

In scope:

- CPU unit tests for all new logic.
- A source-decomposition smoke config and ordinary-real manifest contract.
- Paired residual projection metrics from real image and VAE-roundtrip image measurements.
- Summary tables for `real_domain_gap`, `subject_specific_gap`, and `natural_hard_gap`.
- Figures and a Stage 1.4 Go / Pivot conclusion.

Out of scope:

- Personalization fine-tuning.
- Large dataset downloads.
- Saving full residual tensors by default.
- Modifying completed `ladder_v2` or `ladder_v2_clean` outputs.

## File Structure

Create:

```text
configs/off_prior_measurement_v0/source_decomp_v1.yaml
data/manifests/ordinary_real_controls_v1.yaml
scripts/off_prior_measurement/source_decomp_manifest.py
scripts/off_prior_measurement/source_decomp_measure.py
scripts/off_prior_measurement/source_decomp_summarize.py
scripts/off_prior_measurement/source_decomp_plot.py
scripts/off_prior_measurement/source_decomp_conclusion.py
tests/off_prior_measurement/test_source_decomp_manifest.py
tests/off_prior_measurement/test_source_decomp_measure.py
tests/off_prior_measurement/test_source_decomp_summarize.py
tests/off_prior_measurement/test_source_decomp_plot.py
tests/off_prior_measurement/test_source_decomp_conclusion.py
experiments/off_prior_measurement_v0/source_decomp_v1/README.md
```

Modify:

```text
scripts/off_prior_measurement/config.py
scripts/off_prior_measurement/metrics.py
tests/off_prior_measurement/test_config.py
tests/off_prior_measurement/test_metrics.py
README.md
docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md
```

Commit-worthy generated outputs after a real smoke run:

```text
experiments/off_prior_measurement_v0/source_decomp_v1/config_resolved.yaml
experiments/off_prior_measurement_v0/source_decomp_v1/manifests/source_decomp_manifest.csv
experiments/off_prior_measurement_v0/source_decomp_v1/measurements/raw_source_decomp_metrics.csv
experiments/off_prior_measurement_v0/source_decomp_v1/summaries/source_group_summary.csv
experiments/off_prior_measurement_v0/source_decomp_v1/summaries/source_gap_summary.csv
experiments/off_prior_measurement_v0/source_decomp_v1/summaries/timestep_frequency_summary.csv
experiments/off_prior_measurement_v0/source_decomp_v1/figures/source_gap_bars.png
experiments/off_prior_measurement_v0/source_decomp_v1/figures/artifact_fraction_by_source.png
experiments/off_prior_measurement_v0/source_decomp_v1/figures/clean_timestep_curves.png
experiments/off_prior_measurement_v0/source_decomp_v1/conclusion.md
```

## Task 1: Extend Config For Source Decomposition

**Files:**

- Modify: `scripts/off_prior_measurement/config.py`
- Modify: `tests/off_prior_measurement/test_config.py`
- Create: `configs/off_prior_measurement_v0/source_decomp_v1.yaml`
- Create: `data/manifests/ordinary_real_controls_v1.yaml`

- [x] **Step 1: Write the failing config test**

Append this test to `tests/off_prior_measurement/test_config.py`:

```python
def test_load_config_parses_source_decomp_fields(tmp_path):
    subject_path = tmp_path / "subjects.yaml"
    ordinary_path = tmp_path / "ordinary.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: dog
    hf_subset: dog
    class_name: dog
    class_prompt: a photo of a dog
    class_context_prompt: a photo of a dog in a natural scene
    hard_control_prompt: a photo of a dog in a cluttered room
""".strip(),
        encoding="utf-8",
    )
    ordinary_path.write_text(
        """
ordinary_real_controls:
  - class_name: dog
    image_id: dog_real_00
    image_path: data/local_real_controls/dog/dog_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: local research-only placeholder
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_name: source_decomp_v1
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github_api
subject_manifest: {subject_path}
ordinary_real_manifest: {ordinary_path}
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/source_decomp_v1
debug_output_dir: outputs/off_prior_measurement_v0/source_decomp_v1
timesteps: [50, 200]
noise_seeds: [0, 1]
conditionings: ["class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
source_decomp_images_per_class: 2
source_decomp_save_debug_tensors: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "source_decomp_v1"
    assert config.ordinary_real_manifest == ordinary_path
    assert config.source_decomp_images_per_class == 2
    assert config.source_decomp_save_debug_tensors is False
```

- [x] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_config.py::test_load_config_parses_source_decomp_fields -v
```

Expected: failure because `ExperimentConfig` has no `ordinary_real_manifest` field.

- [x] **Step 3: Extend config dataclass and loader**

Add these fields to `ExperimentConfig`:

```python
ordinary_real_manifest: Path | None = None
source_decomp_images_per_class: int | None = None
source_decomp_save_debug_tensors: bool = False
```

In `load_config`, parse:

```python
ordinary_real_manifest = raw.get("ordinary_real_manifest")
source_decomp_images_per_class = raw.get("source_decomp_images_per_class")
source_decomp_save_debug_tensors = bool(raw.get("source_decomp_save_debug_tensors", False))
```

Return:

```python
ordinary_real_manifest=None if ordinary_real_manifest is None else Path(ordinary_real_manifest),
source_decomp_images_per_class=None
if source_decomp_images_per_class is None
else int(source_decomp_images_per_class),
source_decomp_save_debug_tensors=source_decomp_save_debug_tensors,
```

- [x] **Step 4: Add smoke config and ordinary-real manifest template**

Create `configs/off_prior_measurement_v0/source_decomp_v1.yaml`:

```yaml
experiment_name: source_decomp_v1
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github_api
subject_manifest: data/manifests/dreambooth_ladder_subjects.yaml
ordinary_real_manifest: data/manifests/ordinary_real_controls_v1.yaml
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/source_decomp_v1
debug_output_dir: outputs/off_prior_measurement_v0/source_decomp_v1
timesteps: [50, 200, 500, 800, 999]
noise_seeds: [0, 1, 2, 3, 4]
conditionings: ["class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
source_decomp_images_per_class: 2
source_decomp_save_debug_tensors: false
reference_images_per_subject: 1
```

Create `data/manifests/ordinary_real_controls_v1.yaml` with explicit local placeholders:

```yaml
ordinary_real_controls:
  - class_name: dog
    image_id: dog_real_00
    image_path: data/local_real_controls/dog/dog_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: user-provided local image, not committed
  - class_name: cat
    image_id: cat_real_00
    image_path: data/local_real_controls/cat/cat_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: user-provided local image, not committed
  - class_name: backpack
    image_id: backpack_real_00
    image_path: data/local_real_controls/backpack/backpack_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: user-provided local image, not committed
  - class_name: vase
    image_id: vase_real_00
    image_path: data/local_real_controls/vase/vase_real_00.jpg
    source_dataset: local_real_controls
    source_license_note: user-provided local image, not committed
```

- [x] **Step 5: Run config tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_config.py -v
```

Expected: all config tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/config.py tests/off_prior_measurement/test_config.py configs/off_prior_measurement_v0/source_decomp_v1.yaml data/manifests/ordinary_real_controls_v1.yaml
git commit -m "feat: configure source decomposition"
```

## Task 2: Projection Decomposition Metrics

**Files:**

- Modify: `scripts/off_prior_measurement/metrics.py`
- Modify: `tests/off_prior_measurement/test_metrics.py`

- [x] **Step 1: Write failing projection metric tests**

Append to `tests/off_prior_measurement/test_metrics.py`:

```python
from scripts.off_prior_measurement.metrics import residual_projection_metrics


def test_residual_projection_metrics_zero_clean_when_residuals_match():
    v_ref = np.array([2.0, 0.0], dtype=np.float32)
    v_base = np.array([0.0, 0.0], dtype=np.float32)
    rt_ref = np.array([2.0, 0.0], dtype=np.float32)
    rt_base = np.array([0.0, 0.0], dtype=np.float32)

    metrics = residual_projection_metrics(v_ref, v_base, rt_ref, rt_base)

    assert round(metrics["artifact_fraction"], 6) == 1.0
    assert round(metrics["clean_norm"], 6) == 0.0
    assert round(metrics["artifact_cosine"], 6) == 1.0


def test_residual_projection_metrics_keeps_orthogonal_clean_residual():
    v_ref = np.array([0.0, 2.0], dtype=np.float32)
    v_base = np.array([0.0, 0.0], dtype=np.float32)
    rt_ref = np.array([2.0, 0.0], dtype=np.float32)
    rt_base = np.array([0.0, 0.0], dtype=np.float32)

    metrics = residual_projection_metrics(v_ref, v_base, rt_ref, rt_base)

    assert round(metrics["artifact_fraction"], 6) == 0.0
    assert round(metrics["clean_fraction"], 6) == 1.0
    assert round(metrics["artifact_cosine"], 6) == 0.0
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_metrics.py -v
```

Expected: import failure for `residual_projection_metrics`.

- [x] **Step 3: Implement projection metrics**

Add to `scripts/off_prior_measurement/metrics.py`:

```python
def residual_projection_metrics(v_ref, v_base, rt_v_ref, rt_v_base) -> dict[str, float]:
    ref = _to_numpy(v_ref).astype(np.float64).reshape(-1)
    base = _to_numpy(v_base).astype(np.float64).reshape(-1)
    rt_ref = _to_numpy(rt_v_ref).astype(np.float64).reshape(-1)
    rt_base = _to_numpy(rt_v_base).astype(np.float64).reshape(-1)
    residual = ref - base
    artifact_residual = rt_ref - rt_base
    residual_norm = max(float(np.linalg.norm(residual)), 1e-8)
    artifact_norm = max(float(np.linalg.norm(artifact_residual)), 1e-8)
    ref_norm = max(float(np.linalg.norm(ref)), 1e-8)
    coeff = float(np.dot(residual, artifact_residual) / (artifact_norm**2 + 1e-8))
    artifact_component = coeff * artifact_residual
    clean = residual - artifact_component
    artifact_component_norm = float(np.linalg.norm(artifact_component))
    clean_norm_abs = float(np.linalg.norm(clean))
    artifact_cosine = float(np.dot(residual, artifact_residual) / (residual_norm * artifact_norm))
    artifact_cosine = min(1.0, max(-1.0, artifact_cosine))
    return {
        "raw_norm": float(residual_norm / ref_norm),
        "artifact_coeff": coeff,
        "artifact_cosine": artifact_cosine,
        "artifact_fraction": artifact_component_norm / residual_norm,
        "clean_fraction": clean_norm_abs / residual_norm,
        "clean_norm": clean_norm_abs / ref_norm,
    }
```

- [x] **Step 4: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_metrics.py -v
```

Expected: all metrics tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/metrics.py tests/off_prior_measurement/test_metrics.py
git commit -m "feat: add residual projection metrics"
```

## Task 3: Source-Decomposition Manifest Builder

**Files:**

- Create: `scripts/off_prior_measurement/source_decomp_manifest.py`
- Create: `tests/off_prior_measurement/test_source_decomp_manifest.py`

- [x] **Step 1: Write failing manifest tests**

Create `tests/off_prior_measurement/test_source_decomp_manifest.py`:

```python
import pandas as pd
import pytest
import yaml

from scripts.off_prior_measurement.source_decomp_manifest import (
    build_source_decomp_manifest,
    load_ordinary_real_controls,
)


def test_load_ordinary_real_controls_preserves_string_ids(tmp_path):
    manifest = tmp_path / "ordinary.yaml"
    image = tmp_path / "dog_real_00.jpg"
    image.write_bytes(b"fake")
    manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "00",
                        "image_path": str(image),
                        "source_dataset": "local",
                        "source_license_note": "local placeholder",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    controls = load_ordinary_real_controls(manifest)

    assert controls.iloc[0]["image_id"] == "00"
    assert controls.iloc[0]["source_group"] == "ordinary_real_control"


def test_build_source_decomp_manifest_rejects_missing_ordinary_real_controls(tmp_path):
    reference_path = tmp_path / "reference.csv"
    controls_path = tmp_path / "controls.csv"
    ordinary_path = tmp_path / "ordinary.yaml"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "db_00",
                "image_path": str(tmp_path / "db_00.jpg"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "base_00",
                "image_path": str(tmp_path / "base_00.png"),
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(controls_path, index=False)
    ordinary_path.write_text("ordinary_real_controls: []", encoding="utf-8")

    with pytest.raises(ValueError, match="ordinary real controls"):
        build_source_decomp_manifest(reference_path, controls_path, ordinary_path, tmp_path / "roundtrip")
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_manifest.py -v
```

Expected: import failure for `source_decomp_manifest`.

- [x] **Step 3: Implement manifest builder**

Create `scripts/off_prior_measurement/source_decomp_manifest.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


REQUIRED_COLUMNS = [
    "subject_id",
    "class_name",
    "image_id",
    "image_path",
    "roundtrip_image_path",
    "source_group",
    "reference_regime",
    "hardness_axis",
    "conditioning_key",
    "conditioning_prompt",
    "source_dataset",
    "source_license_note",
]


def load_ordinary_real_controls(path: str | Path) -> pd.DataFrame:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = raw.get("ordinary_real_controls", [])
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["subject_id"] = frame["class_name"].astype(str)
    frame["source_group"] = "ordinary_real_control"
    frame["reference_regime"] = "ordinary_real_control"
    frame["hardness_axis"] = "none"
    frame["conditioning_key"] = "class"
    frame["conditioning_prompt"] = "a photo of a " + frame["class_name"].astype(str)
    return frame


def _roundtrip_path(root: Path, row: dict[str, object]) -> str:
    return str(root / str(row["source_group"]) / str(row["class_name"]) / f"{row['image_id']}.png")


def _normalize_rows(frame: pd.DataFrame, source_group: str, roundtrip_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_group"] = source_group
    if "source_dataset" not in frame.columns:
        frame["source_dataset"] = source_group
    if "source_license_note" not in frame.columns:
        frame["source_license_note"] = "generated or local project artifact"
    if "roundtrip_image_path" not in frame.columns:
        frame["roundtrip_image_path"] = [
            _roundtrip_path(roundtrip_root, row) for row in frame.to_dict("records")
        ]
    return frame


def build_source_decomp_manifest(
    reference_manifest_path: str | Path,
    control_manifest_path: str | Path,
    ordinary_real_manifest_path: str | Path,
    roundtrip_root: str | Path,
) -> pd.DataFrame:
    roundtrip_root = Path(roundtrip_root)
    reference = read_csv_preserve_strings(reference_manifest_path)
    controls = read_csv_preserve_strings(control_manifest_path)
    ordinary = load_ordinary_real_controls(ordinary_real_manifest_path)
    if ordinary.empty:
        raise ValueError("Stage 1.4 requires ordinary real controls; provide ordinary_real_controls entries")
    base = controls[controls["source_group"].isin(["base_easy_control", "base_generated_control"])].copy()
    base = _normalize_rows(base, "base_generated_control", roundtrip_root)
    standard = reference[reference["source_group"] == "dreambooth_reference"].copy()
    standard = _normalize_rows(standard, "dreambooth_reference", roundtrip_root)
    ordinary = _normalize_rows(ordinary, "ordinary_real_control", roundtrip_root)
    combined = pd.concat([base, ordinary, standard], ignore_index=True)
    for column in REQUIRED_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    return combined[REQUIRED_COLUMNS]
```

- [x] **Step 4: Add CLI wrapper**

Add to the same file:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--control-manifest", required=True)
    parser.add_argument("--ordinary-real-manifest", required=True)
    parser.add_argument("--roundtrip-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_source_decomp_manifest(
        args.reference_manifest,
        args.control_manifest,
        args.ordinary_real_manifest,
        args.roundtrip_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_manifest.py -v
```

Expected: all manifest tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/source_decomp_manifest.py tests/off_prior_measurement/test_source_decomp_manifest.py
git commit -m "feat: build source decomposition manifests"
```

## Task 4: Paired Source-Decomposition Measurement

**Files:**

- Create: `scripts/off_prior_measurement/source_decomp_measure.py`
- Create: `tests/off_prior_measurement/test_source_decomp_measure.py`

- [x] **Step 1: Write failing measurement test with fake backend**

Create `tests/off_prior_measurement/test_source_decomp_measure.py`:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.off_prior_measurement.source_decomp_measure import run_source_decomp_measurement


@dataclass
class FakeBatch:
    v_ref: np.ndarray
    v_base: np.ndarray
    snr: float = 1.0


class FakeBackend:
    def measure(self, image_path, prompt, timestep, seed):
        if "roundtrip" in str(image_path):
            return FakeBatch(
                v_ref=np.array([2.0, 0.0], dtype=np.float32),
                v_base=np.array([0.0, 0.0], dtype=np.float32),
            )
        return FakeBatch(
            v_ref=np.array([2.0, 2.0], dtype=np.float32),
            v_base=np.array([0.0, 0.0], dtype=np.float32),
        )


def test_run_source_decomp_measurement_writes_projection_metrics(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "00",
                "image_path": "dog.jpg",
                "roundtrip_image_path": "roundtrip/dog.png",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
                "source_dataset": "dreambooth",
                "source_license_note": "test",
            }
        ]
    ).to_csv(manifest, index=False)

    path = run_source_decomp_measurement(
        manifest_path=manifest,
        output_dir=tmp_path / "experiment",
        timesteps=[50],
        noise_seeds=[0],
        backend=FakeBackend(),
    )

    rows = pd.read_csv(path)
    assert round(rows.iloc[0]["artifact_fraction"], 6) > 0.0
    assert round(rows.iloc[0]["clean_fraction"], 6) > 0.0
    assert rows.iloc[0]["source_group"] == "dreambooth_reference"
```

- [x] **Step 2: Run test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_measure.py -v
```

Expected: import failure for `source_decomp_measure`.

- [x] **Step 3: Implement measurement function**

Create `scripts/off_prior_measurement/source_decomp_measure.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings
from scripts.off_prior_measurement.diffusion_backend import StableDiffusionBackend
from scripts.off_prior_measurement.metrics import dct_band_energy, residual_projection_metrics


def run_source_decomp_measurement(
    manifest_path: str | Path,
    output_dir: str | Path,
    timesteps: list[int],
    noise_seeds: list[int],
    backend,
    output_name: str = "raw_source_decomp_metrics.csv",
) -> Path:
    manifest = read_csv_preserve_strings(manifest_path)
    rows = []
    for row in tqdm(list(manifest.to_dict("records")), desc="source decomposition"):
        for timestep in timesteps:
            for seed in noise_seeds:
                batch = backend.measure(row["image_path"], row["conditioning_prompt"], int(timestep), int(seed))
                rt_batch = backend.measure(
                    row["roundtrip_image_path"],
                    row["conditioning_prompt"],
                    int(timestep),
                    int(seed),
                )
                metrics = residual_projection_metrics(
                    batch.v_ref,
                    batch.v_base,
                    rt_batch.v_ref,
                    rt_batch.v_base,
                )
                clean_residual = (batch.v_ref - batch.v_base) - metrics["artifact_coeff"] * (
                    rt_batch.v_ref - rt_batch.v_base
                )
                artifact_residual = metrics["artifact_coeff"] * (rt_batch.v_ref - rt_batch.v_base)
                clean_bands = dct_band_energy(clean_residual)
                artifact_bands = dct_band_energy(artifact_residual)
                rows.append(
                    {
                        **row,
                        "timestep": int(timestep),
                        "noise_seed": int(seed),
                        "snr": float(getattr(batch, "snr", float("nan"))),
                        **metrics,
                        "dct_clean_low": clean_bands["low"],
                        "dct_clean_mid": clean_bands["mid"],
                        "dct_clean_high": clean_bands["high"],
                        "dct_artifact_low": artifact_bands["low"],
                        "dct_artifact_mid": artifact_bands["mid"],
                        "dct_artifact_high": artifact_bands["high"],
                    }
                )
    output_dir = Path(output_dir)
    measurements_dir = output_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    path = measurements_dir / output_name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
```

- [x] **Step 4: Add CLI wrapper**

Add:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backend = StableDiffusionBackend(config)
    path = run_source_decomp_measurement(
        manifest_path=args.manifest,
        output_dir=config.output_dir,
        timesteps=config.timesteps,
        noise_seeds=config.noise_seeds,
        backend=backend,
    )
    print(path)


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_measure.py -v
```

Expected: all measurement tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/source_decomp_measure.py tests/off_prior_measurement/test_source_decomp_measure.py
git commit -m "feat: measure source decomposition residuals"
```

## Task 5: Source-Gap Summaries

**Files:**

- Create: `scripts/off_prior_measurement/source_decomp_summarize.py`
- Create: `tests/off_prior_measurement/test_source_decomp_summarize.py`

- [x] **Step 1: Write failing summary test**

Create `tests/off_prior_measurement/test_source_decomp_summarize.py`:

```python
import pandas as pd

from scripts.off_prior_measurement.source_decomp_summarize import summarize_source_decomp


def test_summarize_source_decomp_computes_gap_columns(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = []
    for source, clean in [
        ("base_generated_control", 0.10),
        ("ordinary_real_control", 0.20),
        ("dreambooth_reference", 0.35),
        ("natural_hard_reference", 0.45),
    ]:
        rows.append(
            {
                "subject_id": "dog",
                "class_name": "dog",
                "source_group": source,
                "conditioning_key": "class",
                "timestep": 50,
                "clean_norm": clean,
                "raw_norm": clean + 0.05,
                "artifact_fraction": 0.20,
                "artifact_cosine": 0.30,
                "dct_clean_low": clean,
                "dct_clean_mid": clean / 2,
                "dct_clean_high": clean / 4,
            }
        )
    pd.DataFrame(rows).to_csv(raw, index=False)

    paths = summarize_source_decomp(raw, tmp_path / "experiment")

    gaps = pd.read_csv(paths["source_gap_summary"])
    row = gaps.iloc[0]
    assert round(row["real_domain_gap"], 6) == 0.10
    assert round(row["subject_specific_gap"], 6) == 0.15
    assert round(row["natural_hard_gap"], 6) == 0.10
```

- [x] **Step 2: Run test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_summarize.py -v
```

Expected: import failure for `source_decomp_summarize`.

- [x] **Step 3: Implement summaries**

Create `scripts/off_prior_measurement/source_decomp_summarize.py` with `summarize_source_decomp(raw_metrics_path, output_dir) -> dict[str, Path]` that writes:

```text
source_group_summary.csv
source_gap_summary.csv
timestep_frequency_summary.csv
```

Core grouping logic:

```python
source_summary = raw.groupby(["class_name", "conditioning_key", "source_group"], as_index=False).agg(
    mean_clean_norm=("clean_norm", "mean"),
    mean_raw_norm=("raw_norm", "mean"),
    mean_artifact_fraction=("artifact_fraction", "mean"),
    mean_artifact_cosine=("artifact_cosine", "mean"),
    n=("clean_norm", "size"),
)
wide = source_summary.pivot_table(
    index=["class_name", "conditioning_key"],
    columns="source_group",
    values="mean_clean_norm",
    aggfunc="mean",
).reset_index()
```

Then compute:

```python
wide["real_domain_gap"] = wide["ordinary_real_control"] - wide["base_generated_control"]
wide["subject_specific_gap"] = wide["dreambooth_reference"] - wide["ordinary_real_control"]
wide["natural_hard_gap"] = wide["natural_hard_reference"] - wide["dreambooth_reference"]
```

- [x] **Step 4: Add CLI wrapper, run tests, commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_summarize.py -v
```

Commit:

```bash
git add scripts/off_prior_measurement/source_decomp_summarize.py tests/off_prior_measurement/test_source_decomp_summarize.py
git commit -m "feat: summarize source decomposition gaps"
```

## Task 6: Source-Decomposition Plots And Conclusion

**Files:**

- Create: `scripts/off_prior_measurement/source_decomp_plot.py`
- Create: `scripts/off_prior_measurement/source_decomp_conclusion.py`
- Create: `tests/off_prior_measurement/test_source_decomp_plot.py`
- Create: `tests/off_prior_measurement/test_source_decomp_conclusion.py`

- [x] **Step 1: Write failing plot test**

Create a tiny `source_group_summary.csv`, `source_gap_summary.csv`, and `timestep_frequency_summary.csv`. Assert that `create_source_decomp_figures(summary_dir, figures_dir)` writes:

```text
source_gap_bars.png
artifact_fraction_by_source.png
clean_timestep_curves.png
```

- [x] **Step 2: Write failing conclusion tests**

Create two source-gap summaries:

Go case:

```text
3 of 4 class rows have subject_specific_gap > 0
mean subject_specific_gap > 0
mean artifact fraction for DreamBooth < 0.75
```

Pivot case:

```text
ordinary_real_control >= dreambooth_reference for most classes
mean artifact fraction >= 0.75
```

Assert conclusion text contains `Go / pivot decision: Go` or `Go / pivot decision: Pivot`.

- [x] **Step 3: Implement plotting**

Create `source_decomp_plot.py` using matplotlib Agg backend and `read_csv_preserve_strings`. Keep the plotting API:

```python
def create_source_decomp_figures(summary_dir: str | Path, figures_dir: str | Path) -> dict[str, Path]:
    ...
```

- [x] **Step 4: Implement conclusion**

Create `source_decomp_conclusion.py`:

```python
def write_source_decomp_conclusion(experiment_dir: str | Path) -> Path:
    ...
```

Decision rule:

```python
go = (
    positive_subject_gap_classes >= 3
    and mean_subject_specific_gap > 0
    and mean_dreambooth_artifact_fraction < 0.75
    and not_one_class_only
)
```

- [x] **Step 5: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_plot.py tests/off_prior_measurement/test_source_decomp_conclusion.py -v
```

Commit:

```bash
git add scripts/off_prior_measurement/source_decomp_plot.py scripts/off_prior_measurement/source_decomp_conclusion.py tests/off_prior_measurement/test_source_decomp_plot.py tests/off_prior_measurement/test_source_decomp_conclusion.py
git commit -m "feat: report source decomposition decision"
```

## Task 7: Documentation, Dry Run, And Backup

**Files:**

- Create: `experiments/off_prior_measurement_v0/source_decomp_v1/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md`

- [x] **Step 1: Run full lightweight test suite**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
```

Expected: all tests pass.

- [x] **Step 2: Run manifest validation dry run**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/source_decomp_v1.yaml
export EXP=experiments/off_prior_measurement_v0/source_decomp_v1
```

Then run the manifest builder only if ordinary real images exist locally:

```bash
$PYTHON -m scripts.off_prior_measurement.source_decomp_manifest \
  --reference-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/reference_manifest.csv \
  --control-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_v1.yaml \
  --roundtrip-root data/cache/off_prior_measurement_v0/source_decomp_roundtrip \
  --output $EXP/manifests/source_decomp_manifest.csv
```

Expected if local ordinary-real image paths are missing:

```text
clear error naming missing ordinary-real image paths
```

Expected if local images are present:

```text
source_decomp_manifest.csv exists and includes base_generated_control,
ordinary_real_control, dreambooth_reference
```

- [x] **Step 3: Write experiment README**

Create `experiments/off_prior_measurement_v0/source_decomp_v1/README.md` with:

```markdown
# Off-Prior Measurement V0 Source Decomposition V1

Status: implementation prepared; real smoke run requires local ordinary-real control images listed in `data/manifests/ordinary_real_controls_v1.yaml`.

Purpose: separate VAE/projection artifact, ordinary real-image domain gap, and DreamBooth subject-specific target gap before personalization fine-tuning.

Run order:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/source_decomp_v1.yaml
export EXP=experiments/off_prior_measurement_v0/source_decomp_v1

$PYTHON -m scripts.off_prior_measurement.source_decomp_manifest --reference-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/reference_manifest.csv --control-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv --ordinary-real-manifest data/manifests/ordinary_real_controls_v1.yaml --roundtrip-root data/cache/off_prior_measurement_v0/source_decomp_roundtrip --output $EXP/manifests/source_decomp_manifest.csv
$PYTHON -m scripts.off_prior_measurement.source_decomp_measure --config $CONFIG --manifest $EXP/manifests/source_decomp_manifest.csv
$PYTHON -m scripts.off_prior_measurement.source_decomp_summarize --raw-metrics $EXP/measurements/raw_source_decomp_metrics.csv --output-dir $EXP
$PYTHON -m scripts.off_prior_measurement.source_decomp_plot --summary-dir $EXP/summaries --figures-dir $EXP/figures
$PYTHON -m scripts.off_prior_measurement.source_decomp_conclusion --experiment-dir $EXP
```
```

- [x] **Step 4: Update docs with implementation status**

Update the Stage 1.4 spec status:

```text
Status: implementation prepared. Real smoke run is blocked until local ordinary-real control images are provided.
```

Update `README.md` immediate next step to mention local ordinary-real controls if the dry run is blocked.

- [x] **Step 5: Commit and backup**

Commit:

```bash
git add README.md docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md experiments/off_prior_measurement_v0/source_decomp_v1/README.md
git commit -m "docs: document source decomposition runbook"
```

Sync to GitHub with the existing GitHub API workflow. Verify:

```text
contains_token False
```

## Plan Self-Review

- Spec coverage: manifest groups, ordinary-real controls, projection metrics, source-gap summaries, figures, conclusion, and runbook all map to tasks.
- Placeholder scan: no placeholder markers remain.
- Type consistency: the plan consistently uses `raw_norm`, `clean_norm`, `artifact_fraction`, `artifact_cosine`, `real_domain_gap`, `subject_specific_gap`, and `natural_hard_gap`.
- Scope check: no personalization fine-tuning, large dataset download, or modification of completed v2/v2_clean outputs is included.
