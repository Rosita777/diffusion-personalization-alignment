# Stage 2A LF-Late Training Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline TDD execution in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal LoRA DreamBooth smoke framework that can compare vanilla personalization against LF-Late target-aligned personalization.

**Architecture:** Keep target correction independent from training. `dct_target.py` owns latent DCT decomposition, `target_alignment.py` owns the LF-Late target edit, `config.py` validates small YAML configs, and `train_lora_dreambooth.py` wires those pieces into a diffusers LoRA training loop. Evaluation image generation and reporting stay in separate scripts so training can be tested without generating images.

**Tech Stack:** Python, PyTorch, pandas/YAML, diffusers, pytest. Unit tests must not load Stable Diffusion.

---

### Task 1: DCT Target Utilities

**Files:**
- Create: `scripts/personalization_training/__init__.py`
- Create: `scripts/personalization_training/dct_target.py`
- Test: `tests/personalization_training/test_dct_target.py`

- [x] **Step 1: Write failing tests**

```python
import torch

from scripts.personalization_training.dct_target import (
    dct2,
    frequency_masks,
    idct2,
    split_frequency_bands,
)


def test_dct_roundtrip_reconstructs_tensor():
    x = torch.randn(2, 3, 8, 8)
    reconstructed = idct2(dct2(x))
    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=1e-5)


def test_frequency_masks_are_disjoint_and_complete():
    masks = frequency_masks(height=8, width=8, low_radius=2, mid_radius=4, device=torch.device("cpu"))
    total = masks.low.to(torch.int) + masks.mid.to(torch.int) + masks.high.to(torch.int)
    assert torch.equal(total, torch.ones_like(total))
    assert masks.low.sum().item() > 0
    assert masks.mid.sum().item() > 0
    assert masks.high.sum().item() > 0


def test_split_frequency_bands_reconstructs_input():
    x = torch.randn(1, 4, 8, 8)
    bands = split_frequency_bands(x, low_radius=2, mid_radius=4)
    reconstructed = bands.low + bands.mid + bands.high
    assert torch.allclose(reconstructed, x, atol=1e-5)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training/test_dct_target.py -v
```

Expected: fail because `scripts.personalization_training.dct_target` does not exist.

- [x] **Step 3: Implement DCT utilities**

Implement:

```python
@dataclass(frozen=True)
class FrequencyMasks:
    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


@dataclass(frozen=True)
class FrequencyBands:
    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


def dct2(x: torch.Tensor) -> torch.Tensor: ...
def idct2(x: torch.Tensor) -> torch.Tensor: ...
def frequency_masks(height: int, width: int, low_radius: int, mid_radius: int, device: torch.device | str | None = None) -> FrequencyMasks: ...
def split_frequency_bands(x: torch.Tensor, low_radius: int, mid_radius: int) -> FrequencyBands: ...
```

Use orthonormal matrix DCT so tests are deterministic and do not need SciPy.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all tests pass.

### Task 2: LF-Late Target Alignment

**Files:**
- Create: `scripts/personalization_training/target_alignment.py`
- Test: `tests/personalization_training/test_target_alignment.py`

- [x] **Step 1: Write failing tests**

```python
import torch

from scripts.personalization_training.target_alignment import (
    LFLateAlignmentConfig,
    apply_lf_late_alignment,
)


def test_alignment_alpha_zero_returns_reference_target():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=0.0, late_timestep_threshold=800, low_radius=2, mid_radius=4)
    aligned = apply_lf_late_alignment(ref, base, timestep=900, config=cfg)
    assert torch.allclose(aligned, ref)


def test_alignment_skips_early_timestep():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=0.5, late_timestep_threshold=800, low_radius=2, mid_radius=4)
    aligned = apply_lf_late_alignment(ref, base, timestep=200, config=cfg)
    assert torch.allclose(aligned, ref)


def test_alignment_replaces_only_low_frequency_at_alpha_one():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=1.0, late_timestep_threshold=800, low_radius=2, mid_radius=4)
    aligned = apply_lf_late_alignment(ref, base, timestep=900, config=cfg)
    ref_bands = cfg.split(ref)
    base_bands = cfg.split(base)
    aligned_bands = cfg.split(aligned)
    assert torch.allclose(aligned_bands.low, base_bands.low, atol=1e-5)
    assert torch.allclose(aligned_bands.mid, ref_bands.mid, atol=1e-5)
    assert torch.allclose(aligned_bands.high, ref_bands.high, atol=1e-5)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training/test_target_alignment.py -v
```

Expected: fail because `target_alignment.py` does not exist.

- [x] **Step 3: Implement LF-Late alignment**

Implement:

```python
@dataclass(frozen=True)
class LFLateAlignmentConfig:
    alpha: float
    late_timestep_threshold: int
    low_radius: int
    mid_radius: int

    def split(self, x: torch.Tensor) -> FrequencyBands: ...


def apply_lf_late_alignment(reference_target: torch.Tensor, base_prediction: torch.Tensor, timestep: int | torch.Tensor, config: LFLateAlignmentConfig) -> torch.Tensor: ...
```

Validate alpha in `[0, 1]` and radii as positive ordered integers.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all tests pass.

### Task 3: Stage 2A Config Parsing

**Files:**
- Create: `scripts/personalization_training/config.py`
- Create: `configs/stage2a_lf_late/smoke_vase_dog.yaml`
- Test: `tests/personalization_training/test_training_config.py`

- [x] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest
import yaml

from scripts.personalization_training.config import load_training_config


def test_load_training_config_rejects_invalid_alpha(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"fake")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "model_id": "runwayml/stable-diffusion-v1-5",
        "output_dir": str(tmp_path / "out"),
        "subjects": [{"subject_id": "vase", "class_name": "vase", "instance_prompt": "a photo of sks vase", "class_prompt": "a photo of a vase", "image_paths": [str(image)]}],
        "training": {"condition": "dadt_lf_late", "max_train_steps": 1, "learning_rate": 1e-4, "lora_rank": 4, "seed": 0},
        "alignment": {"alpha": 1.5, "late_timestep_threshold": 800, "low_radius": 2, "mid_radius": 4},
    }))
    with pytest.raises(ValueError, match="alpha"):
        load_training_config(path)


def test_load_training_config_rejects_missing_subject_image(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "model_id": "runwayml/stable-diffusion-v1-5",
        "output_dir": str(tmp_path / "out"),
        "subjects": [{"subject_id": "dog", "class_name": "dog", "instance_prompt": "a photo of sks dog", "class_prompt": "a photo of a dog", "image_paths": [str(tmp_path / "missing.png")]}],
        "training": {"condition": "vanilla", "max_train_steps": 1, "learning_rate": 1e-4, "lora_rank": 4, "seed": 0},
        "alignment": {"alpha": 0.5, "late_timestep_threshold": 800, "low_radius": 2, "mid_radius": 4},
    }))
    with pytest.raises(FileNotFoundError):
        load_training_config(path)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training/test_training_config.py -v
```

Expected: fail because `config.py` does not exist.

- [x] **Step 3: Implement config parser and smoke config**

Implement dataclasses for `SubjectConfig`, `TrainingConfig`, `Stage2AConfig`, and load from YAML. The smoke YAML should point at local DreamBooth cache paths already used by Stage 1 where possible.

- [x] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all tests pass.

### Task 4: Training And Evaluation Entrypoints

**Files:**
- Create: `scripts/personalization_training/train_lora_dreambooth.py`
- Create: `scripts/personalization_training/generate_eval_grid.py`
- Create: `scripts/personalization_training/write_stage2a_report.py`
- Test: `tests/personalization_training/test_entrypoints.py`

- [x] **Step 1: Write failing tests**

Test that each CLI exposes `--help` without importing heavy model classes at import time.

- [x] **Step 2: Implement lightweight entrypoint skeletons**

Training script should:

- load config;
- lazily import diffusers/torch heavy classes only inside `main`;
- support `--dry-run` to print planned subjects, condition, and output paths without loading SD;
- wire vanilla and DADT target paths through a small function that unit tests can import.

Evaluation/report scripts should accept config and output paths and create directories in dry-run mode.

- [x] **Step 3: Run tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training -v
```

Expected: all tests pass.

### Task 5: Documentation And Smoke Commands

**Files:**
- Modify: `README.md`
- Create: `experiments/stage2a_lf_late/README.md`

- [x] **Step 1: Document current Stage 2A status**

Record that Stage 2A code is implemented, unit-tested, and ready for a GPU dry run. Include commands:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition vanilla \
  --dry-run

/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition dadt_lf_late \
  --dry-run
```

- [x] **Step 2: Run full relevant tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training tests/off_prior_measurement -v
```

- [x] **Step 3: Commit Stage 2A code only**

Stage only Stage 2A files. Do not stage `token.txt`, model checkpoints, generated images, raw caches, or unrelated video-branch changes.
