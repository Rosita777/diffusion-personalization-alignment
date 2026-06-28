# Stage 2B Strong Alignment Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the Stage 2B `vase` strong-alignment sweep without overwriting Stage 2A outputs.

**Architecture:** Keep the existing Stage 2A training loop and target-alignment code. Add run-label output routing so multiple DADT variants can coexist, add Stage 2B YAML configs, then reuse the existing training/eval entrypoints to generate comparable grids.

**Tech Stack:** Python, pytest, diffusers SD 1.5, PEFT LoRA, YAML configs, local H800 GPUs.

---

### Task 1: Add Run-Label Output Routing

**Files:**
- Modify: `scripts/personalization_training/train_lora_dreambooth.py`
- Modify: `tests/personalization_training/test_entrypoints.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/personalization_training/test_entrypoints.py` near the existing output-dir tests:

```python
def test_training_output_dir_uses_run_name_when_provided(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    assert training_output_dir(config, subject_id="vase", run_name="alpha075") == (
        tmp_path / "out" / "alpha075" / "vase"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/personalization_training/test_entrypoints.py::test_training_output_dir_uses_run_name_when_provided -q
```

Expected: FAIL because `training_output_dir()` does not accept `run_name`.

- [ ] **Step 3: Write minimal implementation**

Change the signature and body in `scripts/personalization_training/train_lora_dreambooth.py`:

```python
def training_output_dir(config: Stage2AConfig, subject_id: str | None = None, run_name: str | None = None) -> Path:
    output_dir = config.output_dir / (run_name or config.training.condition)
    if subject_id:
        output_dir = output_dir / subject_id
    return output_dir
```

Add CLI argument in `main()`:

```python
parser.add_argument("--run-name", help="Output label for sweep variants; defaults to condition.")
```

Pass it into `training_output_dir()` in dry-run output and `run_training()` by adding an optional `run_name` argument:

```python
def run_training(config: Stage2AConfig, run_name: str | None = None) -> None:
    ...
    output_dir = training_output_dir(config, subject_id=_single_subject_id(config), run_name=run_name)
```

Update the CLI call:

```python
run_training(config, run_name=args.run_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/personalization_training/test_entrypoints.py::test_training_output_dir_uses_run_name_when_provided -q
```

Expected: PASS.

- [ ] **Step 5: Run the personalization test suite**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training -q
```

Expected: all tests pass.

### Task 2: Add Stage 2B Configs

**Files:**
- Create: `configs/stage2b_strong_alignment/vase_vanilla.yaml`
- Create: `configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha075.yaml`
- Create: `configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha100.yaml`
- Create: `configs/stage2b_strong_alignment/vase_dadt_lf_midlate_alpha075.yaml`

- [ ] **Step 1: Copy Stage 2A config structure**

Each config should include only the `vase` subject and the same evaluation prompts used by Stage 2A.

Use this shared shape:

```yaml
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
output_dir: experiments/stage2b_strong_alignment/vase
resolution: 512

subjects:
  - subject_id: vase
    class_name: vase
    instance_prompt: a photo of sks vase
    class_prompt: a photo of a vase
    image_paths:
      - data/cache/off_prior_measurement_v0/dreambooth_dataset/dataset/vase/00.jpg
      - data/cache/off_prior_measurement_v0/dreambooth_dataset/dataset/vase/01.jpg
      - data/cache/off_prior_measurement_v0/dreambooth_dataset/dataset/vase/02.jpg

training:
  condition: dadt_lf_late
  max_train_steps: 200
  learning_rate: 0.0001
  lora_rank: 4
  seed: 0
  train_batch_size: 1

alignment:
  alpha: 0.75
  late_timestep_threshold: 800
  low_radius: 2
  mid_radius: 4

evaluation:
  num_images_per_prompt: 2
  prompts:
    - a photo of sks vase on a wooden table
    - a photo of sks vase in a room
  class_prompts:
    - a photo of a vase
    - a vase on a wooden table
```

Variant values:

```text
vase_vanilla.yaml:
  training.condition: vanilla
  alignment.alpha: 0.0
  alignment.late_timestep_threshold: 800

vase_dadt_lf_late_alpha075.yaml:
  training.condition: dadt_lf_late
  alignment.alpha: 0.75
  alignment.late_timestep_threshold: 800

vase_dadt_lf_late_alpha100.yaml:
  training.condition: dadt_lf_late
  alignment.alpha: 1.0
  alignment.late_timestep_threshold: 800

vase_dadt_lf_midlate_alpha075.yaml:
  training.condition: dadt_lf_late
  alignment.alpha: 0.75
  alignment.late_timestep_threshold: 500
```

- [ ] **Step 2: Validate configs**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
from pathlib import Path
from scripts.personalization_training.config import load_training_config
for path in sorted(Path("configs/stage2b_strong_alignment").glob("*.yaml")):
    cfg = load_training_config(path)
    print(path.name, cfg.training.condition, cfg.alignment.alpha, cfg.alignment.late_timestep_threshold)
PY
```

Expected: all four configs load and print the intended condition/alpha/threshold.

### Task 3: Ignore Stage 2B Generated Artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add ignore rule**

Add:

```text
experiments/stage2b_strong_alignment/
```

The Stage 2B result note can be committed later as a small Markdown file if needed; raw images, logs, summaries, and LoRA weights should stay local.

- [ ] **Step 2: Verify generated files stay untracked**

Run:

```bash
git status --short
```

Expected after runs: `experiments/stage2b_strong_alignment/` files should not appear as untracked.

### Task 4: Run Stage 2B Training Sweep

**Files:**
- Read: `configs/stage2b_strong_alignment/*.yaml`
- Generated local artifacts: `experiments/stage2b_strong_alignment/vase/<run_name>/vase/`

- [ ] **Step 1: Launch four training jobs**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2b_strong_alignment/vase_vanilla.yaml \
  --subject-id vase \
  --run-name vanilla \
  > experiments/stage2b_vase_vanilla.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha075.yaml \
  --subject-id vase \
  --run-name dadt_lf_late_alpha075 \
  > experiments/stage2b_vase_dadt_lf_late_alpha075.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha100.yaml \
  --subject-id vase \
  --run-name dadt_lf_late_alpha100 \
  > experiments/stage2b_vase_dadt_lf_late_alpha100.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_midlate_alpha075.yaml \
  --subject-id vase \
  --run-name dadt_lf_midlate_alpha075 \
  > experiments/stage2b_vase_dadt_lf_midlate_alpha075.log 2>&1 &
wait
```

If the machine is busy, use fewer GPUs and run the same commands sequentially.

- [ ] **Step 2: Check summaries**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("experiments/stage2b_strong_alignment/vase")
for run in ["vanilla", "dadt_lf_late_alpha075", "dadt_lf_late_alpha100", "dadt_lf_midlate_alpha075"]:
    path = root / run / "vase" / "training_summary.json"
    data = json.loads(path.read_text())
    print(run, data["loss_first"], data["loss_last"], data["loss_mean"])
PY
```

Expected: all summaries exist and losses are finite.

### Task 5: Generate Stage 2B Evaluation Grids

**Files:**
- Generated local artifacts: `experiments/stage2b_strong_alignment/vase/eval/<run_name>/`

- [ ] **Step 1: Generate base grid**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2b_strong_alignment/vase_vanilla.yaml \
  --subject-id vase \
  --output-dir experiments/stage2b_strong_alignment/vase/eval/base
```

- [ ] **Step 2: Generate trained grids**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2b_strong_alignment/vase_vanilla.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2b_strong_alignment/vase/vanilla/vase \
  --output-dir experiments/stage2b_strong_alignment/vase/eval/vanilla

CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha075.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2b_strong_alignment/vase/dadt_lf_late_alpha075/vase \
  --output-dir experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha075

CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha100.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2b_strong_alignment/vase/dadt_lf_late_alpha100/vase \
  --output-dir experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha100

CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2b_strong_alignment/vase_dadt_lf_midlate_alpha075.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2b_strong_alignment/vase/dadt_lf_midlate_alpha075/vase \
  --output-dir experiments/stage2b_strong_alignment/vase/eval/dadt_lf_midlate_alpha075
```

Expected: each run writes 8 images, `grid.png`, and `manifest.json`.

### Task 6: Record Stage 2B Results

**Files:**
- Create: `experiments/stage2b_strong_alignment/README.md` if results are small enough to track, or update `experiments/stage2a_lf_late/README.md` with a pointer if generated artifacts are ignored.

- [ ] **Step 1: Write concise result note**

Record:

```text
date
configs
commands
loss summaries
grid paths
qualitative observation
Go / Redesign / No-Go decision
```

- [ ] **Step 2: Run final tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit code, configs, docs**

Stage only Stage 2B files and relevant code changes:

```bash
git add \
  configs/stage2b_strong_alignment \
  scripts/personalization_training/train_lora_dreambooth.py \
  tests/personalization_training/test_entrypoints.py \
  docs/superpowers/plans/2026-06-28-stage-2b-strong-alignment-sweep.md \
  experiments/stage2b_strong_alignment/README.md
```

Stage `.gitignore` carefully so unrelated video-branch ignore rules are not included unless already intentional.

Commit:

```bash
git commit -m "feat: run stage 2b alignment sweep"
```

Expected: commit contains Stage 2B code/config/docs only; generated weights and images are ignored.
