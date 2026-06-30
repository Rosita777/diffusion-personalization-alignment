# Stage 2C-0 Residual Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal residual-magnitude-gated DADT training condition for SD1.5 LoRA DreamBooth.

**Architecture:** Extend the existing personalization config and target-alignment module. Keep the training loop unchanged except for loading the frozen base UNet for the new condition and routing the condition to the new target constructor.

**Tech Stack:** Python, PyTorch, YAML, pytest, existing LoRA DreamBooth training entrypoint.

---

### Task 1: Add Residual Gate Target

**Files:**
- Modify: `scripts/personalization_training/target_alignment.py`
- Modify: `tests/personalization_training/test_target_alignment.py`

- [x] Add tests for suppressing only unusually large residuals.
- [x] Add validation tests for quantile and keep values.
- [x] Implement `apply_residual_magnitude_gate`.

### Task 2: Wire Config And Training Condition

**Files:**
- Modify: `scripts/personalization_training/config.py`
- Modify: `scripts/personalization_training/train_lora_dreambooth.py`
- Modify: `tests/personalization_training/test_training_config.py`

- [x] Add `dadt_residual_gate` to allowed conditions.
- [x] Parse `residual_gate_quantile` and `residual_gate_keep`.
- [x] Route `target_for_condition` to the residual gate.
- [x] Load the frozen base UNet for all DADT conditions.

### Task 3: Add Stage 2C-0 Vase Configs

**Files:**
- Create: `configs/stage2c0_residual_gating/vase_residual_gate_q75_keep05.yaml`
- Create: `configs/stage2c0_residual_gating/vase_residual_gate_q90_keep05.yaml`

- [x] Add two 200-step vase configs matching Stage 2B settings.
- [x] Validate configs and dry-run output routing.

### Task 4: Verify And Commit

**Files:**
- All files above.

- [x] Run `pytest tests/personalization_training -q`.
- [x] Commit only Stage 2C-0 files.

### Task 5: Run Vase Training And Evaluation

**Files:**
- Generated local artifacts under `experiments/stage2c0_residual_gating/`
- Generated metric CSVs:
  - `experiments/stage2c0_residual_gate_metric_audit_summary.csv`
  - `experiments/stage2c0_residual_gate_metric_audit_per_image.csv`

- [x] Diagnose GPU visibility and use explicit `CUDA_VISIBLE_DEVICES=<gpu_id>`.
- [x] Train `residual_gate_q75_keep05`.
- [x] Train `residual_gate_q90_keep05`.
- [x] Generate eval grids for both runs.
- [x] Run the lightweight metric audit against Stage 2B base and vanilla.

Commands used:

```bash
CUDA_VISIBLE_DEVICES=3 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2c0_residual_gating/vase_residual_gate_q75_keep05.yaml \
  --subject-id vase \
  --run-name residual_gate_q75_keep05

CUDA_VISIBLE_DEVICES=7 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2c0_residual_gating/vase_residual_gate_q90_keep05.yaml \
  --subject-id vase \
  --run-name residual_gate_q90_keep05

CUDA_VISIBLE_DEVICES=3 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2c0_residual_gating/vase_residual_gate_q75_keep05.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2c0_residual_gating/vase/residual_gate_q75_keep05/vase \
  --output-dir experiments/stage2c0_residual_gating/vase/eval/residual_gate_q75_keep05

CUDA_VISIBLE_DEVICES=7 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2c0_residual_gating/vase_residual_gate_q90_keep05.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2c0_residual_gating/vase/residual_gate_q90_keep05/vase \
  --output-dir experiments/stage2c0_residual_gating/vase/eval/residual_gate_q90_keep05
```

Result:

```text
The GPU issue was not a hardware failure. PyTorch saw zero devices without an explicit
CUDA_VISIBLE_DEVICES setting, but initialized correctly when a GPU id was specified.

Residual gating preserves the base prior more strongly than vanilla, but currently also
weakens subject learning. Treat Stage 2C-0 as a useful direction check, not a win.
```
