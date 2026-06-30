# Stage 2C-1 CFG Residual Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CFG-direction residual gate that suppresses class-aligned residuals while preserving residuals less explained by the generic class prompt.

**Architecture:** Extend the existing personalization target-alignment module with one pure tensor function. Wire a new training condition through config and the existing DreamBooth LoRA loop by adding a null-prompt base prediction only for `dadt_cfg_residual_gate`.

**Tech Stack:** Python, PyTorch, YAML, pytest, existing SD1.5 LoRA DreamBooth training scripts.

---

### Task 1: Add CFG Residual Gate Tensor Logic

**Files:**
- Modify: `scripts/personalization_training/target_alignment.py`
- Modify: `tests/personalization_training/test_target_alignment.py`

- [x] Add a failing test showing positive class-direction alignment is suppressed.
- [x] Add a failing test showing zero `alpha` returns the reference target.
- [x] Add a failing test for shape mismatch between reference, class, and null predictions.
- [x] Implement `apply_cfg_residual_gate(reference_target, class_prediction, null_prediction, config)`.

Expected formula:

```python
residual = reference_target - class_prediction
class_direction = class_prediction - null_prediction
cos = cosine_channel(residual, class_direction)
gate = 1.0 - config.alpha * torch.relu(cos)
return class_prediction + gate * residual
```

### Task 2: Wire Config And Training

**Files:**
- Modify: `scripts/personalization_training/config.py`
- Modify: `scripts/personalization_training/train_lora_dreambooth.py`
- Modify: `tests/personalization_training/test_training_config.py`
- Modify: `tests/personalization_training/test_entrypoints.py`

- [x] Add `dadt_cfg_residual_gate` to allowed conditions.
- [x] Add a config-loading test for the new condition.
- [x] Update `target_for_condition` to require `null_prediction` for the CFG gate.
- [x] In training, compute `v_null = base_unet(..., null_prompt)` only for `dadt_cfg_residual_gate`.

### Task 3: Add Diagnostic Script And Config

**Files:**
- Create: `scripts/personalization_training/diagnose_cfg_residual_gate.py`
- Create: `configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha05.yaml`

- [x] Add a diagnostic script that reports cosine summary stats over reference images without training.
- [x] Add one 200-step `vase` config matching Stage 2B and Stage 2C-0 settings.
- [x] Dry-run the config and diagnostic script.

### Task 4: Verify, Run, And Record

**Files:**
- Create after run: `experiments/stage2c1_cfg_residual_gate_results.md`
- Generated local artifacts under `experiments/stage2c1_cfg_residual_gate/`

- [x] Run `pytest tests/personalization_training -q`.
- [x] Run CFG diagnostic with explicit `CUDA_VISIBLE_DEVICES=<gpu_id>`.
- [x] If diagnostic has non-degenerate cosine stats, train `vase_cfg_gate_alpha05`.
- [x] Generate eval grid and run the lightweight metric audit against Stage 2B base/vanilla and Stage 2C-0.
- [x] Record the result and commit only Stage 2C-1 files.
