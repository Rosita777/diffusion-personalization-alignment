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

- [ ] Add tests for suppressing only unusually large residuals.
- [ ] Add validation tests for quantile and keep values.
- [ ] Implement `apply_residual_magnitude_gate`.

### Task 2: Wire Config And Training Condition

**Files:**
- Modify: `scripts/personalization_training/config.py`
- Modify: `scripts/personalization_training/train_lora_dreambooth.py`
- Modify: `tests/personalization_training/test_training_config.py`

- [ ] Add `dadt_residual_gate` to allowed conditions.
- [ ] Parse `residual_gate_quantile` and `residual_gate_keep`.
- [ ] Route `target_for_condition` to the residual gate.
- [ ] Load the frozen base UNet for all DADT conditions.

### Task 3: Add Stage 2C-0 Vase Configs

**Files:**
- Create: `configs/stage2c0_residual_gating/vase_residual_gate_q75_keep05.yaml`
- Create: `configs/stage2c0_residual_gating/vase_residual_gate_q90_keep05.yaml`

- [ ] Add two 200-step vase configs matching Stage 2B settings.
- [ ] Validate configs and dry-run output routing.

### Task 4: Verify And Commit

**Files:**
- All files above.

- [ ] Run `pytest tests/personalization_training -q`.
- [ ] Commit only Stage 2C-0 files.
