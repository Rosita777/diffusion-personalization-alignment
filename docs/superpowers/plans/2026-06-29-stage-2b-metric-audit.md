# Stage 2B Metric Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a lightweight metric audit over existing Stage 2B evaluation images.

**Architecture:** Add one dependency-light script that reads eval `manifest.json` files, matches images by prompt/seed metadata, computes simple RGB-distance proxies, and writes small CSV summaries. Keep training and generation code unchanged.

**Tech Stack:** Python, Pillow, CSV, pytest, existing Stage 2B eval manifests.

---

### Task 1: Add Metric Audit Tests

**Files:**
- Create: `tests/personalization_training/test_audit_eval_metrics.py`

- [ ] **Step 1: Write tiny image fixtures**

Create helper functions that write RGB PNGs and manifests under a pytest `tmp_path`.

- [ ] **Step 2: Test matched image drift**

Assert that a run image with RGB value `(20, 20, 20)` compared to base `(10, 10, 10)` produces `10.0` mean absolute RGB drift.

- [ ] **Step 3: Test summary grouping**

Assert that subject and class rows are summarized separately and that a DADT run can be compared against both base and vanilla manifests.

- [ ] **Step 4: Run the tests and verify they fail**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training/test_audit_eval_metrics.py -q
```

Expected: fail because `scripts.personalization_training.audit_eval_metrics` does not exist yet.

### Task 2: Implement Metric Audit Script

**Files:**
- Create: `scripts/personalization_training/audit_eval_metrics.py`

- [ ] **Step 1: Add manifest loading**

Implement a small `EvalRecord` dataclass and `load_manifest(eval_dir)` that loads `manifest.json` and resolves image paths.

- [ ] **Step 2: Add matched-key comparison**

Match records by:

```text
kind, text, prompt_index, image_index, seed
```

Raise a clear error if a run record has no matching base record.

- [ ] **Step 3: Add image distance and diversity helpers**

Use Pillow only:

```text
ImageChops.difference(image_a, image_b)
ImageStat.Stat(diff).mean
```

Compute mean over RGB channel means.

- [ ] **Step 4: Add CSV writers and CLI**

Expose:

```bash
python -m scripts.personalization_training.audit_eval_metrics \
  --base-dir <dir> \
  --run-dir name=path \
  --vanilla-dir <dir> \
  --output-summary <csv> \
  --output-per-image <csv>
```

- [ ] **Step 5: Run tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training/test_audit_eval_metrics.py -q
```

Expected: pass.

### Task 3: Run Stage 2B Audit

**Files:**
- Generated: `experiments/stage2b_metric_audit_summary.csv`
- Generated: `experiments/stage2b_metric_audit_per_image.csv`
- Modify: `experiments/stage2b_strong_alignment_results.md`

- [ ] **Step 1: Run the audit on existing Stage 2B images**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.audit_eval_metrics \
  --base-dir experiments/stage2b_strong_alignment/vase/eval/base \
  --vanilla-dir experiments/stage2b_strong_alignment/vase/eval/vanilla \
  --run-dir vanilla=experiments/stage2b_strong_alignment/vase/eval/vanilla \
  --run-dir dadt_lf_late_alpha075=experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha075 \
  --run-dir dadt_lf_late_alpha100=experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha100 \
  --run-dir dadt_lf_midlate_alpha075=experiments/stage2b_strong_alignment/vase/eval/dadt_lf_midlate_alpha075 \
  --output-summary experiments/stage2b_metric_audit_summary.csv \
  --output-per-image experiments/stage2b_metric_audit_per_image.csv
```

- [ ] **Step 2: Update the Stage 2B result note**

Add a short metric-audit section with the summary table and decision.

- [ ] **Step 3: Run focused and full personalization tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/personalization_training -q
```

Expected: pass.

### Task 4: Commit Relevant Files

**Files:**
- Add: `docs/superpowers/specs/2026-06-29-stage-2b-metric-audit-design.md`
- Add: `docs/superpowers/plans/2026-06-29-stage-2b-metric-audit.md`
- Add: `scripts/personalization_training/audit_eval_metrics.py`
- Add: `tests/personalization_training/test_audit_eval_metrics.py`
- Add: `experiments/stage2b_metric_audit_summary.csv`
- Add: `experiments/stage2b_metric_audit_per_image.csv`
- Modify: `experiments/stage2b_strong_alignment_results.md`

- [ ] **Step 1: Inspect status**

Run:

```bash
git status --short
```

Only stage Stage 2B metric-audit files. Leave unrelated video/README/requirements changes untouched.

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-06-29-stage-2b-metric-audit-design.md \
  docs/superpowers/plans/2026-06-29-stage-2b-metric-audit.md \
  scripts/personalization_training/audit_eval_metrics.py \
  tests/personalization_training/test_audit_eval_metrics.py \
  experiments/stage2b_metric_audit_summary.csv \
  experiments/stage2b_metric_audit_per_image.csv \
  experiments/stage2b_strong_alignment_results.md
git commit -m "feat: audit stage 2b eval metrics"
```
