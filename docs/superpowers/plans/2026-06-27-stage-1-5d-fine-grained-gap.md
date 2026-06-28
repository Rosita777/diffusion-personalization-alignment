# Stage 1.5D Fine-Grained Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline task execution because only local tools are available in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible offline analysis that checks whether DreamBooth-specific target-gap signal is hidden in timestep or frequency axes.

**Architecture:** Add one small analysis module that reads raw source-decomposition metrics and writes summary CSVs. Keep it separate from the existing Stage 1.5C metric-ablation script because this stage answers a different question: where a signal may be hiding, not how raw/clean/artifact scalar metrics compare.

**Tech Stack:** Python, pandas, pytest, existing `read_csv_preserve_strings` helper.

---

### Task 1: Fine-Grained Analysis Tests

**Files:**
- Create: `tests/off_prior_measurement/test_source_decomp_fine_grained.py`

- [x] **Step 1: Write failing tests**

Create tests for:

```text
1. timestep gap computation preserves timestep and computes real-domain and subject-specific gaps.
2. frequency candidate ranking turns dct_clean_low/mid/high into a long table and sorts positive DreamBooth pockets first.
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_fine_grained.py -v
```

Expected: fail because `scripts.off_prior_measurement.source_decomp_fine_grained` does not exist yet.

### Task 2: Fine-Grained Analysis Implementation

**Files:**
- Create: `scripts/off_prior_measurement/source_decomp_fine_grained.py`

- [x] **Step 1: Implement summary generation**

Add `summarize_fine_grained(raw_metrics_path, output_dir, label)` that writes:

```text
source_timestep_summary_<label>.csv
gap_by_timestep_<label>.csv
frequency_gap_summary_<label>.csv
signal_candidates_<label>.csv
```

- [x] **Step 2: Run tests to verify green**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_fine_grained.py -v
```

Expected: all tests pass.

### Task 3: Stage 1.5D Execution And Documentation

**Files:**
- Create directory: `experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/`
- Create: `experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md`

- [x] **Step 1: Run Stage 1.5D on Stage 1.5A and Stage 1.5B raw metrics**

Run the new script twice, once with `stage15a_class` and once with `stage15b_prompt_matched`.

- [x] **Step 2: Write the experiment README**

Record commands, key tables, plain-language interpretation, and decision.

- [x] **Step 3: Run full off-prior tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
```

- [x] **Step 4: Commit Stage 1.5D only**

Commit only the Stage 1.5D script, tests, result summaries, and DADT documentation updates. Do not stage local secrets, raw data caches, or unrelated video-branch files.
