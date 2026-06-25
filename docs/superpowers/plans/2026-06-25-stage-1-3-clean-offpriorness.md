# Stage 1.3 Clean Off-Priorness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only Stage 1.3 analysis that subtracts VAE roundtrip artifacts from the completed ladder v2 scored metrics and decides whether clean off-priorness is usable for Stage 2.

**Architecture:** Add three small analysis modules beside the existing Stage 1 scripts: `clean_offprior.py` computes raw/clean tables, `plot_clean_offprior.py` renders diagnostic figures, and `write_clean_conclusion.py` writes the Go / No-Go decision. The completed v2 output remains immutable; Stage 1.3 writes a separate `experiments/off_prior_measurement_v0/ladder_v2_clean/` directory with source-experiment metadata.

**Tech Stack:** Python 3.10 in `/home/deepseek_VG/.conda/envs/dyme`, pandas, NumPy, matplotlib, pytest, existing `scripts.off_prior_measurement.csv_io.read_csv_preserve_strings`.

**Implementation Status:** Implemented and run on 2026-06-25. The real `ladder_v2_clean` analysis produced a No-Go: clean standard-reference positive subjects 0/8, mean clean standard-reference residual -0.0067, and standard roundtrip attribution ratio 1.1502.

---

## Scope

Implement only post-hoc analysis on existing `ladder_v2` outputs. Do not run GPU measurement, download data, or alter the completed v2 experiment directory.

## File Structure

Create:

```text
scripts/off_prior_measurement/clean_offprior.py
scripts/off_prior_measurement/plot_clean_offprior.py
scripts/off_prior_measurement/write_clean_conclusion.py
tests/off_prior_measurement/test_clean_offprior.py
tests/off_prior_measurement/test_plot_clean_offprior.py
tests/off_prior_measurement/test_write_clean_conclusion.py
experiments/off_prior_measurement_v0/ladder_v2_clean/README.md
```

Modify:

```text
README.md
docs/superpowers/specs/2026-06-23-roundtrip-confound-clean-offpriorness-design.md
```

Generated and commit-worthy after the run:

```text
experiments/off_prior_measurement_v0/ladder_v2_clean/config_resolved.yaml
experiments/off_prior_measurement_v0/ladder_v2_clean/conclusion.md
experiments/off_prior_measurement_v0/ladder_v2_clean/summaries/clean_scored_metrics.csv
experiments/off_prior_measurement_v0/ladder_v2_clean/summaries/clean_subject_summary.csv
experiments/off_prior_measurement_v0/ladder_v2_clean/summaries/clean_regime_summary.csv
experiments/off_prior_measurement_v0/ladder_v2_clean/summaries/clean_ladder_summary.csv
experiments/off_prior_measurement_v0/ladder_v2_clean/figures/raw_vs_clean_ladder.png
experiments/off_prior_measurement_v0/ladder_v2_clean/figures/roundtrip_attribution_by_subject.png
experiments/off_prior_measurement_v0/ladder_v2_clean/figures/clean_timestep_curves.png
experiments/off_prior_measurement_v0/ladder_v2_clean/figures/clean_frequency_heatmap.png
```

Generated but not commit-worthy:

```text
experiments/off_prior_measurement_v0/ladder_v2_clean/*.log
```

## Task 1: Clean Off-Priorness Tables

**Files:**

- Create: `tests/off_prior_measurement/test_clean_offprior.py`
- Create: `scripts/off_prior_measurement/clean_offprior.py`

- [ ] **Step 1: Write failing pairing and clean-score tests**

Create `tests/off_prior_measurement/test_clean_offprior.py` with tests that build a tiny scored table containing:

```python
rows = [
    {
        "subject_id": "dog",
        "image_id": "00",
        "source_group": "dreambooth_reference",
        "reference_regime": "standard_reference",
        "hardness_axis": "none",
        "source_standard_image": "",
        "variant_id": "",
        "conditioning_key": "class",
        "timestep": 50,
        "noise_seed": 0,
        "floor_adjusted_l2": 0.30,
        "dct_delta_low": 3.0,
        "dct_delta_mid": 2.0,
        "dct_delta_high": 1.0,
    },
    {
        "subject_id": "dog",
        "image_id": "00",
        "source_group": "vae_roundtrip_control",
        "reference_regime": "roundtrip_control",
        "hardness_axis": "none",
        "source_standard_image": "00",
        "variant_id": "",
        "conditioning_key": "class",
        "timestep": 50,
        "noise_seed": 0,
        "floor_adjusted_l2": 0.10,
        "dct_delta_low": 1.0,
        "dct_delta_mid": 1.0,
        "dct_delta_high": 1.0,
    },
    {
        "subject_id": "dog",
        "image_id": "00__crop_large_subject",
        "source_group": "dreambooth_hard_reference",
        "reference_regime": "hard_reference",
        "hardness_axis": "crop",
        "source_standard_image": "00",
        "variant_id": "crop_large_subject",
        "conditioning_key": "class",
        "timestep": 50,
        "noise_seed": 0,
        "floor_adjusted_l2": 0.45,
        "dct_delta_low": 4.0,
        "dct_delta_mid": 2.0,
        "dct_delta_high": 1.0,
    },
]
```

Assertions:

```python
paths = compute_clean_offprior(scored_path, output_dir, source_experiment="experiments/off_prior_measurement_v0/ladder_v2")
clean = pd.read_csv(paths["clean_scored_metrics"], keep_default_na=False)
standard = clean[clean["source_group"] == "dreambooth_reference"].iloc[0]
hard = clean[clean["source_group"] == "dreambooth_hard_reference"].iloc[0]
assert standard["image_id"] == "00"
assert round(standard["roundtrip_baseline_l2"], 6) == 0.10
assert round(standard["clean_pair_l2"], 6) == 0.20
assert round(hard["clean_pair_l2"], 6) == 0.35
assert paths["clean_ladder_summary"].exists()
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_clean_offprior.py -v
```

Expected: import failure for `scripts.off_prior_measurement.clean_offprior`.

- [ ] **Step 3: Implement clean table generation**

Create `scripts/off_prior_measurement/clean_offprior.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


PAIR_KEYS = ["subject_id", "conditioning_key", "timestep", "noise_seed"]


def _standard_image_key(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("source_standard_image", pd.Series([""] * len(frame), index=frame.index)).fillna("")
    image = frame.get("image_id", pd.Series([""] * len(frame), index=frame.index)).fillna("")
    return source.where(source.astype(str) != "", image).astype(str)


def _roundtrip_baseline(scored: pd.DataFrame) -> pd.DataFrame:
    roundtrip = scored[scored["source_group"] == "vae_roundtrip_control"].copy()
    roundtrip["standard_image_key"] = _standard_image_key(roundtrip)
    return roundtrip[PAIR_KEYS + ["standard_image_key", "floor_adjusted_l2"]].rename(
        columns={"floor_adjusted_l2": "roundtrip_baseline_l2"}
    )


def compute_clean_scores(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    for column in ["source_standard_image", "variant_id", "hardness_axis"]:
        if column not in scored.columns:
            scored[column] = ""
    scored["standard_image_key"] = _standard_image_key(scored)
    baseline = _roundtrip_baseline(scored)
    clean = scored.merge(baseline, on=PAIR_KEYS + ["standard_image_key"], how="left")
    subject_baseline = (
        scored[scored["source_group"] == "vae_roundtrip_control"]
        .groupby(PAIR_KEYS, as_index=False)["floor_adjusted_l2"]
        .median()
        .rename(columns={"floor_adjusted_l2": "subject_roundtrip_baseline_l2"})
    )
    clean = clean.merge(subject_baseline, on=PAIR_KEYS, how="left")
    clean["roundtrip_baseline_l2"] = clean["roundtrip_baseline_l2"].fillna(
        clean["subject_roundtrip_baseline_l2"]
    )
    clean["clean_pair_l2"] = clean["floor_adjusted_l2"] - clean["roundtrip_baseline_l2"]
    clean["clean_subject_l2"] = clean["floor_adjusted_l2"] - clean["subject_roundtrip_baseline_l2"]
    eps = 1e-8
    clean["roundtrip_ratio"] = clean["roundtrip_baseline_l2"].abs() / (
        clean["floor_adjusted_l2"].abs() + eps
    )
    return clean


def _summaries(clean: pd.DataFrame, summaries_dir: Path) -> dict[str, Path]:
    regime_summary = (
        clean.groupby(["source_group", "reference_regime", "hardness_axis", "conditioning_key", "timestep"], as_index=False)
        .agg(
            mean_raw_l2=("floor_adjusted_l2", "mean"),
            mean_clean_pair_l2=("clean_pair_l2", "mean"),
            mean_clean_subject_l2=("clean_subject_l2", "mean"),
            mean_roundtrip_ratio=("roundtrip_ratio", "mean"),
            n=("floor_adjusted_l2", "size"),
        )
    )
    subject_summary = (
        clean.groupby(["subject_id", "source_group", "reference_regime", "conditioning_key"], as_index=False)
        .agg(
            mean_raw_l2=("floor_adjusted_l2", "mean"),
            mean_clean_pair_l2=("clean_pair_l2", "mean"),
            mean_clean_subject_l2=("clean_subject_l2", "mean"),
            mean_roundtrip_ratio=("roundtrip_ratio", "mean"),
            n=("floor_adjusted_l2", "size"),
        )
    )
    ladder_base = (
        clean.groupby(["subject_id", "conditioning_key", "reference_regime"], as_index=False)
        .agg(mean_raw_l2=("floor_adjusted_l2", "mean"), mean_clean_pair_l2=("clean_pair_l2", "mean"))
    )
    raw_wide = ladder_base.pivot_table(index=["subject_id", "conditioning_key"], columns="reference_regime", values="mean_raw_l2", aggfunc="mean").add_prefix("raw_")
    clean_wide = ladder_base.pivot_table(index=["subject_id", "conditioning_key"], columns="reference_regime", values="mean_clean_pair_l2", aggfunc="mean").add_prefix("clean_")
    ladder = pd.concat([raw_wide, clean_wide], axis=1).reset_index()
    for column in ["raw_easy_control", "raw_standard_reference", "raw_hard_reference", "raw_roundtrip_control", "clean_standard_reference", "clean_hard_reference"]:
        if column not in ladder.columns:
            ladder[column] = np.nan
    ladder["raw_standard_minus_easy"] = ladder["raw_standard_reference"] - ladder["raw_easy_control"]
    ladder["raw_hard_minus_standard"] = ladder["raw_hard_reference"] - ladder["raw_standard_reference"]
    ladder["clean_hard_minus_standard"] = ladder["clean_hard_reference"] - ladder["clean_standard_reference"]
    paths = {
        "clean_regime_summary": summaries_dir / "clean_regime_summary.csv",
        "clean_subject_summary": summaries_dir / "clean_subject_summary.csv",
        "clean_ladder_summary": summaries_dir / "clean_ladder_summary.csv",
    }
    regime_summary.to_csv(paths["clean_regime_summary"], index=False)
    subject_summary.to_csv(paths["clean_subject_summary"], index=False)
    ladder.to_csv(paths["clean_ladder_summary"], index=False)
    return paths


def compute_clean_offprior(scored_metrics_path: str | Path, output_dir: str | Path, source_experiment: str) -> dict[str, Path]:
    scored = read_csv_preserve_strings(scored_metrics_path)
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    clean = compute_clean_scores(scored)
    clean_path = summaries_dir / "clean_scored_metrics.csv"
    clean.to_csv(clean_path, index=False)
    paths = {"clean_scored_metrics": clean_path}
    paths.update(_summaries(clean, summaries_dir))
    config_path = output_dir / "config_resolved.yaml"
    config_path.write_text(yaml.safe_dump({"source_experiment": source_experiment, "scored_metrics": str(scored_metrics_path)}, sort_keys=True), encoding="utf-8")
    paths["config_resolved"] = config_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-experiment", required=True)
    args = parser.parse_args()
    paths = compute_clean_offprior(args.scored_metrics, args.output_dir, args.source_experiment)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_clean_offprior.py -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/clean_offprior.py tests/off_prior_measurement/test_clean_offprior.py
git commit -m "feat: compute clean off-priorness scores"
```

## Task 2: Clean Diagnostic Figures

**Files:**

- Create: `tests/off_prior_measurement/test_plot_clean_offprior.py`
- Create: `scripts/off_prior_measurement/plot_clean_offprior.py`

- [ ] **Step 1: Write failing figure test**

Create a tiny clean-scored CSV with `reference_regime`, `conditioning_key`, `timestep`, `floor_adjusted_l2`, `clean_pair_l2`, `roundtrip_ratio`, and DCT band columns. Assert that these paths exist:

```python
raw_vs_clean_ladder
roundtrip_attribution_by_subject
clean_timestep_curves
clean_frequency_heatmap
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_plot_clean_offprior.py -v
```

Expected: import failure for `scripts.off_prior_measurement.plot_clean_offprior`.

- [ ] **Step 3: Implement plotting module**

Create `plot_clean_offprior.py` with `create_clean_figures(clean_scored_metrics_path, figures_dir) -> dict[str, Path]`. Use matplotlib Agg backend, group means with pandas, and write:

```text
raw_vs_clean_ladder.png
roundtrip_attribution_by_subject.png
clean_timestep_curves.png
clean_frequency_heatmap.png
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_plot_clean_offprior.py -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/plot_clean_offprior.py tests/off_prior_measurement/test_plot_clean_offprior.py
git commit -m "feat: plot clean off-prior diagnostics"
```

## Task 3: Clean Go / No-Go Conclusion

**Files:**

- Create: `tests/off_prior_measurement/test_write_clean_conclusion.py`
- Create: `scripts/off_prior_measurement/write_clean_conclusion.py`

- [ ] **Step 1: Write failing conclusion tests**

Create two synthetic `clean_ladder_summary.csv` files:

- Go case: 8 subjects, `clean_standard_reference > 0` for 5 subjects, positive mean, `clean_hard_reference >= clean_standard_reference` for most subjects, standard roundtrip ratio below 0.75.
- No-Go case: clean standard is mostly negative or roundtrip ratio is high.

Assert conclusion text contains:

```text
Stage 1.3 Clean Off-Priorness Conclusion
Go / no-go decision: Go
```

and:

```text
Go / no-go decision: No-Go
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_write_clean_conclusion.py -v
```

Expected: import failure for `scripts.off_prior_measurement.write_clean_conclusion`.

- [ ] **Step 3: Implement conclusion module**

Create `write_clean_conclusion.py` with:

```python
def write_clean_conclusion(experiment_dir: str | Path) -> Path:
    ...
```

The selected conditioning should prefer `class`, then `class_context`, then `null`. The Go rule must match the spec:

```text
clean standard positive subjects >= 5
clean standard mean > 0
clean hard is not systematically below clean standard
standard roundtrip ratio < 0.75
signal is not one-subject-only
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_write_clean_conclusion.py -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/off_prior_measurement/write_clean_conclusion.py tests/off_prior_measurement/test_write_clean_conclusion.py
git commit -m "feat: write clean off-priorness conclusion"
```

## Task 4: Documentation And Real Stage 1.3 Run

**Files:**

- Create: `experiments/off_prior_measurement_v0/ladder_v2_clean/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-23-roundtrip-confound-clean-offpriorness-design.md`
- Generate: `experiments/off_prior_measurement_v0/ladder_v2_clean/`

- [ ] **Step 1: Run full test suite**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
```

Expected: all tests pass.

- [ ] **Step 2: Run Stage 1.3 analysis on completed v2 data**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export SRC=experiments/off_prior_measurement_v0/ladder_v2
export DST=experiments/off_prior_measurement_v0/ladder_v2_clean

$PYTHON -m scripts.off_prior_measurement.clean_offprior \
  --scored-metrics $SRC/summaries/scored_metrics.csv \
  --output-dir $DST \
  --source-experiment $SRC

$PYTHON -m scripts.off_prior_measurement.plot_clean_offprior \
  --clean-scored-metrics $DST/summaries/clean_scored_metrics.csv \
  --figures-dir $DST/figures

$PYTHON -m scripts.off_prior_measurement.write_clean_conclusion \
  --experiment-dir $DST
```

Expected output files match the commit-worthy list above.

- [ ] **Step 3: Write experiment README**

Create `experiments/off_prior_measurement_v0/ladder_v2_clean/README.md` with:

```markdown
# Off-Prior Measurement V0 Ladder V2 Clean

Status: completed Stage 1.3 roundtrip-confound diagnostic.

Source experiment: `experiments/off_prior_measurement_v0/ladder_v2/`.

Purpose: subtract VAE roundtrip artifacts from the v2 floor-adjusted residuals before deciding whether Stage 2 personalization fine-tuning is justified.

Run order:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export SRC=experiments/off_prior_measurement_v0/ladder_v2
export DST=experiments/off_prior_measurement_v0/ladder_v2_clean

$PYTHON -m scripts.off_prior_measurement.clean_offprior --scored-metrics $SRC/summaries/scored_metrics.csv --output-dir $DST --source-experiment $SRC
$PYTHON -m scripts.off_prior_measurement.plot_clean_offprior --clean-scored-metrics $DST/summaries/clean_scored_metrics.csv --figures-dir $DST/figures
$PYTHON -m scripts.off_prior_measurement.write_clean_conclusion --experiment-dir $DST
```
```

- [ ] **Step 4: Update docs with final Stage 1.3 result**

Update `README.md` immediate next step and the Stage 1.3 spec status to reflect the actual conclusion generated by `ladder_v2_clean/conclusion.md`.

- [ ] **Step 5: Commit results and backup**

Run:

```bash
git add README.md docs/superpowers/specs/2026-06-23-roundtrip-confound-clean-offpriorness-design.md experiments/off_prior_measurement_v0/ladder_v2_clean
git commit -m "data: record clean off-priorness diagnostic"
```

Then sync to GitHub with the existing GitHub API workflow. Verify:

```text
contains_token False
```

## Plan Self-Review

- Spec coverage: clean scores, pairwise fallback, ratio, tables, figures, conclusion, docs, and CPU-only real run are all assigned to tasks.
- Placeholder scan: no placeholder markers remain.
- Type consistency: `clean_pair_l2`, `clean_subject_l2`, `roundtrip_baseline_l2`, and `roundtrip_ratio` are used consistently across tasks.
- Scope check: the plan does not add personalization fine-tuning, new data downloads, or GPU measurement.
