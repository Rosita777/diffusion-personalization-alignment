from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _select_conditioning(ladder: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    for key in ["class", "class_context", "null"]:
        rows = ladder[ladder["conditioning_key"] == key]
        if not rows.empty:
            return key, rows
    return "none", ladder.iloc[0:0]


def _positive_concentration(values: pd.Series) -> float:
    positive = values[values > 0]
    total = float(positive.sum())
    if total <= 0:
        return 1.0
    return float(positive.max() / total)


def _safe_mean(rows: pd.DataFrame, column: str) -> float:
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows[column].mean())


def write_clean_conclusion(experiment_dir: str | Path) -> Path:
    experiment_dir = Path(experiment_dir)
    summaries_dir = experiment_dir / "summaries"
    ladder = read_csv_preserve_strings(summaries_dir / "clean_ladder_summary.csv")
    regime_summary = read_csv_preserve_strings(summaries_dir / "clean_regime_summary.csv")

    for column in [
        "clean_standard_reference",
        "clean_hard_reference",
        "raw_standard_reference",
        "raw_hard_reference",
        "ratio_standard_reference",
        "clean_hard_minus_standard",
    ]:
        if column not in ladder.columns:
            ladder[column] = float("nan")

    best_conditioning, rows = _select_conditioning(ladder)
    subject_count = int(rows["subject_id"].nunique()) if not rows.empty else 0
    clean_standard_positive = int((rows["clean_standard_reference"] > 0).sum()) if subject_count else 0
    clean_hard_not_below = int((rows["clean_hard_reference"] >= rows["clean_standard_reference"]).sum()) if subject_count else 0
    clean_standard_mean = _safe_mean(rows, "clean_standard_reference")
    clean_hard_mean = _safe_mean(rows, "clean_hard_reference")
    raw_standard_mean = _safe_mean(rows, "raw_standard_reference")
    raw_hard_mean = _safe_mean(rows, "raw_hard_reference")
    ratio_mean = _safe_mean(rows, "ratio_standard_reference")
    concentration = _positive_concentration(rows["clean_standard_reference"]) if subject_count else 1.0

    enough_positive = clean_standard_positive >= 5
    positive_mean = clean_standard_mean > 0
    hard_not_systematically_below = clean_hard_not_below >= max(1, subject_count // 2)
    ratio_ok = ratio_mean < 0.75
    not_one_subject_only = concentration < 0.80
    go = (
        enough_positive
        and positive_mean
        and hard_not_systematically_below
        and ratio_ok
        and not_one_subject_only
    )

    standard_rows = regime_summary[
        (regime_summary["reference_regime"] == "standard_reference")
        & (regime_summary["conditioning_key"] == best_conditioning)
    ]
    if standard_rows.empty:
        strongest_timestep = "not available"
    else:
        strongest_timestep = str(
            int(standard_rows.sort_values("mean_clean_pair_l2", ascending=False).iloc[0]["timestep"])
        )

    conclusion = f"""# Stage 1.3 Clean Off-Priorness Conclusion

Date: 2026-06-25

Source experiment:

```text
experiments/off_prior_measurement_v0/ladder_v2
```

Selected conditioning:

```text
{best_conditioning}
```

Result summary:

- Clean standard-reference positive subjects: {clean_standard_positive} of {subject_count}.
- Clean hard-reference not below standard subjects: {clean_hard_not_below} of {subject_count}.
- Mean raw standard-reference residual: {_fmt(raw_standard_mean)}.
- Mean raw hard-reference residual: {_fmt(raw_hard_mean)}.
- Mean clean standard-reference residual: {_fmt(clean_standard_mean)}.
- Mean clean hard-reference residual: {_fmt(clean_hard_mean)}.
- Mean standard-reference roundtrip attribution ratio: {_fmt(ratio_mean)}.
- Positive clean-standard concentration: {_fmt(concentration)}.
- Strongest clean standard-reference timestep: {strongest_timestep}.

Go / No-Go checks:

- Clean standard positive for at least 5 subjects: {enough_positive}.
- Clean standard mean is positive: {positive_mean}.
- Clean hard is not systematically below clean standard: {hard_not_systematically_below}.
- Standard roundtrip attribution ratio is below 0.75: {ratio_ok}.
- Signal is not concentrated in one subject: {not_one_subject_only}.

Interpretation:

- Go / no-go decision: {"Go" if go else "No-Go"}.
- If Go: proceed to Stage 2 correlation-with-forgetting using clean off-priorness.
- If No-Go: revise off-priorness measurement before personalization fine-tuning.

Caveat:

This diagnostic subtracts VAE roundtrip artifacts from existing v2 measurements. It still does not prove downstream personalization forgetting.
"""
    path = experiment_dir / "conclusion.md"
    path.write_text(conclusion, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    print(write_clean_conclusion(args.experiment_dir))


if __name__ == "__main__":
    main()
