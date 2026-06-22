from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings

SMOKE_SUBJECTS = ["dog", "cat", "backpack", "clock", "vase"]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _conditioning_rows(reference: pd.DataFrame, key: str) -> pd.DataFrame:
    return reference[reference["conditioning_key"] == key]


def _positive_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    return int((rows["mean_floor_adjusted_l2"] > 0).sum())


def _mean_adjusted(rows: pd.DataFrame) -> float:
    if rows.empty:
        return float("nan")
    return float(rows["mean_floor_adjusted_l2"].mean())


def write_conclusion(experiment_dir: str | Path) -> Path:
    experiment_dir = Path(experiment_dir)
    subject_summary = read_csv_preserve_strings(experiment_dir / "summaries" / "subject_summary.csv")
    regime_summary = read_csv_preserve_strings(experiment_dir / "summaries" / "regime_summary.csv")
    scored = read_csv_preserve_strings(experiment_dir / "summaries" / "scored_metrics.csv")

    reference = subject_summary[subject_summary["source_group"] == "dreambooth_reference"]
    class_rows = _conditioning_rows(reference, "class")
    context_rows = _conditioning_rows(reference, "class_context")
    null_rows = _conditioning_rows(reference, "null")

    class_positive = _positive_count(class_rows)
    context_positive = _positive_count(context_rows)
    null_positive = _positive_count(null_rows)
    go = class_positive >= 4 or context_positive >= 4

    strongest_timestep_row = (
        regime_summary[regime_summary["source_group"] == "dreambooth_reference"]
        .sort_values("mean_floor_adjusted_l2", ascending=False)
        .iloc[0]
    )
    dreambooth_scored = scored[scored["source_group"] == "dreambooth_reference"]
    band_means = {
        "low": dreambooth_scored["dct_delta_low"].mean(),
        "mid": dreambooth_scored["dct_delta_mid"].mean(),
        "high": dreambooth_scored["dct_delta_high"].mean(),
    }
    strongest_band = max(band_means, key=band_means.get)

    conclusion = f"""# Stage 1 Smoke-Test Conclusion

Date: 2026-06-22

Config:

```text
configs/off_prior_measurement_v0/smoke_test.yaml
```

Primary comparison:

```text
dreambooth_reference vs. base_easy_control
```

Subjects:

```text
{", ".join(SMOKE_SUBJECTS)}
```

Result summary:

- Class-prompt conditioning: {class_positive} of 5 subjects have positive mean floor-adjusted residual; mean value = {_fmt(_mean_adjusted(class_rows))}.
- Class-plus-context conditioning: {context_positive} of 5 subjects have positive mean floor-adjusted residual; mean value = {_fmt(_mean_adjusted(context_rows))}.
- Null conditioning: {null_positive} of 5 subjects have positive mean floor-adjusted residual; mean value = {_fmt(_mean_adjusted(null_rows))}.
- Strongest timestep by mean floor-adjusted residual: {int(strongest_timestep_row['timestep'])}.
- Strongest latent DCT band by mean residual energy: {strongest_band}.

Interpretation:

- Go / no-go decision: {"Go" if go else "No-Go"}.
- Main reason: class or class-plus-context conditioning passes the 4-of-5 subject rule = {go}.
- Most important caveat: this smoke test measures target residual structure only; it does not prove downstream forgetting until Stage 2.

Next step:

- If Go: create the Stage 2 correlation-with-forgetting plan.
- If No-Go: revise the off-priorness metric or conditioning design before any personalization training.
"""
    path = experiment_dir / "conclusion.md"
    path.write_text(conclusion, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    print(write_conclusion(args.experiment_dir))


if __name__ == "__main__":
    main()
