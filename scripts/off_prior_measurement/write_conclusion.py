from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _best_conditioning_ladder(ladder: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    candidates: list[tuple[int, str, pd.DataFrame]] = []
    for key in ["class", "class_context"]:
        rows = ladder[ladder["conditioning_key"] == key]
        if rows.empty:
            continue
        hard_positive = int((rows["hard_reference"] > 0).sum())
        hard_gt_standard = int((rows["hard_reference"] > rows["standard_reference"]).sum())
        standard_gt_easy = int((rows["standard_reference"] > rows["easy_control"]).sum())
        candidates.append((hard_positive + hard_gt_standard + standard_gt_easy, key, rows))
    if not candidates:
        null_rows = ladder[ladder["conditioning_key"] == "null"]
        if not null_rows.empty:
            return "null", null_rows
        return "none", ladder.iloc[0:0]
    _, key, rows = max(candidates, key=lambda item: item[0])
    return key, rows


def _strongest_timestep(regime_summary: pd.DataFrame) -> str:
    rows = regime_summary[regime_summary["source_group"] == "dreambooth_hard_reference"]
    if rows.empty:
        rows = regime_summary
    if rows.empty or "timestep" not in rows.columns:
        return "not available"
    strongest = rows.sort_values("mean_floor_adjusted_l2", ascending=False).iloc[0]
    return str(int(strongest["timestep"]))


def _strongest_band(scored: pd.DataFrame) -> str:
    rows = scored[scored["source_group"] == "dreambooth_hard_reference"]
    if rows.empty:
        rows = scored
    band_means = {
        "low": rows["dct_delta_low"].mean(),
        "mid": rows["dct_delta_mid"].mean(),
        "high": rows["dct_delta_high"].mean(),
    }
    return max(band_means, key=band_means.get)


def write_conclusion(experiment_dir: str | Path) -> Path:
    experiment_dir = Path(experiment_dir)
    subject_summary = read_csv_preserve_strings(experiment_dir / "summaries" / "subject_summary.csv")
    regime_summary = read_csv_preserve_strings(experiment_dir / "summaries" / "regime_summary.csv")
    scored = read_csv_preserve_strings(experiment_dir / "summaries" / "scored_metrics.csv")
    ladder = read_csv_preserve_strings(experiment_dir / "summaries" / "ladder_summary.csv")

    for column in ["easy_control", "standard_reference", "hard_reference", "hard_control", "roundtrip_control"]:
        if column not in ladder.columns:
            ladder[column] = float("nan")

    best_conditioning, ladder_rows = _best_conditioning_ladder(ladder)
    subject_count = int(ladder_rows["subject_id"].nunique()) if not ladder_rows.empty else 0

    hard_positive = int((ladder_rows["hard_reference"] > 0).sum()) if subject_count else 0
    hard_gt_standard = (
        int((ladder_rows["hard_reference"] > ladder_rows["standard_reference"]).sum()) if subject_count else 0
    )
    standard_gt_easy = (
        int((ladder_rows["standard_reference"] > ladder_rows["easy_control"]).sum()) if subject_count else 0
    )

    base_hard_rows = subject_summary[
        (subject_summary["source_group"] == "base_hard_control")
        & (subject_summary["conditioning_key"] == best_conditioning)
    ]
    base_hard_positive = int((base_hard_rows["mean_floor_adjusted_l2"] > 0).sum())
    roundtrip_ok = bool(
        subject_count
        and (ladder_rows["roundtrip_control"] <= ladder_rows["standard_reference"]).sum() >= subject_count // 2
    )

    go = (
        hard_positive >= 6
        and hard_gt_standard >= 6
        and standard_gt_easy >= 4
        and base_hard_positive >= max(1, subject_count // 2)
        and roundtrip_ok
    )

    strongest_timestep = _strongest_timestep(regime_summary)
    strongest_band = _strongest_band(scored)
    hard_mean = float(ladder_rows["hard_reference"].mean()) if subject_count else float("nan")
    standard_mean = float(ladder_rows["standard_reference"].mean()) if subject_count else float("nan")
    easy_mean = float(ladder_rows["easy_control"].mean()) if subject_count else float("nan")

    conclusion = f"""# Stage 1 V2 Prior-Compatibility Ladder Conclusion

Date: 2026-06-23

Primary comparison:

```text
easy_control < standard_reference < hard_reference
```

Selected conditioning:

```text
{best_conditioning}
```

Result summary:

- Hard-reference positive subjects: {hard_positive} of {subject_count}.
- Hard greater than standard: {hard_gt_standard} of {subject_count}.
- Standard greater than easy: {standard_gt_easy} of {subject_count}.
- Base hard-control positive subjects: {base_hard_positive} of {subject_count}.
- Roundtrip sanity check passed: {roundtrip_ok}.
- Mean easy-control floor-adjusted residual: {_fmt(easy_mean)}.
- Mean standard-reference floor-adjusted residual: {_fmt(standard_mean)}.
- Mean hard-reference floor-adjusted residual: {_fmt(hard_mean)}.
- Strongest timestep by mean floor-adjusted residual: {strongest_timestep}.
- Strongest latent DCT band by mean residual energy: {strongest_band}.

Interpretation:

- Go / no-go decision: {"Go" if go else "No-Go"}.
- If Go: proceed to Stage 2 correlation-with-forgetting and DADT target-correction design.
- If No-Go: revise off-priorness measurement before any personalization fine-tuning.

Caveat:

This conclusion measures target residual structure only. It does not prove downstream forgetting until Stage 2 fine-tuning and prior-drift evaluation are run.
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
