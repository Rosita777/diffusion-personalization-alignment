from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _select_conditioning(gaps: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    for key in ["class", "prompt_matched", "class_context", "null"]:
        rows = gaps[gaps["conditioning_key"] == key]
        if not rows.empty:
            return key, rows
    return "none", gaps.iloc[0:0]


def _positive_concentration(values: pd.Series) -> float:
    positive = values[values > 0]
    total = float(positive.sum())
    if total <= 0:
        return 1.0
    return float(positive.max() / total)


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def write_source_decomp_conclusion(experiment_dir: str | Path) -> Path:
    experiment_dir = Path(experiment_dir)
    summaries_dir = experiment_dir / "summaries"
    gaps = read_csv_preserve_strings(summaries_dir / "source_gap_summary.csv")
    source_groups = read_csv_preserve_strings(summaries_dir / "source_group_summary.csv")
    gaps = _coerce_numeric(
        gaps,
        [
            "real_domain_gap",
            "subject_specific_gap",
            "natural_hard_gap",
        ],
    )
    source_groups = _coerce_numeric(source_groups, ["mean_artifact_fraction"])
    selected, rows = _select_conditioning(gaps)
    class_count = int(rows["class_name"].nunique()) if not rows.empty else 0
    positive_subject_classes = int((rows["subject_specific_gap"] > 0).sum()) if class_count else 0
    mean_subject_gap = float(rows["subject_specific_gap"].mean()) if class_count else float("nan")
    mean_real_gap = float(rows["real_domain_gap"].mean()) if class_count else float("nan")
    mean_natural_hard_gap = float(rows["natural_hard_gap"].mean()) if class_count else float("nan")
    concentration = _positive_concentration(rows["subject_specific_gap"]) if class_count else 1.0

    dreambooth_rows = source_groups[
        (source_groups["source_group"] == "dreambooth_reference")
        & (source_groups["conditioning_key"] == selected)
    ]
    has_dreambooth_rows = not dreambooth_rows.empty
    can_evaluate_personalization = has_dreambooth_rows and bool(rows["subject_specific_gap"].notna().any())
    mean_artifact_fraction = (
        float(dreambooth_rows["mean_artifact_fraction"].mean()) if not dreambooth_rows.empty else float("nan")
    )

    enough_positive = positive_subject_classes >= 3
    positive_mean = mean_subject_gap > 0
    artifact_ok = mean_artifact_fraction < 0.75
    not_one_class_only = concentration < 0.80
    go = can_evaluate_personalization and enough_positive and positive_mean and artifact_ok and not_one_class_only
    decision = "Go" if go else "Pivot" if can_evaluate_personalization else "Control-only diagnosis"
    interpretation = (
        "- If Go: run Stage 2 forgetting correlation with source-decomposed clean gaps.\n"
        "- If Pivot: revise the paper story toward real-image projection/domain alignment or redesign the measurement."
        if can_evaluate_personalization
        else "- This run does not contain DreamBooth reference rows for the selected conditioning; use it only to diagnose the base-generated versus ordinary-real control gap."
    )

    conclusion = f"""# Target-Gap Source Decomposition Conclusion

Selected conditioning:

```text
{selected}
```

Result summary:

- Subject-specific positive classes: {positive_subject_classes} of {class_count}.
- Mean real-domain gap: {_fmt(mean_real_gap)}.
- Mean subject-specific gap: {_fmt(mean_subject_gap)}.
- Mean natural-hard gap: {_fmt(mean_natural_hard_gap)}.
- Mean DreamBooth artifact fraction: {_fmt(mean_artifact_fraction)}.
- DreamBooth reference rows present: {has_dreambooth_rows}.
- Positive subject-gap concentration: {_fmt(concentration)}.

Go / Pivot checks:

- Subject-specific gap is positive for at least 3 classes: {enough_positive}.
- Mean subject-specific gap is positive: {positive_mean}.
- DreamBooth artifact fraction is below 0.75: {artifact_ok}.
- Signal is not concentrated in one class: {not_one_class_only}.

Interpretation:

- Go / pivot decision: {decision}.
{interpretation}
"""
    path = experiment_dir / "conclusion.md"
    path.write_text(conclusion, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    print(write_source_decomp_conclusion(args.experiment_dir))


if __name__ == "__main__":
    main()
