from __future__ import annotations

from pathlib import Path

import pandas as pd

STRING_COLUMNS = [
    "subject_id",
    "image_id",
    "image_path",
    "source_group",
    "reference_regime",
    "hardness_axis",
    "source_standard_image",
    "variant_id",
    "transform_parameters",
    "class_name",
    "class_prompt",
    "class_context_prompt",
    "conditioning_key",
    "conditioning_prompt",
]


def normalize_conditioning_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "conditioning_key" in frame.columns:
        frame["conditioning_key"] = frame["conditioning_key"].fillna("").replace("", "null")
    if "conditioning_prompt" in frame.columns:
        frame["conditioning_prompt"] = frame["conditioning_prompt"].fillna("")
    return frame


def read_csv_preserve_strings(path: str | Path) -> pd.DataFrame:
    return normalize_conditioning_columns(
        pd.read_csv(path, keep_default_na=False, dtype={column: str for column in STRING_COLUMNS})
    )
