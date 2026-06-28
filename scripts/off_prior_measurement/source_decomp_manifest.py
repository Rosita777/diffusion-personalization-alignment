from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings

REQUIRED_COLUMNS = [
    "subject_id",
    "class_name",
    "image_id",
    "image_path",
    "roundtrip_image_path",
    "source_group",
    "reference_regime",
    "hardness_axis",
    "conditioning_key",
    "conditioning_prompt",
    "source_dataset",
    "source_license_note",
]


def _read_yaml(path: str | Path) -> dict:
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}


def load_ordinary_real_controls(path: str | Path) -> pd.DataFrame:
    raw = _read_yaml(path)
    rows = raw.get("ordinary_real_controls", [])
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame = pd.DataFrame(rows, dtype=str)
    frame["subject_id"] = frame["class_name"].astype(str)
    frame["source_group"] = "ordinary_real_control"
    if "reference_regime" not in frame.columns:
        frame["reference_regime"] = "ordinary_real_control"
    if "hardness_axis" not in frame.columns:
        frame["hardness_axis"] = "none"
    if "conditioning_key" not in frame.columns:
        frame["conditioning_key"] = "class"
    if "conditioning_prompt" not in frame.columns:
        frame["conditioning_prompt"] = "a photo of a " + frame["class_name"].astype(str)
    return frame


def _validate_ordinary_image_paths(ordinary: pd.DataFrame) -> None:
    missing = [path for path in ordinary["image_path"].astype(str).tolist() if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing ordinary real control image paths: {preview}")


def _roundtrip_path(root: Path, row: dict[str, object]) -> str:
    return str(root / str(row["source_group"]) / str(row["class_name"]) / f"{row['image_id']}.png")


def _normalize_rows(frame: pd.DataFrame, source_group: str, roundtrip_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_group"] = source_group
    if "source_dataset" not in frame.columns:
        frame["source_dataset"] = source_group
    if "source_license_note" not in frame.columns:
        frame["source_license_note"] = "generated or local project artifact"
    if "roundtrip_image_path" not in frame.columns:
        frame["roundtrip_image_path"] = [
            _roundtrip_path(roundtrip_root, row) for row in frame.to_dict("records")
        ]
    return frame


def build_source_decomp_manifest(
    reference_manifest_path: str | Path,
    control_manifest_path: str | Path,
    ordinary_real_manifest_path: str | Path,
    roundtrip_root: str | Path,
) -> pd.DataFrame:
    roundtrip_root = Path(roundtrip_root)
    reference = read_csv_preserve_strings(reference_manifest_path)
    controls = read_csv_preserve_strings(control_manifest_path)
    ordinary = load_ordinary_real_controls(ordinary_real_manifest_path)
    if ordinary.empty:
        raise ValueError("Stage 1.4 requires ordinary real controls; provide ordinary_real_controls entries")
    _validate_ordinary_image_paths(ordinary)
    ordinary_classes = set(ordinary["class_name"].astype(str).tolist())

    base = controls[controls["source_group"].isin(["base_easy_control", "base_generated_control"])].copy()
    base = base[base["class_name"].astype(str).isin(ordinary_classes)]
    base = _normalize_rows(base, "base_generated_control", roundtrip_root)
    standard = reference[reference["source_group"] == "dreambooth_reference"].copy()
    standard = standard[standard["class_name"].astype(str).isin(ordinary_classes)]
    standard = _normalize_rows(standard, "dreambooth_reference", roundtrip_root)
    ordinary = _normalize_rows(ordinary, "ordinary_real_control", roundtrip_root)

    combined = pd.concat([base, ordinary, standard], ignore_index=True)
    for column in REQUIRED_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    return combined[REQUIRED_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--control-manifest", required=True)
    parser.add_argument("--ordinary-real-manifest", required=True)
    parser.add_argument("--roundtrip-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_source_decomp_manifest(
        args.reference_manifest,
        args.control_manifest,
        args.ordinary_real_manifest,
        args.roundtrip_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
