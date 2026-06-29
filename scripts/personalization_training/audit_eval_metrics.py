from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageStat


@dataclass(frozen=True)
class EvalRecord:
    kind: str
    text: str
    prompt_index: int
    image_index: int
    seed: int
    path: Path

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (self.kind, self.text, self.prompt_index, self.image_index, self.seed)


def _resolve_image_path(eval_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return eval_dir / path


def load_manifest(eval_dir: Path) -> list[EvalRecord]:
    manifest_path = eval_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in data["images"]:
        records.append(
            EvalRecord(
                kind=str(item["kind"]),
                text=str(item["text"]),
                prompt_index=int(item["prompt_index"]),
                image_index=int(item["image_index"]),
                seed=int(item["seed"]),
                path=_resolve_image_path(eval_dir, str(item["path"])),
            )
        )
    return records


def mean_abs_rgb_distance(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        rgb_a = image_a.convert("RGB")
        rgb_b = image_b.convert("RGB")
        if rgb_a.size != rgb_b.size:
            raise ValueError(f"Image sizes differ: {path_a}={rgb_a.size}, {path_b}={rgb_b.size}")
        diff = ImageChops.difference(rgb_a, rgb_b)
        channel_means = ImageStat.Stat(diff).mean
    return sum(channel_means[:3]) / 3.0


def _index_records(records: Sequence[EvalRecord]) -> dict[tuple[str, str, int, int, int], EvalRecord]:
    indexed = {}
    for record in records:
        if record.key in indexed:
            raise ValueError(f"Duplicate eval manifest key: {record.key}")
        indexed[record.key] = record
    return indexed


def compare_runs(
    run_name: str,
    base: Sequence[EvalRecord],
    run: Sequence[EvalRecord],
    vanilla: Sequence[EvalRecord] | None,
) -> list[dict[str, object]]:
    base_by_key = _index_records(base)
    vanilla_by_key = _index_records(vanilla) if vanilla is not None else {}
    rows: list[dict[str, object]] = []
    for record in run:
        if record.key not in base_by_key:
            raise ValueError(f"Run '{run_name}' record has no matching base image: {record.key}")
        base_record = base_by_key[record.key]
        row: dict[str, object] = {
            "run": run_name,
            "kind": record.kind,
            "text": record.text,
            "prompt_index": record.prompt_index,
            "image_index": record.image_index,
            "seed": record.seed,
            "image_path": str(record.path),
            "base_image_path": str(base_record.path),
            "distance_to_base": mean_abs_rgb_distance(record.path, base_record.path),
        }
        if vanilla is not None:
            if record.key not in vanilla_by_key:
                raise ValueError(f"Run '{run_name}' record has no matching vanilla image: {record.key}")
            vanilla_record = vanilla_by_key[record.key]
            row["vanilla_image_path"] = str(vanilla_record.path)
            row["distance_to_vanilla"] = mean_abs_rgb_distance(record.path, vanilla_record.path)
        else:
            row["vanilla_image_path"] = ""
            row["distance_to_vanilla"] = ""
        rows.append(row)
    return rows


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pairwise_diversity(records: Sequence[EvalRecord]) -> float | str:
    if len(records) < 2:
        return ""
    distances = [mean_abs_rgb_distance(left.path, right.path) for left, right in combinations(records, 2)]
    return _mean(distances)


def summarize_rows(
    rows: Sequence[dict[str, object]],
    *,
    run_records: dict[str, Sequence[EvalRecord]] | None = None,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["run"]), str(row["kind"])), []).append(row)

    summaries = []
    for run_name, kind in sorted(groups):
        group = groups[(run_name, kind)]
        diversity: float | str = ""
        if run_records is not None and run_name in run_records:
            diversity = _pairwise_diversity([record for record in run_records[run_name] if record.kind == kind])
        vanilla_values = [row["distance_to_vanilla"] for row in group if row["distance_to_vanilla"] != ""]
        summaries.append(
            {
                "run": run_name,
                "kind": kind,
                "num_images": len(group),
                "mean_distance_to_base": _mean([float(row["distance_to_base"]) for row in group]),
                "mean_distance_to_vanilla": _mean([float(value) for value in vanilla_values])
                if vanilla_values
                else "",
                "mean_pairwise_diversity": diversity,
            }
        )
    return summaries


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_csv_value(row.get(name, "")) for name in fieldnames})


def _parse_run_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run-dir must use name=path format")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("--run-dir name must be non-empty")
    return name, Path(path)


def run_audit(
    *,
    base_dir: Path,
    run_dirs: Sequence[tuple[str, Path]],
    vanilla_dir: Path | None,
    output_summary: Path,
    output_per_image: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_records = load_manifest(base_dir)
    vanilla_records = load_manifest(vanilla_dir) if vanilla_dir is not None else None

    all_rows: list[dict[str, object]] = []
    records_by_run: dict[str, Sequence[EvalRecord]] = {}
    for run_name, run_dir in run_dirs:
        records = load_manifest(run_dir)
        records_by_run[run_name] = records
        all_rows.extend(compare_runs(run_name, base_records, records, vanilla=vanilla_records))

    summary_rows = summarize_rows(all_rows, run_records=records_by_run)
    write_csv(
        output_per_image,
        all_rows,
        [
            "run",
            "kind",
            "text",
            "prompt_index",
            "image_index",
            "seed",
            "distance_to_base",
            "distance_to_vanilla",
            "image_path",
            "base_image_path",
            "vanilla_image_path",
        ],
    )
    write_csv(
        output_summary,
        summary_rows,
        [
            "run",
            "kind",
            "num_images",
            "mean_distance_to_base",
            "mean_distance_to_vanilla",
            "mean_pairwise_diversity",
        ],
    )
    return all_rows, summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit generated eval images with lightweight pixel metrics.")
    parser.add_argument("--base-dir", required=True, type=Path, help="Eval directory for the frozen base run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        type=_parse_run_dir,
        help="Run to audit in name=path format. Can be passed multiple times.",
    )
    parser.add_argument("--vanilla-dir", type=Path, help="Eval directory for the vanilla personalized run.")
    parser.add_argument("--output-summary", required=True, type=Path, help="Path for summary CSV output.")
    parser.add_argument("--output-per-image", required=True, type=Path, help="Path for per-image CSV output.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_audit(
        base_dir=args.base_dir,
        run_dirs=args.run_dir,
        vanilla_dir=args.vanilla_dir,
        output_summary=args.output_summary,
        output_per_image=args.output_per_image,
    )


if __name__ == "__main__":
    main()
