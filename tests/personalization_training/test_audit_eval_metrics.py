from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


def _write_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), rgb).save(path)


def _record(kind: str, prompt_index: int, image_index: int, seed: int, path: Path) -> dict[str, object]:
    text = "a photo of sks vase" if kind == "subject" else "a photo of a vase"
    return {
        "kind": kind,
        "text": text,
        "prompt_index": prompt_index,
        "image_index": image_index,
        "seed": seed,
        "path": str(path),
    }


def _write_manifest(eval_dir: Path, values: dict[str, tuple[int, int, int]]) -> None:
    images = []
    for kind, prompt_index, image_index, seed in [
        ("subject", 0, 0, 0),
        ("subject", 0, 1, 1),
        ("class", 1, 0, 100),
        ("class", 1, 1, 101),
    ]:
        name = f"{kind}_{image_index}.png"
        image_path = eval_dir / name
        _write_image(image_path, values[f"{kind}_{image_index}"])
        images.append(_record(kind, prompt_index, image_index, seed, image_path))
    (eval_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "fake",
                "subject_id": "vase",
                "weights_dir": None,
                "num_images": len(images),
                "images": images,
            }
        ),
        encoding="utf-8",
    )


def test_mean_abs_rgb_distance_uses_matched_manifest_records(tmp_path):
    from scripts.personalization_training.audit_eval_metrics import compare_runs, load_manifest

    base_dir = tmp_path / "base"
    run_dir = tmp_path / "run"
    _write_manifest(
        base_dir,
        {
            "subject_0": (10, 10, 10),
            "subject_1": (10, 10, 10),
            "class_0": (10, 10, 10),
            "class_1": (10, 10, 10),
        },
    )
    _write_manifest(
        run_dir,
        {
            "subject_0": (20, 20, 20),
            "subject_1": (30, 30, 30),
            "class_0": (15, 15, 15),
            "class_1": (25, 25, 25),
        },
    )

    rows = compare_runs("run", load_manifest(base_dir), load_manifest(run_dir), vanilla=None)

    assert [row["kind"] for row in rows] == ["subject", "subject", "class", "class"]
    assert [row["distance_to_base"] for row in rows] == [10.0, 20.0, 5.0, 15.0]


def test_summarize_rows_groups_by_run_and_kind(tmp_path):
    from scripts.personalization_training.audit_eval_metrics import summarize_rows

    rows = [
        {"run": "vanilla", "kind": "subject", "distance_to_base": 10.0, "distance_to_vanilla": 0.0},
        {"run": "vanilla", "kind": "subject", "distance_to_base": 20.0, "distance_to_vanilla": 0.0},
        {"run": "vanilla", "kind": "class", "distance_to_base": 5.0, "distance_to_vanilla": 0.0},
        {"run": "dadt", "kind": "class", "distance_to_base": 7.0, "distance_to_vanilla": 2.0},
        {"run": "dadt", "kind": "class", "distance_to_base": 9.0, "distance_to_vanilla": 4.0},
    ]

    summary = summarize_rows(rows)

    assert summary == [
        {
            "run": "dadt",
            "kind": "class",
            "num_images": 2,
            "mean_distance_to_base": 8.0,
            "mean_distance_to_vanilla": 3.0,
            "mean_pairwise_diversity": "",
        },
        {
            "run": "vanilla",
            "kind": "class",
            "num_images": 1,
            "mean_distance_to_base": 5.0,
            "mean_distance_to_vanilla": 0.0,
            "mean_pairwise_diversity": "",
        },
        {
            "run": "vanilla",
            "kind": "subject",
            "num_images": 2,
            "mean_distance_to_base": 15.0,
            "mean_distance_to_vanilla": 0.0,
            "mean_pairwise_diversity": "",
        },
    ]


def test_cli_writes_summary_and_per_image_csv(tmp_path):
    from scripts.personalization_training.audit_eval_metrics import main

    base_dir = tmp_path / "base"
    vanilla_dir = tmp_path / "vanilla"
    dadt_dir = tmp_path / "dadt"
    _write_manifest(
        base_dir,
        {
            "subject_0": (10, 10, 10),
            "subject_1": (20, 20, 20),
            "class_0": (30, 30, 30),
            "class_1": (40, 40, 40),
        },
    )
    _write_manifest(
        vanilla_dir,
        {
            "subject_0": (20, 20, 20),
            "subject_1": (30, 30, 30),
            "class_0": (40, 40, 40),
            "class_1": (50, 50, 50),
        },
    )
    _write_manifest(
        dadt_dir,
        {
            "subject_0": (21, 21, 21),
            "subject_1": (31, 31, 31),
            "class_0": (35, 35, 35),
            "class_1": (45, 45, 45),
        },
    )
    summary_path = tmp_path / "summary.csv"
    per_image_path = tmp_path / "per_image.csv"

    main(
        [
            "--base-dir",
            str(base_dir),
            "--vanilla-dir",
            str(vanilla_dir),
            "--run-dir",
            f"vanilla={vanilla_dir}",
            "--run-dir",
            f"dadt={dadt_dir}",
            "--output-summary",
            str(summary_path),
            "--output-per-image",
            str(per_image_path),
        ]
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        per_image_rows = list(csv.DictReader(handle))

    assert b"\r\n" not in summary_path.read_bytes()
    assert b"\r\n" not in per_image_path.read_bytes()
    assert len(per_image_rows) == 8
    assert {
        (row["run"], row["kind"], row["mean_distance_to_base"], row["mean_distance_to_vanilla"])
        for row in summary_rows
    } == {
        ("dadt", "class", "5.000000", "5.000000"),
        ("dadt", "subject", "11.000000", "1.000000"),
        ("vanilla", "class", "10.000000", "0.000000"),
        ("vanilla", "subject", "10.000000", "0.000000"),
    }
