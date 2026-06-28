from __future__ import annotations

import argparse
from pathlib import Path

from scripts.personalization_training.config import load_training_config


def report_text(config_path: Path) -> str:
    config = load_training_config(config_path)
    subjects = ", ".join(subject.subject_id for subject in config.subjects)
    return "\n".join(
        [
            "# Stage 2A LF-Late Training Validation",
            "",
            "Status: pending training run.",
            "",
            f"Config: `{config_path}`",
            f"Subjects: {subjects}",
            f"Default condition: {config.training.condition}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Stage 2A result report.")
    parser.add_argument("--config", required=True, help="Path to Stage 2A YAML config.")
    parser.add_argument("--output", help="Report markdown path.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing it.")
    args = parser.parse_args()

    config_path = Path(args.config)
    text = report_text(config_path)
    if args.dry_run:
        print(text)
        return
    output = Path(args.output) if args.output else load_training_config(config_path).output_dir / "README.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote report: {output}")


if __name__ == "__main__":
    main()
