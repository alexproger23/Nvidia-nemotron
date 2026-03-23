from __future__ import annotations

import csv
import shutil
from pathlib import Path


def run_inference(input_dir: Path, output_file: Path, model_dir: Path | None = None) -> None:
    """Replace this stub with real model inference logic."""
    del model_dir

    input_dir = input_dir.resolve()
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sample_submission = input_dir / "sample_submission.csv"
    if sample_submission.exists():
        shutil.copyfile(sample_submission, output_file)
        return

    test_csv = input_dir / "test.csv"
    if test_csv.exists():
        rows = list(csv.DictReader(test_csv.open("r", encoding="utf-8-sig", newline="")))
        if rows:
            first_column = next(iter(rows[0].keys()))
            with output_file.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([first_column, "target"])
                for row in rows:
                    writer.writerow([row[first_column], 0])
            return

    with output_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "target"])
        writer.writerow([0, 0])
