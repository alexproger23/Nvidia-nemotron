from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.data import prepare_reasoning_dataset_from_competition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare normalized reasoning data from Kaggle competition files.")
    parser.add_argument(
        "--competition-data",
        type=Path,
        default=Path("data/raw"),
        help="Directory or .zip archive with Kaggle competition data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reasoning"),
        help="Directory where normalized JSONL files should be written.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of train.csv reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before splitting train and validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = prepare_reasoning_dataset_from_competition(
        source=args.competition_data,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    for split, path in outputs.items():
        print(f"{split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
