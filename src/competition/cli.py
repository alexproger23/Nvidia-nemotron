from __future__ import annotations

import argparse
from pathlib import Path

from competition.pipeline import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run competition inference locally or on Kaggle.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with competition input files.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("submission.csv"),
        help="Path where submission.csv should be written.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional path to model weights/checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_inference(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_dir=args.model_dir,
    )
    print(f"Submission written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
