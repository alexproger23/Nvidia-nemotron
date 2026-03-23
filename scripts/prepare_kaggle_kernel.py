from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "kaggle" / "kernel-config.json"
BUILD_DIR = ROOT / "build" / "kaggle_kernel"
SOURCE_DIR = ROOT / "src"

KAGGLE_MAIN = """from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from competition.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--input-dir",
                "/kaggle/input/__COMPETITION_SLUG__",
                "--output-file",
                "/kaggle/working/submission.csv",
            ]
        )
    )
"""


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    competition_slug = config.get("competition_slug", "").strip()
    if not competition_slug:
        raise ValueError("kaggle/kernel-config.json: 'competition_slug' must be set")

    username = config.get("username", "").strip()
    kernel_slug = config.get("kernel_slug", "").strip()
    if not username or not kernel_slug:
        raise ValueError("kaggle/kernel-config.json: 'username' and 'kernel_slug' must be set")

    config["competition_sources"] = [competition_slug]
    config["id"] = f"{username}/{kernel_slug}"
    config["code_file"] = "main.py"
    return config


def recreate_build_dir() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def copy_sources() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError("src directory not found")
    shutil.copytree(SOURCE_DIR, BUILD_DIR / "src")


def write_main_file(competition_slug: str) -> None:
    main_code = KAGGLE_MAIN.replace(
        "__COMPETITION_SLUG__",
        competition_slug,
    )
    (BUILD_DIR / "main.py").write_text(main_code, encoding="utf-8")


def write_metadata(config: dict) -> None:
    metadata = {
        "id": config["id"],
        "title": config["title"],
        "code_file": config["code_file"],
        "language": config["language"],
        "kernel_type": config["kernel_type"],
        "is_private": config["is_private"],
        "enable_gpu": config["enable_gpu"],
        "enable_internet": config["enable_internet"],
        "dataset_sources": config.get("dataset_sources", []),
        "competition_sources": config.get("competition_sources", []),
        "kernel_sources": config.get("kernel_sources", []),
        "model_sources": config.get("model_sources", []),
    }
    (BUILD_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    config = load_config()
    recreate_build_dir()
    copy_sources()
    write_main_file(config["competition_slug"])
    write_metadata(config)
    print(f"Kaggle kernel prepared in: {BUILD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
