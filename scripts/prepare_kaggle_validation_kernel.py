from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "kaggle" / "validation-kernel-config.json"
BUILD_DIR = ROOT / "build" / "kaggle_validation_kernel"
PROJECT_PATHS = (
    ("src", "src"),
    ("config", "config"),
    ("data/reasoning", "data/reasoning"),
)

KAGGLE_MAIN_TEMPLATE = """from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(__ARGS__))
"""


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    username = config.get("username", "").strip()
    kernel_slug = config.get("kernel_slug", "").strip()
    title = config.get("title", "").strip()
    if not username or not kernel_slug or not title:
        raise ValueError("validation-kernel-config.json must define username, kernel_slug, and title")

    config["id"] = f"{username}/{kernel_slug}"
    config["code_file"] = "main.py"
    return config


def recreate_build_dir() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def copy_project_files() -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for source_relative, target_relative in PROJECT_PATHS:
        source_path = ROOT / source_relative
        if source_path.exists():
            shutil.copytree(source_path, BUILD_DIR / target_relative, ignore=ignore)


def build_validation_args(config: dict) -> list[str]:
    args = [
        "--recipe",
        config.get("recipe", "rl_bootstrap"),
        "--config-root",
        "config",
        "--output-dir",
        config.get("output_dir", "/kaggle/working/validation_eval"),
        "--model-source",
        config.get("model_source", "auto"),
        "--kaggle-model",
        config.get("kaggle_model", "metric/nemotron-3-nano-30b-a3b-bf16"),
        "--max-model-len",
        str(config.get("max_model_len", 4096)),
        "--max-new-tokens",
        str(config.get("max_new_tokens", 3584)),
        "--temperature",
        str(config.get("temperature", 1.0)),
        "--top-p",
        str(config.get("top_p", 1.0)),
        "--gpu-memory-utilization",
        str(config.get("gpu_memory_utilization", 0.85)),
        "--tensor-parallel-size",
        str(config.get("tensor_parallel_size", 1)),
        "--max-num-seqs",
        str(config.get("max_num_seqs", 128)),
    ]
    if config.get("data_profile"):
        args.extend(["--data-profile", config["data_profile"]])
    if config.get("metric_profile"):
        args.extend(["--metric-profile", config["metric_profile"]])
    for metric_name in config.get("metrics", []):
        args.extend(["--metric", metric_name])
    if config.get("validation_path"):
        args.extend(["--validation-path", config["validation_path"]])
    if config.get("validation_format"):
        args.extend(["--validation-format", config["validation_format"]])
    if config.get("hf_model_id"):
        args.extend(["--hf-model-id", config["hf_model_id"]])
    if config.get("model_path"):
        args.extend(["--model-path", config["model_path"]])
    if config.get("checkpoint_path"):
        args.extend(["--checkpoint-path", config["checkpoint_path"]])
    if config.get("max_samples") is not None:
        args.extend(["--max-samples", str(config["max_samples"])])
    if config.get("seed") is not None:
        args.extend(["--seed", str(config["seed"])])
    if config.get("max_lora_rank") is not None:
        args.extend(["--max-lora-rank", str(config["max_lora_rank"])])
    if config.get("answer_format_hint"):
        args.extend(["--answer-format-hint", config["answer_format_hint"]])
    if config.get("use_chat_template") is False:
        args.append("--no-chat-template")
    if config.get("enable_thinking") is False:
        args.append("--disable-thinking")
    if config.get("extract_answers") is False:
        args.append("--no-extract-answers")
    return args


def write_main_file(validation_args: list[str]) -> None:
    main_code = KAGGLE_MAIN_TEMPLATE.replace("__ARGS__", repr(validation_args))
    (BUILD_DIR / "main.py").write_text(main_code, encoding="utf-8")


def write_metadata(config: dict) -> None:
    metadata = {
        "id": config["id"],
        "title": config["title"],
        "code_file": config["code_file"],
        "language": config.get("language", "python"),
        "kernel_type": config.get("kernel_type", "script"),
        "is_private": config.get("is_private", True),
        "enable_gpu": config.get("enable_gpu", True),
        "enable_internet": config.get("enable_internet", True),
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
    copy_project_files()
    write_main_file(build_validation_args(config))
    write_metadata(config)
    print(f"Kaggle validation kernel prepared in: {BUILD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
