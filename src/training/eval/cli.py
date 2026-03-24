from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.eval.validation import (
    DEFAULT_BOXED_ANSWER_HINT,
    DEFAULT_KAGGLE_MODEL,
    build_checkpoint_ref,
    load_experiment,
    resolve_base_model_path,
    resolve_metric_stack,
    resolve_validation_examples,
    run_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation for a base model or checkpoint with selected metrics.")
    parser.add_argument("--recipe", default="rl_bootstrap", help="Recipe used to resolve model/data/tracking defaults.")
    parser.add_argument("--config-root", type=Path, default=Path("config"), help="Path to config root.")
    parser.add_argument("--data-profile", default=None, help="Optional data profile override.")
    parser.add_argument("--metric-profile", default=None, help="Optional metric profile override.")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=[],
        help="Metric component name. Can be passed multiple times for ad-hoc selection.",
    )
    parser.add_argument("--validation-path", type=Path, default=None, help="Optional explicit validation dataset path.")
    parser.add_argument(
        "--validation-format",
        default=None,
        help="Optional validation dataset format override, for example jsonl or csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out/validation_eval"), help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for validation examples.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")

    parser.add_argument(
        "--model-source",
        choices=("auto", "kaggle", "hf", "local"),
        default="auto",
        help="How to resolve the base model.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Explicit local base model path.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional adapter or full model checkpoint path to validate.",
    )
    parser.add_argument(
        "--kaggle-model",
        default=DEFAULT_KAGGLE_MODEL,
        help="Kaggle model slug or full handle. Default: metric/nemotron-3-nano-30b-a3b-bf16",
    )
    parser.add_argument("--hf-model-id", default=None, help="Optional Hugging Face fallback model id.")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=3584)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-lora-rank", type=int, default=None)

    parser.set_defaults(use_chat_template=True, enable_thinking=True, extract_answers=True)
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--no-extract-answers", dest="extract_answers", action="store_false")
    parser.add_argument(
        "--answer-format-hint",
        default=DEFAULT_BOXED_ANSWER_HINT,
        help="Instruction appended to every prompt before generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    experiment = load_experiment(
        recipe_name=args.recipe,
        config_root=args.config_root,
        data_profile=args.data_profile,
        metric_profile=args.metric_profile,
    )
    project_root = Path.cwd()
    examples = resolve_validation_examples(
        experiment,
        validation_path=args.validation_path,
        validation_format=args.validation_format,
        project_root=project_root,
    )
    metric_stack = resolve_metric_stack(experiment, metric_names=args.metrics)
    base_model_path, resolved_model_source = resolve_base_model_path(
        experiment,
        model_source=args.model_source,
        model_path=args.model_path,
        kaggle_model=args.kaggle_model,
        hf_model_id=args.hf_model_id,
    )
    checkpoint = build_checkpoint_ref(
        experiment,
        base_model_path=base_model_path,
        checkpoint_path=args.checkpoint_path,
    )

    max_lora_rank = args.max_lora_rank
    if max_lora_rank is None and checkpoint.adapter_path is not None:
        max_lora_rank = experiment.model.adapter.rank

    predictor_kwargs = {
        "dtype": experiment.model.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_num_seqs": args.max_num_seqs,
        "max_lora_rank": max_lora_rank,
        "trust_remote_code": True,
        "generation_config": "vllm",
        "answer_format_hint": args.answer_format_hint,
        "use_chat_template": args.use_chat_template,
        "enable_thinking": args.enable_thinking,
        "extract_answers": args.extract_answers,
    }

    result = run_validation(
        experiment=experiment,
        metric_stack=metric_stack,
        checkpoint=checkpoint,
        examples=examples,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        predictor_kwargs=predictor_kwargs,
        project_root=project_root,
        include_data_profile_summary=args.validation_path is None,
    )

    summary = {
        "resolved_model_source": resolved_model_source,
        "base_model_path": base_model_path,
        "checkpoint_name": checkpoint.name,
        "metric_components": list(metric_stack.component_names),
        "metrics": result["metrics"],
        "manifest_path": str(result["manifest_path"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
