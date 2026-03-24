from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from config.models import RewardProfile, RlStageConfig
from training.contracts import StageContext
from training.rewards import RewardStack

from .common import multi_objective_aggregation, project_root, scale_rewards


def run_grpo_training(
    *,
    context: StageContext,
    config: RlStageConfig,
    reward_profile: RewardProfile | None,
    reward_stack: RewardStack,
    train_examples: list[Any],
) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "RL stage requires 'torch', 'datasets', 'transformers', 'peft', and 'trl' to be installed"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        context.experiment.model.tokenizer or context.experiment.model.base_model,
        trust_remote_code=True,
        revision=context.experiment.model.revision,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(
        [
            {
                "prompt": example.prompt,
                "answer": example.answer,
                "source_name": example.source_name,
                "split": example.split,
                "example_id": example.example_id,
                "metadata": json.dumps(example.metadata, ensure_ascii=False),
            }
            for example in train_examples
        ]
    )

    trainer_output_dir = context.output_dir / "trainer"
    trainer_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = context.output_dir / "checkpoint-final"
    checkpoint_source = resolve_checkpoint_source(context, config)

    model_arg: Any = checkpoint_source
    peft_config = None
    if is_adapter_checkpoint(checkpoint_source):
        model_arg = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_source,
            is_trainable=True,
            torch_dtype=torch_dtype(torch, context.experiment.model.dtype),
            trust_remote_code=True,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=context.experiment.model.adapter.rank,
            lora_alpha=context.experiment.model.adapter.alpha,
            lora_dropout=context.experiment.model.adapter.dropout,
            target_modules=list(context.experiment.model.adapter.target_modules),
        )

    training_args = GRPOConfig(
        output_dir=str(trainer_output_dir),
        run_name=f"{context.run_id}:{context.stage_name}",
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=report_to(context),
        seed=context.experiment.recipe.run.seed,
        remove_unused_columns=False,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        shuffle_dataset=config.shuffle_dataset,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        beta=config.beta,
        num_iterations=config.num_iterations,
        epsilon=config.epsilon,
        reward_weights=list(reward_stack.weights),
        scale_rewards=scale_rewards(config, reward_profile),
        multi_objective_aggregation=multi_objective_aggregation(config, reward_profile),
        log_completions=config.log_completions,
        model_init_kwargs=model_init_kwargs(torch, context),
    )

    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=list(reward_stack.functions),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    configure_wandb_env(context)
    train_result = trainer.train()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(checkpoint_dir)

    metrics = dict(train_result.metrics)
    metrics.setdefault("train_samples", len(train_examples))
    metrics.setdefault("reward_component_count", len(reward_stack.component_names))
    metrics.setdefault("uses_stub_reward", reward_stack.uses_stub)

    return {
        "metrics": metrics,
        "trainer_state": {
            "global_step": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_global_step": getattr(trainer.state, "best_global_step", None),
            "log_history": list(trainer.state.log_history),
        },
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_source": checkpoint_source,
    }


def resolve_checkpoint_source(context: StageContext, config: RlStageConfig) -> str:
    if config.checkpoint_source == "base_model":
        return context.experiment.model.base_model

    artifact = context.input_artifacts.get(config.checkpoint_source)
    if artifact is not None:
        return str(artifact.path)

    source_path = Path(config.checkpoint_source)
    if not source_path.is_absolute():
        source_path = project_root(context.experiment) / source_path
    if source_path.exists():
        return str(source_path)

    raise ValueError(
        f"Unsupported RL checkpoint_source '{config.checkpoint_source}'. "
        "Use 'base_model', an input artifact key, or an existing path."
    )


def is_adapter_checkpoint(checkpoint_source: str) -> bool:
    path = Path(checkpoint_source)
    return path.exists() and (path / "adapter_config.json").exists()


def model_init_kwargs(torch_module: Any, context: StageContext) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if context.experiment.model.revision is not None:
        kwargs["revision"] = context.experiment.model.revision

    dtype = torch_dtype(torch_module, context.experiment.model.dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def torch_dtype(torch_module: Any, dtype_name: str) -> Any | None:
    normalized = dtype_name.lower()
    mapping = {
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "float16": getattr(torch_module, "float16", None),
        "float32": getattr(torch_module, "float32", None),
    }
    return mapping.get(normalized)


def report_to(context: StageContext) -> str:
    mode = context.experiment.tracking.mode.lower()
    if mode in {"disabled", "none"}:
        return "none"
    if importlib.util.find_spec("wandb") is None:
        return "none"
    return "wandb"


def configure_wandb_env(context: StageContext) -> None:
    if report_to(context) != "wandb":
        return

    tracking = context.experiment.tracking
    os.environ.setdefault("WANDB_PROJECT", tracking.project)
    if tracking.entity:
        os.environ.setdefault("WANDB_ENTITY", tracking.entity)
    if tracking.mode.lower() == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
