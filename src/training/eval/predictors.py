from __future__ import annotations

from training.data import ReasoningExample
from training.eval.contracts import CheckpointPrediction, CheckpointRef


class VllmCheckpointPredictor:
    name = "vllm_checkpoint_predictor"

    def __init__(
        self,
        *,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        trust_remote_code: bool = True,
        generation_config: str = "vllm",
    ) -> None:
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self.generation_config = generation_config
        self._engine = None
        self._engine_key: tuple[str, str | None, str | None, str] | None = None

    def predict_many(
        self,
        checkpoint: CheckpointRef,
        examples: list[ReasoningExample],
    ) -> list[CheckpointPrediction]:
        llm, sampling_params, lora_request = self._prepare_runtime(checkpoint)
        outputs = llm.generate(
            [example.prompt for example in examples],
            sampling_params,
            lora_request=lora_request,
        )

        predictions: list[CheckpointPrediction] = []
        for example, output in zip(examples, outputs, strict=True):
            text = output.outputs[0].text.strip() if output.outputs else ""
            predictions.append(
                CheckpointPrediction(
                    example_id=example.example_id,
                    prompt=example.prompt,
                    target_answer=example.answer,
                    prediction=text,
                    raw_output=text,
                )
            )
        return predictions

    def _prepare_runtime(self, checkpoint: CheckpointRef):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "VllmCheckpointPredictor requires 'vllm' to be installed in a supported Linux GPU environment"
            ) from exc

        engine_key = (
            checkpoint.model_path,
            checkpoint.tokenizer_path,
            checkpoint.revision,
            self.dtype,
        )
        if self._engine is None or self._engine_key != engine_key:
            llm_kwargs = {
                "model": checkpoint.model_path,
                "tokenizer": checkpoint.tokenizer_path or checkpoint.model_path,
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enable_lora": checkpoint.adapter_path is not None,
                "generation_config": self.generation_config,
            }
            if checkpoint.revision is not None:
                llm_kwargs["revision"] = checkpoint.revision
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len

            self._engine = LLM(**llm_kwargs)
            self._engine_key = engine_key

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        lora_request = None
        if checkpoint.adapter_path is not None:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA checkpoint evaluation requires vLLM LoRA support"
                ) from exc
            lora_request = LoRARequest(
                checkpoint.name,
                1,
                checkpoint.adapter_path,
            )

        return self._engine, sampling_params, lora_request
