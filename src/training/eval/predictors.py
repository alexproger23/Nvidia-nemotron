from __future__ import annotations

from pathlib import Path

from training.data import ReasoningExample
from training.eval.contracts import CheckpointPrediction, CheckpointRef
from training.metrics import extract_final_answer


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
        max_num_seqs: int | None = None,
        max_lora_rank: int | None = None,
        trust_remote_code: bool = True,
        generation_config: str = "vllm",
        answer_format_hint: str | None = None,
        use_chat_template: bool = False,
        enable_thinking: bool = False,
        extract_answers: bool = False,
    ) -> None:
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_num_seqs = max_num_seqs
        self.max_lora_rank = max_lora_rank
        self.trust_remote_code = trust_remote_code
        self.generation_config = generation_config
        self.answer_format_hint = answer_format_hint
        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking
        self.extract_answers = extract_answers
        self._engine = None
        self._engine_key: tuple[str, str | None, str | None, str] | None = None

    def predict_many(
        self,
        checkpoint: CheckpointRef,
        examples: list[ReasoningExample],
    ) -> list[CheckpointPrediction]:
        llm, sampling_params, lora_request = self._prepare_runtime(checkpoint)
        prompts = self._build_prompts(llm, examples)
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )

        predictions: list[CheckpointPrediction] = []
        for example, output in zip(examples, outputs, strict=True):
            text = output.outputs[0].text.strip() if output.outputs else ""
            prediction = extract_final_answer(text) if self.extract_answers else text
            predictions.append(
                CheckpointPrediction(
                    example_id=example.example_id,
                    prompt=example.prompt,
                    target_answer=example.answer,
                    prediction=prediction,
                    raw_output=text,
                )
            )
        return predictions

    def _build_prompts(self, llm, examples: list[ReasoningExample]) -> list[str]:
        prompts: list[str] = []
        tokenizer = None
        if self.use_chat_template:
            try:
                tokenizer = llm.get_tokenizer()
            except Exception:
                tokenizer = None

        for example in examples:
            user_content = example.prompt.strip()
            if self.answer_format_hint:
                user_content = f"{user_content}\n{self.answer_format_hint}"

            if tokenizer is None:
                prompts.append(user_content)
                continue

            prompt = self._apply_chat_template(tokenizer, user_content)
            prompts.append(prompt)
        return prompts

    def _apply_chat_template(self, tokenizer, user_content: str) -> str:
        message = [{"role": "user", "content": user_content}]

        if self.enable_thinking:
            try:
                return tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                pass
            except Exception:
                return user_content

        try:
            return tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return user_content

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
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            }
            if checkpoint.revision is not None:
                llm_kwargs["revision"] = checkpoint.revision
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len
            if self.max_num_seqs is not None:
                llm_kwargs["max_num_seqs"] = self.max_num_seqs
            if checkpoint.adapter_path is not None and self.max_lora_rank is not None:
                llm_kwargs["max_lora_rank"] = self.max_lora_rank

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
                Path(checkpoint.adapter_path).name,
                1,
                checkpoint.adapter_path,
            )

        return self._engine, sampling_params, lora_request
