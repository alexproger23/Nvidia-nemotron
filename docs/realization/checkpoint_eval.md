# Checkpoint Eval

## Что теперь является основой eval

Вместо отдельной `baseline_eval` stage в коде теперь есть общий слой `training.eval`:

- `CheckpointRef`
- `CheckpointPredictor`
- `CheckpointEvaluator`
- `VllmCheckpointPredictor`

Смысл теперь такой:

`checkpoint -> predictor -> predictions -> metrics -> artifacts`

## Что это дает

- base model и fine-tuned checkpoint оцениваются одинаково;
- eval больше не привязан к слову `baseline`;
- stage-обертки можно будет вернуть позже, но поверх уже общего evaluator;
- inference backend можно менять без переписывания data flow.

## Текущий статус

Реализован общий evaluator и `vLLM`-based predictor с lazy import:

- нужен `vllm`
- для LoRA adapter используется `LoRARequest`
- ориентир по запуску: Linux + GPU

## Вывод

Теперь baseline в архитектуре означает не отдельный тип stage, а просто `CheckpointRef` на base checkpoint.
