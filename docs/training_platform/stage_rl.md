# Стадия RL

## Назначение

`rl` отвечает за reward-driven дообучение модели через `GRPO`.

Стадия может запускаться:

- поверх `base_model`, если нужен самостоятельный RL-эксперимент без SFT;
- поверх входного checkpoint-артефакта, если перед этим уже была другая training stage;
- поверх явного локального пути к checkpoint.

`rl` не зависит от `sft` на уровне контракта recipe и может использоваться как самодостаточная стадия.

## Текущая реализация

Bootstrap-версия построена на `trl.GRPOTrainer` и использует:

- `model` profile для base model и LoRA-параметров;
- `data` profile для train split;
- `reward` profile как список reward-компонентов;
- `metric` profile как список metric-компонентов;
- `tracking` profile для output root и режима логирования.

Если `reward.components = []`, stage подставляет нулевой stub-reward.
Это позволяет собрать и прогнать RL pipeline до появления реальных reward-функций.

## Входы

- checkpoint source: `base_model`, artifact key или локальный путь;
- reward profile;
- data profile;
- RL train config из `recipe.stages.rl`;
- tracking config.

## Выходы

- RL checkpoint;
- `metrics_summary.json`;
- `trainer_state.json`;
- `dataset_summary.json`;
- `reward_manifest.json`;
- `metric_manifest.json`;
- checkpoint artifact для следующей стадии.

## Reward pack

Reward profile хранит:

- список компонентов;
- веса компонентов;
- режим `scale_rewards`;
- режим `multi_objective_aggregation`.

Пример:

```toml
name = "some_reward_pack"
scale_rewards = "none"
multi_objective_aggregation = "sum_then_normalize"

[[components]]
name = "format_valid"
weight = 1.0

[components.params]
some_flag = true
```

Инфраструктура уже поддерживает произвольное количество компонентов и отдельный вес для каждого, но project-specific reward-функции ещё не реализованы.

## Metric pack

Metric profile хранит:

- список metric-компонентов;
- параметры для каждой metric-функции;
- имена метрик, которые должны логироваться в `GRPOTrainer` и W&B.

Metric pack не участвует в оптимизации и не влияет на reward/advantages.
Он нужен для proxy-мониторинга на тех же prompt/completion, которые уже генерирует RL stage.

## Что важно логировать

- train metrics из `GRPOTrainer`;
- число reward-компонентов;
- число metric-компонентов;
- флаг использования stub-reward;
- curves вида `metrics/<name>/mean`, `metrics/<name>/std`, `metrics/<name>/coverage`;
- источник checkpoint;
- summary по train dataset.

## Что должно быть понятно после стадии

- можно ли запускать RL независимо от SFT;
- какой reward pack реально использовался;
- из какого checkpoint стартовало обучение;
- какие артефакты нужно передать дальше в eval/export.
