# Начальный Каркас Платформы

## Что есть

- `src/config`:
  - [models.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\config\models.py)
  - [registry.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\config\registry.py)
  - [loader.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\config\loader.py)
- `src/training`:
  - [contracts.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\contracts.py)
  - [registry.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\registry.py)
  - [runner.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\runner.py)
  - `eval/`
    - [contracts.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\eval\contracts.py)
    - [evaluator.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\eval\evaluator.py)
    - [predictors.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\eval\predictors.py)
    - [artifacts.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\eval\artifacts.py)
- `src/competition`:
  - [cli.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\competition\cli.py)
  - [pipeline.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\competition\pipeline.py)

## Конфиги

Текущие домены:

- `recipe`
- `model`
- `data`
- `tracking`

Текущие файлы:

- [config/recipe/baseline_sft_final.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\recipe\baseline_sft_final.toml)
- [config/model/nemotron_lora_bootstrap.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\model\nemotron_lora_bootstrap.toml)
- [config/data/reasoning_mixture_v1.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\data\reasoning_mixture_v1.toml)
- [config/tracking/default_offline.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\tracking\default_offline.toml)

`recipe` хранит:

- `refs`
- `run`
- `stages`

`ConfigLoader`:

- грузит recipe;
- подтягивает доменные профили;
- применяет dot-path overrides;
- возвращает `ResolvedExperiment`.

Пример:

```python
from config import ConfigLoader

experiment = ConfigLoader().resolve(
    "baseline_sft_final",
    overrides={"run.seed": 7},
)
```

## Контракты

`training/contracts.py`:

- `ArtifactRef`
- `StageContext`
- `StageResult`
- `RunResult`
- `Stage`

`training/registry.py`:

- `StageRegistry` связывает имя стадии и реализацию

`training/runner.py`:

- идет по enabled stages;
- создает output dir стадии;
- передает артефакты дальше;
- собирает `RunResult`

## Что проверено

- `ConfigLoader().resolve("baseline_sft_final")`
- override'ы
- сериализация `ResolvedExperiment`
- dry-run `RecipeRunner`
- импорты после переноса на плоскую структуру `src/`

## Что дальше

- stage-обертки поверх `training.eval`
- реальный model-backed checkpoint eval
- CLI для запуска recipe
