# RL Stage

## Что добавлено

- новый домен конфигов `reward`;
- новый `recipe.stages.rl`;
- stage-реализация [src/training/stages/rl/stage.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\stages\rl\stage.py);
- reward registry [src/training/rewards/registry.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\rewards\registry.py);
- пользовательские reward-функции [src/training/rewards/functions.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\rewards\functions.py);
- bootstrap recipe [config/recipe/rl_bootstrap.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\recipe\rl_bootstrap.toml);
- stub reward profile [config/reward/rl_stub_v1.toml](c:\Users\Александр\gitprojects\NvidiaKaggle\config\reward\rl_stub_v1.toml).

## Архитектурные решения

- RL построен на `trl.GRPOTrainer`, потому что он уже умеет принимать список `reward_funcs` и `reward_weights`.
- Reward pack вынесен в отдельный домен конфигов, чтобы менять reward независимо от model/data/tracking.
- Stage допускает пустой `components`, подставляя zero-reward stub. Это нужно, чтобы сначала собрать стабильную инфраструктуру.
- `checkpoint_source` умеет ссылаться на `base_model`, входной artifact key или локальный путь. Поэтому RL stage самодостаточен и не требует `sft`.

## Куда писать свои reward-функции

Пользовательские reward-функции нужно добавлять в [src/training/rewards/functions.py](c:\Users\Александр\gitprojects\NvidiaKaggle\src\training\rewards\functions.py).

Этот файл специально выделен под ручное редактирование:

- добавляешь туда новую функцию;
- регистрируешь её в `register_user_rewards`;
- указываешь имя функции в `config/reward/*.toml`.

## Артефакты stage

RL stage сохраняет:

- `resolved_stage_config.json`
- `dataset_summary.json`
- `reward_manifest.json`
- `metrics_summary.json`
- `trainer_state.json`
- `checkpoint-final/`

## Ограничения текущей версии

- baseline/sft/final_eval stage-реализации пока не подключены в default registry;
- project-specific reward functions ещё не реализованы;
- для запуска нужны внешние зависимости `torch`, `datasets`, `transformers`, `peft`, `trl`.
