# Система Конфигов

## Задача

Конфиги должны работать как конструктор: мы должны менять отдельные части эксперимента независимо друг от друга, не переписывая весь запуск целиком.

Ожидаемый сценарий:

- меняем только `reward`, не трогая `model` и `train`;
- меняем только `eval pack`, не трогая обучение;
- запускаем одну и ту же `recipe` с разными `data mix`;
- переиспользуем один `model config` в нескольких экспериментах.

## Домены конфигов

Конфиг не должен быть одним огромным YAML-файлом. Он должен быть разложен по доменам.

### Model config

Содержит:

- базовый checkpoint;
- формат весов;
- tokenizer;
- LoRA/QLoRA параметры;
- длину контекста;
- настройки reasoning mode, если применимо;
- ограничения на память и тип загрузки.

### Data config

Содержит:

- список источников данных;
- правила фильтрации;
- правила дедупликации;
- микс датасетов и веса;
- шаблон преобразования в training format;
- synthetic data policy;
- train/val split policy.

### Prompt config

Содержит:

- system prompt;
- chat template;
- формат reasoning trace;
- формат финального ответа;
- stop tokens;
- decoding profile для baseline и eval.

### Train config

Содержит:

- learning rate;
- scheduler;
- batch size;
- gradient accumulation;
- max steps или epochs;
- warmup;
- save/eval cadence;
- optimizer;
- clipping;
- checkpoint policy.

### Reward config

Reward должен быть отдельной модульной сущностью, а не набором случайных флагов внутри RL trainer.

Reward pack должен описывать:

- список reward-компонентов;
- веса компонентов;
- нормализацию;
- clipping;
- hard constraints;
- агрегацию в final reward.

Примеры reward-компонентов:

- `format_valid`
- `answer_extractable`
- `verifiable_correctness`
- `step_budget_penalty`
- `verbosity_penalty`
- `reasoning_style_bonus`

### Eval config

Eval pack должен описывать:

- набор benchmark или proxy tasks;
- prompt profile;
- decoding profile;
- метрики;
- правила агрегации;
- таблицу финальных ключевых метрик.

### Runtime config

Содержит:

- hardware profile;
- distributed backend;
- precision;
- seed;
- resume policy;
- paths для cache и артефактов;
- ограничения по времени и ресурсам.

### Tracking config

Содержит:

- `wandb_project`
- `wandb_entity`
- теги запуска;
- режим логирования;
- частоту логирования;
- политику логирования checkpoint и таблиц;
- локальные пути для `Parquet` и `DuckDB`.

### Recipe config

Recipe определяет:

- список стадий;
- порядок стадий;
- какие артефакты передаются дальше;
- политику resume;
- правила именования run и stage.

## Что хотим получать на выходе

Каждый запуск должен сохранять `resolved config`, чтобы всегда было понятно:

- из каких модулей он собран;
- какие значения были вычислены после merge;
- чем именно один запуск отличался от другого.

## Принцип минимального v1

На первом этапе не нужен перегруженный конфиг-комбайн. Достаточно, чтобы система уверенно компоновала:

- `recipe`
- `model`
- `data`
- `train`
- `eval`
- `tracking`

После этого уже можно добавлять более сложные `reward` и `runtime` профили.
