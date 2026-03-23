# Трекинг И Аналитика

## Основная идея

У системы должно быть два уровня наблюдаемости:

- `W&B` как основной внешний UI;
- `DuckDB/Parquet` как локальный слой аналитики.

Такой подход дает удобный мониторинг и одновременно не делает нас полностью зависимыми от внешнего сервиса.

## Что делает W&B

`W&B` используется как основной интерфейс для:

- мониторинга train loss и eval loss;
- просмотра reward curves;
- хранения config и run metadata;
- сравнения серий запусков;
- построения dashboard и sweep;
- таблиц с итоговыми метриками и ссылками на артефакты.

## Что хотим видеть в W&B

### На уровне одного run

- train loss;
- eval loss;
- reward curves;
- длину ответов;
- длину reasoning trace;
- success rate по reward constraints;
- таблицу итоговых eval-метрик;
- ссылки на checkpoints и артефакты.

### На уровне группы запусков

- сравнение SFT запусков между собой;
- сравнение reward packs;
- сравнение data mix;
- сравнение prompt/decode профилей;
- топ запусков по выбранной aggregate-метрике;
- фильтрацию по тегам и recipe.

### На уровне dashboard

Минимум хотим видеть:

- leaderboard по итоговому score;
- leaderboard по proxy reasoning suite;
- лучший SFT baseline;
- лучший RL run;
- сравнение run по compute budget;
- сравнение gain относительно baseline.

## Что делает локальный слой

Локальный слой нужен как страховка и как быстрый источник запросов без W&B UI.

Состав:

- `Parquet` как формат таблиц результатов и событий;
- `DuckDB` как локальный движок для запросов и сравнений.

## Минимальный набор локальных таблиц

- `runs`
- `run_tags`
- `stages`
- `metrics_summary`
- `metrics_timeseries`
- `artifacts`
- `checkpoints`

## Полезные локальные запросы

- лучшие run по конкретному eval suite;
- сравнение всех запусков с одним и тем же base checkpoint;
- сравнение reward packs при фиксированном data mix;
- сравнение data mix при фиксированном trainer;
- выборка run с деградацией после RL;
- топ run по отношению качества к compute.

## Что каждый run должен сохранять

Каждый run должен сохранять:

- `run_id`
- `parent_run_id`
- timestamp
- git commit
- resolved config
- список stage
- seed
- ссылки на входные и выходные checkpoint
- train metrics
- eval metrics
- reward summary
- путь к локальным артефактам
- ссылку на `W&B` run

Это позволит строить понятный lineage и сопоставлять ветки экспериментов между собой.
