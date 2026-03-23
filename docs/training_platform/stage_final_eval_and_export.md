# Стадии Final Eval И Export

## Final Eval

### Назначение

`final_eval` нужна для честного и унифицированного сравнения кандидатов после baseline, SFT и RL.

Она должна быть единой точкой истины для выбора лучшего run.

### Входы

- candidate checkpoint;
- eval config;
- prompt profile;
- decoding profile;
- tracking config.

### Выходы

- финальная таблица метрик;
- aggregate score;
- сравнение с baseline и parent run;
- shortlist лучших checkpoint.

### Что важно логировать

- итоговые benchmark-метрики;
- proxy reasoning metrics;
- aggregate score;
- дельту относительно baseline;
- дельту относительно parent checkpoint;
- compute-to-quality summary.

## Export

### Назначение

`export` подготавливает итоговый артефакт для инференса, последующей валидации или сабмита в контур соревнования.

### Входы

- лучший checkpoint;
- export policy;
- runtime compatibility requirements.

### Выходы

- готовый export artifact;
- manifest экспортированных файлов;
- метаданные о совместимости;
- ссылка на родительский run.

## Что хотим видеть в результате

После `final_eval` и `export` должно быть понятно:

- какой checkpoint признан лучшим;
- почему он выбран;
- какой именно артефакт должен идти дальше в инференсный контур.
