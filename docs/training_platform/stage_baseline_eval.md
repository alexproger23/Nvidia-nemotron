# Стадия Baseline Eval

## Назначение

`baseline_eval` нужна для того, чтобы понять исходный уровень модели до обучения и быстро отсеять слабые prompt/decode конфигурации.

Для Nemotron Challenge это особенно важно, потому что часть прироста можно получить без обучения, только за счет:

- выбора prompt profile;
- reasoning mode;
- thinking budget;
- decoding settings;
- формата финального ответа.

## Входы

- base checkpoint;
- prompt config;
- eval config;
- runtime config;
- optional data subset для proxy-eval.

## Выходы

- таблица baseline-метрик;
- сравнение prompt/decode профилей;
- shortlist лучших baseline-конфигураций;
- артефакт, на который могут опираться следующие стадии.

## Что важно логировать

- accuracy или task-specific metrics;
- aggregate score;
- длину reasoning trace;
- длину финального ответа;
- latency и throughput;
- ошибки парсинга ответа;
- долю extractable answers.

## Что хотим видеть в результате

После этой стадии должно быть понятно:

- какая конфигурация baseline сильнее;
- какие режимы модели слишком дорогие по compute;
- какие prompt/decode профили стоит перенести в `sft` и `final_eval`.
