# Стадия SFT

## Назначение

`sft` отвечает за основной контур adapter-based fine-tuning. Для текущей задачи это базовая обучающая стадия, от которой мы ожидаем основной устойчивый прирост перед RL.

## Что входит в scope

- LoRA или QLoRA;
- работа от готового base checkpoint;
- обучение на reasoning-oriented dataset mix;
- сохранение checkpoint для последующего eval и RL.

## Что пока не входит в scope

- full finetune всех параметров;
- pretraining;
- сложные многоэтапные curriculum-схемы;
- избыточная платформенная логика.

## Входы

- base checkpoint;
- data config;
- prompt config, если формат обучения зависит от шаблона;
- train config;
- runtime config;
- tracking config.

## Выходы

- SFT checkpoint;
- train/eval metrics;
- summary по dataset mix;
- summary по compute;
- артефакт для следующего `final_eval` или `rl`.

## Что важно логировать

- train loss;
- eval loss;
- learning rate;
- token throughput;
- effective batch size;
- длину последовательностей;
- стоимость по шагам и по эпохам;
- метрики на validation proxy-set.

## Что хотим видеть в результате

После этой стадии должно быть понятно:

- какой data mix реально помогает;
- какие train-параметры устойчивы;
- какой checkpoint стоит считать лучшим кандидатом для RL;
- стоит ли вообще идти в RL или SFT уже дал достаточный прирост.
