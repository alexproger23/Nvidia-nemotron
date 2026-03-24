# Модуль Работы С Данными

## Что добавлено

- `src/training/data/competition.py`
  - обнаружение raw Kaggle source из директории или `.zip`
  - чтение `train.csv` и `test.csv`
  - summary по raw split
- `src/training/data/reasoning.py`
  - чтение нормализованных reasoning sources из `jsonl/csv`
  - загрузка `train` и `validation` по `DataProfile`
  - запись `jsonl`
- `src/training/data/preparation.py`
  - подготовка `data/reasoning/train.jsonl`
  - подготовка `data/reasoning/validation.jsonl`
  - стабильный split по seed

## Поток данных

1. Скачиваем raw Kaggle archive в `data/raw`
2. Запускаем:

```powershell
python scripts\prepare_reasoning_data.py
```

3. Получаем:

```text
data/reasoning/train.jsonl
data/reasoning/validation.jsonl
```

4. Эти файлы уже соответствуют `config/data/reasoning_mixture_v1.toml`

## Принцип

Слой данных разделен на два контура:

- `competition raw` для файлов соревнования;
- `reasoning normalized` для обучения и eval.

Это позволяет не тащить Kaggle-специфику в будущие stage реализации.
