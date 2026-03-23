# NvidiaKaggle

Шаблон репозитория для запуска кода на Kaggle Notebooks/Kernels и отправки сабмита в соревнование.

Текущая настройка уже привязана к соревнованию `nvidia-nemotron-model-reasoning-challenge`.

## Что уже подготовлено

- единая точка входа `python -m nvidia_kaggle.cli`
- локальный запуск и запуск на Kaggle через один и тот же код
- сборка папки `build/kaggle_kernel` для `kaggle kernels push`
- шаблон конфига `kaggle/kernel-config.json`
- PowerShell-скрипт для подготовки и отправки kernel на Kaggle

## Структура

```text
.
|-- kaggle/
|   `-- kernel-config.json
|-- scripts/
|   |-- prepare_kaggle_kernel.py
|   `-- submit_kaggle.ps1
`-- src/
    `-- nvidia_kaggle/
        |-- cli.py
        `-- pipeline.py
```

## 1. Настройка Kaggle API

1. Установите CLI:

```powershell
pip install kaggle
```

2. На странице `https://www.kaggle.com/settings` создайте API token.
3. Сохраните скачанный `kaggle.json` в:

```text
C:\Users\<ваш_пользователь>\.kaggle\kaggle.json
```

4. Перед первым запуском примите правила нужного соревнования в браузере Kaggle, иначе API не даст скачать данные или сделать submit.

Если `kaggle.json` на Windows работает нестабильно, можно использовать переменные окружения через подготовленный скрипт:

```powershell
.\scripts\set_kaggle_env.ps1 -Username alexproger23 -Key KGAT_ВАШ_ПОЛНЫЙ_КЛЮЧ
```

Если хотите сохранить их в user environment Windows:

```powershell
.\scripts\set_kaggle_env.ps1 -Username alexproger23 -Key KGAT_ВАШ_ПОЛНЫЙ_КЛЮЧ -Persist
```

## 2. Заполните конфиг соревнования

Откройте `kaggle/kernel-config.json` и заполните:

- `username`: ваш Kaggle username
- `competition_slug`: slug соревнования
- `kernel_slug`: slug notebook/kernel
- `title`: человекочитаемое название kernel
- `enable_gpu`: нужен ли GPU
- `enable_internet`: нужен ли интернет
- `dataset_sources`: дополнительные dataset-источники, если нужны
- `kernel_sources`: если хотите подключить output другого notebook
- `model_sources`: если используете Kaggle Models

## 3. Скачайте данные соревнования

```powershell
.\scripts\download_competition.ps1
```

После этого распакуйте архивы в папку, удобную для локального запуска.

Если для этого соревнования `kaggle competitions files` или `kaggle competitions download` возвращает `401`, но в UI Kaggle вы уже можете смотреть данные, создавать notebooks и делать submit, это не блокирует запуск kernel на стороне Kaggle. В таком случае используйте данные напрямую внутри Kaggle kernel и проверяйте формат `sample_submission` на вкладке `Data` в браузере.

## 4. Реализуйте свою логику инференса

Основное место для вашей логики:

- `src/nvidia_kaggle/pipeline.py`

Нужно заменить заглушку в функции `run_inference(...)` на реальный код:

- загрузка модели
- чтение test/sample_submission
- предсказание
- сохранение `submission.csv`

## 5. Локальный прогон

```powershell
$env:PYTHONPATH="src"
python -m nvidia_kaggle.cli --input-dir data\competition --output-file out\submission.csv
```

Если `sample_submission.csv` найден, текущая заглушка просто скопирует его в output. Это нужно только чтобы проверить пайплайн до внедрения модели.

## 6. Подготовка папки для Kaggle

```powershell
python scripts\prepare_kaggle_kernel.py
```

Скрипт создаст:

```text
build/kaggle_kernel/
|-- kernel-metadata.json
|-- main.py
`-- src/...
```

## 7. Отправка на Kaggle server

```powershell
.\scripts\submit_kaggle.ps1
```

Или вручную:

```powershell
python scripts\prepare_kaggle_kernel.py
kaggle kernels push -p build\kaggle_kernel
```

После этого kernel запустится на Kaggle server c competition source `nvidia-nemotron-model-reasoning-challenge`. Когда ран завершится, сабмит обычно можно сделать из UI Kaggle, если соревнование принимает notebook submissions. Если соревнование требует файл-сабмит, скачайте `submission.csv` из output kernel и отправьте:

```powershell
kaggle competitions submit -c <competition-slug> -f submission.csv -m "baseline"
```

## 8. Как это работает на Kaggle

Подготовленный `main.py` внутри kernel:

- ищет входные данные в `/kaggle/input/<competition-slug>`
- пишет результат в `/kaggle/working/submission.csv`
- использует тот же модуль `src/nvidia_kaggle/pipeline.py`, что и локально

Это значит, что локальный и серверный сценарии остаются одинаковыми.

## Ограничения и практические замечания

- Если нужных Python-пакетов нет в стандартном образе Kaggle, их придется либо ставить через internet-enabled kernel, либо вендорить в репозиторий/датасет.
- Если модель весит много, удобнее хранить веса как отдельный Kaggle Dataset и подключать через `dataset_sources`.
- Если соревнование использует notebook-only submit, лучше держать итоговый `submission.csv` в `/kaggle/working/`.
- Я не смог надежно вычитать правила именно этого соревнования с Kaggle-страницы в веб-просмотре из-за антибот-защиты, поэтому формат финального submission файла и ограничения kernel submissions нужно будет проверить уже в UI конкурса.

## Источники

- Official Kaggle CLI: https://github.com/Kaggle/kaggle-api
- Kernel metadata fields: https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata
