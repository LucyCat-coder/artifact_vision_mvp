# Artifact Vision MVP

MVP-система для первичного анализа исторических и археологических находок по фотографиям.

Система умеет:

- добавлять размеченные экспертами фотографии в базу эталонов;
- извлекать визуальные признаки предобученной моделью DINOv2;
- искать визуально похожие находки;
- предполагать тип изделия, материал, эпоху и диапазон датировки;
- обучать простые классификационные головы поверх сохранённых эмбеддингов;
- опционально получать структурированное описание видимых признаков через vision-модель в Ollama? Пока нет

## Почему Ollama не является основной моделью

«Память» о размеченных находках в этом MVP реализована не внутри LLM, а через:

1. визуальный энкодер DINOv2;
2. векторный индекс Qdrant;
3. поиск ближайших эталонов;
4. взвешенное голосование или обученные линейные классификаторы.

Так результат можно объяснить конкретными похожими эталонами, а добавление новой находки не требует повторного обучения большой модели.

## Архитектура

```text
                         ┌──────────────────────────┐
Новая фотография ──────►│ DINOv2 image encoder     │
                         │ 384-мерный embedding      │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          ┌───────────────────┐               ┌────────────────────┐
          │ Qdrant similarity │               │ Linear classifiers │
          │ top-K эталонов    │               │ type/material/era  │
          └─────────┬─────────┘               └──────────┬─────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Итог анализа             │
                         │ категории + уверенность  │
                         │ датировка + аналоги      │
                         └──────────────────────────┘
                                      │
                                      ▼ опционально
                         ┌──────────────────────────┐
                         │ Ollama vision model      │
                         │ наблюдаемые признаки     │
                         └──────────────────────────┘
```

## Структура проекта

```text
app/
  main.py                 FastAPI endpoints
  schemas.py              входные и выходные схемы
  config.py               настройки из .env
  services/
    encoder.py            DINOv2 embeddings
    vector_store.py       Qdrant local/server mode
    classifier.py         k-NN voting и linear heads
    ollama_vision.py      структурированный vision-анализ
    image_io.py           проверка и сохранение изображений
scripts/
  import_dataset.py       импорт CSV-датасета
  train_linear_heads.py   обучение классификаторов
tests/
streamlit_app.py          простой веб-интерфейс
data/dataset.example.csv  пример манифеста
```

## Быстрый запуск

Требуется Python 3.11 или 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

API будет доступно по адресу `http://localhost:8000`, Swagger — `http://localhost:8000/docs`.

При первом запуске будет загружен checkpoint `facebook/dinov2-small`.

### Streamlit UI

В другом терминале:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Интерфейс откроется на `http://localhost:8501`.

## Формат датасета

Подготовьте CSV по образцу `data/dataset.example.csv`.

Обязательна только колонка `image_path`, но для обучения нужны размеченные поля:

- `artifact_type` — тип изделия;
- `material` — материал;
- `era` — эпоха;
- `year_from`, `year_to` — диапазон датировки;
- `object_group_id` — ID физического предмета;
- `description`, `expert_source`, `tags` — дополнительная информация.

Годы до нашей эры задаются отрицательными значениями: например, `-300`.

Несколько фотографий одного предмета должны иметь одинаковый `object_group_id`. При последующей оценке качества все изображения одного предмета необходимо помещать только в train или только в validation/test, иначе возникнет утечка данных.

### Импорт

```bash
python scripts/import_dataset.py data/my_dataset.csv
```

После импорта API уже может работать в режиме поиска ближайших соседей.

### Обучение линейных классификаторов

```bash
python scripts/train_linear_heads.py
```

Для каждого поля создаётся отдельный файл:

```text
data/models/artifact_type.joblib
data/models/material.joblib
data/models/era.joblib
```

После обучения перезапустите API. Если файла классификатора нет, система автоматически использует взвешенное голосование ближайших эталонов.

## Ollama

Установите Ollama и загрузите vision-модель, например:

```bash
ollama pull gemma3:4b
```

В `.env`:

```env
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_VISION_MODEL=gemma3:4b
```

Ollama в этом проекте не определяет финальную эпоху как источник истины. Она описывает наблюдаемые признаки, а классификация опирается на экспертный датасет.

## Пример добавления эталона

```bash
curl -X POST http://localhost:8000/v1/artifacts \
  -F 'file=@example.jpg' \
  -F 'metadata={"title":"Фибула 001","artifact_type":"фибула","material":"бронза","era":"римский период","year_from":50,"year_to":200,"object_group_id":"object-001"}'
```

## Пример анализа

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -F 'file=@unknown.jpg' \
  -F 'top_k=7' \
  -F 'use_ollama=false'
```

Пример ответа:

```json
{
  "artifact_type": {
    "label": "фибула",
    "confidence": 0.84,
    "source": "nearest_neighbors",
    "evidence_count": 4,
    "alternatives": []
  },
  "material": {
    "label": "бронза",
    "confidence": 0.79,
    "source": "nearest_neighbors",
    "evidence_count": 3,
    "alternatives": []
  },
  "era": {
    "label": "римский период",
    "confidence": 0.74,
    "source": "nearest_neighbors",
    "evidence_count": 3,
    "alternatives": []
  },
  "dating": {
    "year_from": 70,
    "year_to": 210,
    "approximate_age_years": 1886,
    "confidence": 0.68,
    "evidence_count": 5
  },
  "similar_artifacts": [],
  "visual_observation": null,
  "warnings": []
}
```

## Что означает confidence

В режиме `nearest_neighbors` confidence является инженерной эвристикой, вычисленной по согласованности и визуальному сходству ближайших эталонов. Это не откалиброванная вероятность.

В режиме `linear_head` возвращается вероятность sklearn-классификатора, но её также необходимо отдельно калибровать на отложенной выборке перед использованием в экспертных процессах.

## Минимальные требования к данным

Для первого прототипа можно начать с 20–50 размеченных предметов на класс, но это не гарантирует хорошее качество. Важнее обеспечить:

- несколько ракурсов каждого объекта;
- одинаковые правила разметки;
- отсутствие противоречащих друг другу меток;
- разнообразие фона, освещения, масштаба и состояния предметов;
- отдельный тестовый набор физических объектов, отсутствующих в обучении;
- класс `неизвестно / недостаточно данных` и порог отказа от ответа.

## Следующий этап после MVP

1. Собрать baseline-метрики для каждого поля: macro-F1, top-k accuracy, confusion matrix.
2. Добавить crop/segmentation предмета, чтобы фон меньше влиял на embedding.
3. Дообучить DINOv2 или DINOv3 через adapters/LoRA либо обучить multi-head PyTorch-модель.
4. Использовать несколько фотографий одного объекта и агрегировать эмбеддинги.
5. Откалибровать confidence и реализовать отказ от классификации при низком сходстве.
6. Добавить экспертную обратную связь и версионирование разметки.

## Официальная документация

- DINOv2: https://github.com/facebookresearch/dinov2
- Transformers DINOv2: https://huggingface.co/docs/transformers/model_doc/dinov2
- Qdrant similarity search: https://qdrant.tech/documentation/search/search/
- Ollama vision: https://docs.ollama.com/capabilities/vision
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
