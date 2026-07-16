#!/usr/bin/env python3
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.config import get_settings
from app.services.classifier import CATEGORICAL_FIELDS
from app.services.container import build_services


MIN_SAMPLES_PER_CLASS = 3


def main() -> None:
    settings = get_settings()
    svc = build_services(settings)

    vectors: list[np.ndarray] = []
    payloads: list[dict] = []
    for point in svc.vector_store.iter_points_with_vectors():
        if point.vector is None:
            continue
        vector = point.vector
        if isinstance(vector, dict):
            raise RuntimeError("Скрипт ожидает один безымянный вектор на точку")
        vectors.append(np.asarray(vector, dtype=np.float32))
        payloads.append(dict(point.payload or {}))

    if not vectors:
        raise SystemExit("В индексе нет данных. Сначала выполните import_dataset.py")

    X = np.vstack(vectors)
    for field in CATEGORICAL_FIELDS:
        indices = [index for index, payload in enumerate(payloads) if payload.get(field)]
        if not indices:
            print(f"[{field}] пропуск: нет разметки")
            continue

        y = np.asarray([str(payloads[index][field]) for index in indices])
        counts = Counter(y.tolist())
        valid_classes = {label for label, count in counts.items() if count >= MIN_SAMPLES_PER_CLASS}
        selected = [index for index in indices if str(payloads[index][field]) in valid_classes]
        y_selected = np.asarray([str(payloads[index][field]) for index in selected])

        if len(set(y_selected.tolist())) < 2:
            print(
                f"[{field}] пропуск: нужно минимум 2 класса и "
                f"{MIN_SAMPLES_PER_CLASS} примера каждого класса"
            )
            continue

        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X[selected], y_selected)
        output = settings.classifier_dir / f"{field}.joblib"
        joblib.dump(model, output)
        print(f"[{field}] сохранено: {output}; классы={list(model.classes_)}")

    print(
        "Важно: это обучение на всех данных без честной оценки качества. "
        "Для валидации разделяйте данные по object_group_id, а не по фотографиям."
    )


if __name__ == "__main__":
    main()
