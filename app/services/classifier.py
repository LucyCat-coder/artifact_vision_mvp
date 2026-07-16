from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.schemas import Alternative, DatingPrediction, FieldPrediction
from app.services.types import Neighbor


CATEGORICAL_FIELDS = ("artifact_type", "material", "era")


def _weight(score: float) -> float:
    # Косинусное сходство усиливается, чтобы ближайшие эталоны влияли заметнее.
    return max(score, 0.0) ** 4


def predict_categorical(neighbors: list[Neighbor], field: str) -> FieldPrediction:
    votes: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    strongest_similarity = 0.0

    for neighbor in neighbors:
        label = neighbor.payload.get(field)
        if not label:
            continue
        label = str(label).strip()
        if not label:
            continue
        votes[label] += _weight(neighbor.score)
        counts[label] += 1
        strongest_similarity = max(strongest_similarity, neighbor.score)

    if not votes:
        return FieldPrediction(label=None, confidence=0.0, evidence_count=0)

    ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    total = sum(votes.values()) or 1.0
    best_label, best_weight = ranked[0]
    vote_share = best_weight / total

    # Это эвристика, а не статистически откалиброванная вероятность.
    confidence = min(1.0, 0.75 * vote_share + 0.25 * max(strongest_similarity, 0.0))
    alternatives = [
        Alternative(label=label, confidence=round(weight / total, 4))
        for label, weight in ranked[1:4]
    ]

    return FieldPrediction(
        label=best_label,
        confidence=round(confidence, 4),
        source="nearest_neighbors",
        evidence_count=counts[best_label],
        alternatives=alternatives,
    )


def predict_dating(neighbors: list[Neighbor]) -> DatingPrediction:
    samples: list[tuple[float, int, int]] = []
    for neighbor in neighbors:
        start = neighbor.payload.get("year_from")
        end = neighbor.payload.get("year_to")
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        samples.append((_weight(neighbor.score), int(start), int(end)))

    if not samples:
        return DatingPrediction()

    weights = np.asarray([item[0] for item in samples], dtype=np.float64)
    if float(weights.sum()) == 0.0:
        weights = np.ones_like(weights)
    starts = np.asarray([item[1] for item in samples], dtype=np.float64)
    ends = np.asarray([item[2] for item in samples], dtype=np.float64)

    year_from = int(round(float(np.average(starts, weights=weights))))
    year_to = int(round(float(np.average(ends, weights=weights))))
    midpoint = (year_from + year_to) / 2
    approximate_age = max(0, int(round(datetime.now().year - midpoint)))

    dispersion = float(np.average(np.abs((starts + ends) / 2 - midpoint), weights=weights))
    spread_factor = 1.0 / (1.0 + dispersion / 500.0)
    similarity_factor = max((neighbor.score for neighbor in neighbors), default=0.0)
    confidence = min(1.0, 0.6 * spread_factor + 0.4 * max(similarity_factor, 0.0))

    return DatingPrediction(
        year_from=min(year_from, year_to),
        year_to=max(year_from, year_to),
        approximate_age_years=approximate_age,
        confidence=round(confidence, 4),
        evidence_count=len(samples),
    )


class LinearHeadRepository:
    """Загружает обученные sklearn-классификаторы, если они существуют."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._models: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self._models.clear()
        for field in CATEGORICAL_FIELDS:
            path = self.model_dir / f"{field}.joblib"
            if path.exists():
                self._models[field] = joblib.load(path)

    def predict(self, field: str, embedding: np.ndarray) -> FieldPrediction | None:
        model = self._models.get(field)
        if model is None:
            return None

        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
        classes = model.classes_
        order = np.argsort(probabilities)[::-1]
        best_index = int(order[0])
        alternatives = [
            Alternative(
                label=str(classes[index]),
                confidence=round(float(probabilities[index]), 4),
            )
            for index in order[1:4]
        ]
        return FieldPrediction(
            label=str(classes[best_index]),
            confidence=round(float(probabilities[best_index]), 4),
            source="linear_head",
            evidence_count=0,
            alternatives=alternatives,
        )
