from app.services.classifier import predict_categorical, predict_dating
from app.services.types import Neighbor


def test_weighted_vote_prefers_strong_neighbors() -> None:
    neighbors = [
        Neighbor("1", 0.95, {"material": "бронза"}),
        Neighbor("2", 0.91, {"material": "бронза"}),
        Neighbor("3", 0.60, {"material": "железо"}),
    ]
    result = predict_categorical(neighbors, "material")
    assert result.label == "бронза"
    assert result.confidence > 0.7


def test_dating_uses_neighbor_ranges() -> None:
    neighbors = [
        Neighbor("1", 0.95, {"year_from": -300, "year_to": -200}),
        Neighbor("2", 0.90, {"year_from": -250, "year_to": -150}),
    ]
    result = predict_dating(neighbors)
    assert result.year_from is not None
    assert result.year_to is not None
    assert result.year_from < result.year_to
