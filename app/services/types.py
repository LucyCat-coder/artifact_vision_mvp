from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Neighbor:
    artifact_id: str
    score: float
    payload: dict[str, Any]
