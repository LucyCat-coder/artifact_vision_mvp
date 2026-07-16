from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.types import Neighbor


class ArtifactVectorStore:
    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        local_path: Path,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=str(local_path))
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, artifact_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=artifact_id, vector=vector, payload=payload)],
            wait=True,
        )

    def query(self, vector: list[float], limit: int) -> list[Neighbor]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Neighbor(
                artifact_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def count(self) -> int:
        return int(self.client.count(self.collection_name, exact=True).count)

    def iter_points_with_vectors(self, batch_size: int = 256) -> Iterable[Any]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            yield from points
            if offset is None:
                break
