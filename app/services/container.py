from dataclasses import dataclass

from app.config import Settings
from app.services.classifier import LinearHeadRepository
from app.services.encoder import ImageEncoder
from app.services.ollama_vision import OllamaVisionObserver
from app.services.vector_store import ArtifactVectorStore


@dataclass
class Services:
    encoder: ImageEncoder
    vector_store: ArtifactVectorStore
    classifiers: LinearHeadRepository
    ollama: OllamaVisionObserver | None


def build_services(settings: Settings) -> Services:
    encoder = ImageEncoder(
    model_name=settings.image_model_name,
    device=settings.model_device,
)
    if encoder.dimension != settings.embedding_size:
        raise RuntimeError(
            f"EMBEDDING_SIZE={settings.embedding_size}, но модель возвращает "
            f"{encoder.dimension}. Обновите .env или создайте новую коллекцию."
        )

    vector_store = ArtifactVectorStore(
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_size,
        local_path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    classifiers = LinearHeadRepository(settings.classifier_dir)
    ollama = (
        OllamaVisionObserver(settings.ollama_host, settings.ollama_vision_model)
        if settings.ollama_enabled
        else None
    )
    return Services(
        encoder=encoder,
        vector_store=vector_store,
        classifiers=classifiers,
        ollama=ollama,
    )
