from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Artifact Vision MVP"
    data_dir: Path = Path("./data")
    image_model_name: str = "facebook/dinov2-small"
    model_device: str = "auto"
    embedding_size: int = 384

    qdrant_collection: str = "artifacts"
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    ollama_enabled: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_vision_model: str = "gemma3:4b"

    classifier_dir: Path = Path("./data/models")
    top_k: int = 7

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.classifier_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
