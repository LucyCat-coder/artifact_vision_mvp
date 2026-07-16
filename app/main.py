from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import (
    AnalysisResponse,
    ArtifactMetadata,
    SimilarArtifact,
    StoredArtifact,
)
from app.services.classifier import CATEGORICAL_FIELDS, predict_categorical, predict_dating
from app.services.container import Services, build_services
from app.services.image_io import InvalidImageError, save_image


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services(settings)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "MVP для индексации эталонных находок, поиска визуально похожих объектов "
        "и первичной классификации по размеченному датасету."
    ),
    lifespan=lifespan,
)


def services(request: Request) -> Services:
    return request.app.state.services


def parse_metadata(raw: str) -> ArtifactMetadata:
    try:
        return ArtifactMetadata.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    svc = services(request)
    return {
        "status": "ok",
        "device": str(svc.encoder.device),
        "indexed_artifacts": svc.vector_store.count(),
        "ollama_enabled": svc.ollama is not None,
    }


@app.post("/v1/artifacts", response_model=StoredArtifact)
def add_artifact(
    request: Request,
    file: Annotated[UploadFile, File(description="Фотография эталонной находки")],
    metadata: Annotated[str, Form(description="JSON по схеме ArtifactMetadata")],
) -> StoredArtifact:
    svc = services(request)
    image_bytes = file.file.read()
    parsed_metadata = parse_metadata(metadata)

    try:
        saved = save_image(image_bytes, settings.image_dir)
        embedding = svc.encoder.embed_bytes(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {exc}") from exc

    payload = parsed_metadata.model_dump()
    payload.update(
        {
            "image_path": str(saved.path),
            "sha256": saved.sha256,
            "width": saved.width,
            "height": saved.height,
        }
    )
    svc.vector_store.upsert(saved.artifact_id, embedding.tolist(), payload)

    return StoredArtifact(
        artifact_id=saved.artifact_id,
        image_path=str(saved.path),
        sha256=saved.sha256,
        width=saved.width,
        height=saved.height,
        metadata=parsed_metadata,
    )


@app.post("/v1/analyze", response_model=AnalysisResponse)
def analyze_artifact(
    request: Request,
    file: Annotated[UploadFile, File(description="Фотография новой находки")],
    top_k: Annotated[int | None, Form()] = None,
    use_ollama: Annotated[bool, Form()] = False,
) -> AnalysisResponse:
    svc = services(request)
    image_bytes = file.file.read()
    limit = max(1, min(top_k or settings.top_k, 50))

    try:
        embedding = svc.encoder.embed_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось обработать изображение: {exc}") from exc

    neighbors = svc.vector_store.query(embedding.tolist(), limit=limit)
    predictions = {
        field: predict_categorical(neighbors, field) for field in CATEGORICAL_FIELDS
    }

    # При наличии обученных линейных голов используем их для классификации,
    # а поиск похожих объектов всё равно оставляем как объясняющую часть результата.
    for field in CATEGORICAL_FIELDS:
        head_prediction = svc.classifiers.predict(field, embedding)
        if head_prediction is not None:
            predictions[field] = head_prediction

    similar = []
    for neighbor in neighbors:
        metadata_dict = {
            key: neighbor.payload.get(key)
            for key in ArtifactMetadata.model_fields
        }
        similar.append(
            SimilarArtifact(
                artifact_id=neighbor.artifact_id,
                similarity=round(neighbor.score, 4),
                image_path=neighbor.payload.get("image_path"),
                metadata=ArtifactMetadata.model_validate(metadata_dict),
            )
        )

    warnings: list[str] = []
    visual_observation = None
    if use_ollama:
        if svc.ollama is None:
            warnings.append("Ollama отключена. Установите OLLAMA_ENABLED=true.")
        else:
            try:
                visual_observation = svc.ollama.observe(image_bytes)
            except Exception as exc:
                warnings.append(f"Ollama не смогла обработать изображение: {exc}")

    if not neighbors:
        warnings.append("В базе пока нет эталонных находок; классификация невозможна.")

    return AnalysisResponse(
        artifact_type=predictions["artifact_type"],
        material=predictions["material"],
        era=predictions["era"],
        dating=predict_dating(neighbors),
        similar_artifacts=similar,
        visual_observation=visual_observation,
        warnings=warnings,
    )
