from typing import Any

from pydantic import BaseModel, Field, model_validator


class ArtifactMetadata(BaseModel):
    title: str | None = None
    artifact_type: str | None = None
    material: str | None = None
    era: str | None = None
    year_from: int | None = Field(
        default=None,
        description="Начало датировки; годы до н.э. задаются отрицательными значениями.",
    )
    year_to: int | None = Field(
        default=None,
        description="Конец датировки; годы до н.э. задаются отрицательными значениями.",
    )
    description: str | None = None
    expert_source: str | None = None
    object_group_id: str | None = Field(
        default=None,
        description="Идентификатор физического объекта, если у него несколько фотографий.",
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_years(self) -> "ArtifactMetadata":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from не может быть больше year_to")
        return self


class StoredArtifact(BaseModel):
    artifact_id: str
    image_path: str
    sha256: str
    width: int
    height: int
    metadata: ArtifactMetadata


class Alternative(BaseModel):
    label: str
    confidence: float


class FieldPrediction(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    source: str = "nearest_neighbors"
    evidence_count: int = 0
    alternatives: list[Alternative] = Field(default_factory=list)


class DatingPrediction(BaseModel):
    year_from: int | None = None
    year_to: int | None = None
    approximate_age_years: int | None = None
    confidence: float = 0.0
    evidence_count: int = 0


class SimilarArtifact(BaseModel):
    artifact_id: str
    similarity: float
    image_path: str | None = None
    metadata: ArtifactMetadata


class AnalysisResponse(BaseModel):
    artifact_type: FieldPrediction
    material: FieldPrediction
    era: FieldPrediction
    dating: DatingPrediction
    similar_artifacts: list[SimilarArtifact]
    visual_observation: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
