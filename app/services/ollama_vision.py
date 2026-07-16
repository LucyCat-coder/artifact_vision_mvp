from typing import Any

from ollama import Client
from pydantic import BaseModel, Field


class VisualObservation(BaseModel):
    summary: str
    visible_features: list[str] = Field(default_factory=list)
    shape: str | None = None
    ornament: str | None = None
    surface_and_wear: str | None = None
    probable_material_by_appearance: str | None = None
    inscriptions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


class OllamaVisionObserver:
    def __init__(self, host: str, model: str) -> None:
        self.client = Client(host=host)
        self.model = model

    def observe(self, image_bytes: bytes) -> dict[str, Any]:
        prompt = (
            "Опиши только визуально наблюдаемые признаки археологического или исторического "
            "предмета на фотографии. Не выдавай точную датировку или атрибуцию как установленный "
            "факт. Отдельно отметь форму, орнамент, поверхность, износ, возможный материал по виду, "
            "надписи и ограничения качества фотографии. Ответ должен соответствовать JSON-схеме."
        )
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
            format=VisualObservation.model_json_schema(),
            options={"temperature": 0},
        )
        content = response.message.content
        return VisualObservation.model_validate_json(content).model_dump()
