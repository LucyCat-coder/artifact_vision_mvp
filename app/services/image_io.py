from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class SavedImage:
    artifact_id: str
    path: Path
    sha256: str
    width: int
    height: int


class InvalidImageError(ValueError):
    pass


def open_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise InvalidImageError("Файл изображения пуст")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Не удалось прочитать изображение") from exc

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def save_image(image_bytes: bytes, image_dir: Path) -> SavedImage:
    image = open_image(image_bytes)
    artifact_id = str(uuid4())
    digest = sha256(image_bytes).hexdigest()
    destination = image_dir / f"{artifact_id}.jpg"
    image.save(destination, format="JPEG", quality=95, optimize=True)

    return SavedImage(
        artifact_id=artifact_id,
        path=destination,
        sha256=digest,
        width=image.width,
        height=image.height,
    )
