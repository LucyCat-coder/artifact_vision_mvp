from io import BytesIO
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


class ImageEncoder:
    """Лениво загружает визуальный энкодер и возвращает L2-нормированный вектор."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._processor = None
        self._model = None
        self._lock = Lock()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = device.strip().lower()

        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                "MODEL_DEVICE должен иметь значение auto, cuda или cpu"
            )

        if requested == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MODEL_DEVICE=cuda, но CUDA недоступна. "
                "Проверьте установку GPU-версии PyTorch."
            )

        return torch.device(requested)

    def _load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name
            )
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval().to(self.device)

    @property
    def dimension(self) -> int:
        self._load()

        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                "Не удалось определить размер эмбеддинга модели"
            )

        return int(hidden_size)

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")

        return self.embed_image(image)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self._load()

        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)

        pooled = getattr(outputs, "pooler_output", None)

        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        normalized = F.normalize(
            pooled.float(),
            p=2,
            dim=-1,
        )

        return normalized[0].cpu().numpy().astype(np.float32)