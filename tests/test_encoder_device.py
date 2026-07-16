import pytest
import torch

from app.services.encoder import ImageEncoder


def test_cpu_device() -> None:
    device = ImageEncoder._resolve_device("cpu")

    assert device == torch.device("cpu")


def test_auto_uses_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    device = ImageEncoder._resolve_device("auto")

    assert device == torch.device("cuda")


def test_auto_uses_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    device = ImageEncoder._resolve_device("auto")

    assert device == torch.device("cpu")


def test_cuda_fails_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA недоступна",
    ):
        ImageEncoder._resolve_device("cuda")


def test_invalid_device_fails() -> None:
    with pytest.raises(
        ValueError,
        match="auto, cuda или cpu",
    ):
        ImageEncoder._resolve_device("unknown")