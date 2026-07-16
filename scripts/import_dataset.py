#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from pydantic import ValidationError

from app.schemas import ArtifactMetadata


def optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None

    return int(value)


def clean(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None


def parse_metadata(
    row: dict[str, str],
    line_number: int,
) -> ArtifactMetadata:
    try:
        return ArtifactMetadata(
            title=clean(row.get("title")),
            artifact_type=clean(row.get("artifact_type")),
            material=clean(row.get("material")),
            era=clean(row.get("era")),
            year_from=optional_int(row.get("year_from")),
            year_to=optional_int(row.get("year_to")),
            description=clean(row.get("description")),
            expert_source=clean(row.get("expert_source")),
            object_group_id=clean(row.get("object_group_id")),
            tags=[
                item.strip()
                for item in (row.get("tags") or "").split(";")
                if item.strip()
            ],
        )
    except (ValueError, ValidationError) as exc:
        raise ValueError(
            f"Ошибка метаданных в строке {line_number}: {exc}"
        ) from exc


def resolve_image_path(
    manifest_dir: Path,
    raw_value: str,
) -> Path:
    raw_path = Path(raw_value.strip())

    if raw_path.is_absolute():
        return raw_path.resolve()

    return (manifest_dir / raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Импорт размеченного CSV в векторный индекс"
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="CSV-файл с колонкой image_path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить CSV без изменения базы",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Считать отсутствующие изображения ошибкой",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()

    if not manifest.exists():
        raise SystemExit(f"CSV-файл не найден: {manifest}")

    manifest_dir = manifest.parent
    prepared_rows: list[
        tuple[int, Path, ArtifactMetadata]
    ] = []

    warnings: list[str] = []
    errors: list[str] = []

    with manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise SystemExit("CSV-файл не содержит заголовков")

        if "image_path" not in reader.fieldnames:
            raise SystemExit(
                "В CSV обязательна колонка image_path"
            )

        for line_number, row in enumerate(reader, start=2):
            raw_image_path = (row.get("image_path") or "").strip()

            if not raw_image_path:
                errors.append(
                    f"Строка {line_number}: не указан image_path"
                )
                continue

            image_path = resolve_image_path(
                manifest_dir,
                raw_image_path,
            )

            try:
                metadata = parse_metadata(row, line_number)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            if not image_path.exists():
                message = (
                    f"Строка {line_number}: файл не найден: "
                    f"{image_path}"
                )

                if args.strict:
                    errors.append(message)
                else:
                    warnings.append(message)

            prepared_rows.append(
                (line_number, image_path, metadata)
            )

    print("\nПодготовка импорта")
    print("=" * 50)
    print(f"CSV: {manifest}")
    print(f"Строк подготовлено: {len(prepared_rows)}")
    print(f"Предупреждений: {len(warnings)}")
    print(f"Ошибок: {len(errors)}")

    if warnings:
        print("\nПредупреждения:")
        for warning in warnings:
            print(f"  [WARN] {warning}")

    if errors:
        print("\nОшибки:")
        for error in errors:
            print(f"  [ERROR] {error}")

        raise SystemExit(1)

    if args.dry_run:
        print("\nDry-run завершён. База не изменена.")

        for line_number, image_path, metadata in prepared_rows:
            print(
                f"  Строка {line_number}: "
                f"{image_path.name} | "
                f"type={metadata.artifact_type!r} | "
                f"material={metadata.material!r} | "
                f"era={metadata.era!r}"
            )

        return

    # Тяжёлые сервисы загружаются только при реальном импорте.
    from app.config import get_settings
    from app.services.container import build_services
    from app.services.image_io import save_image

    settings = get_settings()
    services = build_services(settings)

    imported = 0
    skipped = 0

    for line_number, image_path, metadata in prepared_rows:
        if not image_path.exists():
            skipped += 1
            print(
                f"[skip line {line_number}] "
                f"Файл не найден: {image_path}"
            )
            continue

        try:
            image_bytes = image_path.read_bytes()
            saved = save_image(
                image_bytes,
                settings.image_dir,
            )
            embedding = services.encoder.embed_bytes(
                image_bytes
            )

            payload = metadata.model_dump()
            payload.update(
                {
                    "image_path": str(saved.path),
                    "sha256": saved.sha256,
                    "width": saved.width,
                    "height": saved.height,
                }
            )

            services.vector_store.upsert(
                saved.artifact_id,
                embedding.tolist(),
                payload,
            )

            imported += 1
            print(
                f"[{imported}] "
                f"{image_path.name} -> "
                f"{saved.artifact_id}"
            )
        except Exception as exc:
            skipped += 1
            print(
                f"[skip line {line_number}] "
                f"Ошибка обработки {image_path}: {exc}"
            )

    print("\nИмпорт завершён")
    print(f"Импортировано: {imported}")
    print(f"Пропущено: {skipped}")


if __name__ == "__main__":
    main()