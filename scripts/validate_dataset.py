#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = {
    "image_path",
    "title",
    "artifact_type",
    "material",
    "era",
    "year_from",
    "year_to",
    "description",
    "expert_source",
    "object_group_id",
    "tags",
}

CONSISTENT_FIELDS = (
    "artifact_type",
    "material",
    "era",
    "year_from",
    "year_to",
)


def parse_optional_int(
    value: str | None,
    field_name: str,
    line_number: int,
    errors: list[str],
) -> int | None:
    if value is None or not value.strip():
        return None

    try:
        return int(value)
    except ValueError:
        errors.append(
            f"Строка {line_number}: поле {field_name} "
            f"должно быть целым числом, получено: {value!r}"
        )
        return None


def normalize(value: str | None) -> str:
    return (value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка CSV-разметки датасета"
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Путь к CSV-файлу",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Считать отсутствующие изображения ошибками",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()

    if not manifest.exists():
        raise SystemExit(f"CSV-файл не найден: {manifest}")

    manifest_dir = manifest.parent
    errors: list[str] = []
    warnings: list[str] = []
    image_paths: set[Path] = set()
    object_groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)

    row_count = 0
    existing_images = 0
    missing_images = 0

    with manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise SystemExit("CSV-файл не содержит заголовков")

        fieldnames = {
            field.strip()
            for field in reader.fieldnames
            if field
        }

        missing_columns = REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            errors.append(
                "Отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing_columns))
            )

        for line_number, row in enumerate(reader, start=2):
            row_count += 1

            image_path_raw = normalize(row.get("image_path"))

            if not image_path_raw:
                errors.append(
                    f"Строка {line_number}: не указан image_path"
                )
                continue

            raw_path = Path(image_path_raw)
            image_path = (
                raw_path
                if raw_path.is_absolute()
                else (manifest_dir / raw_path).resolve()
            )

            if image_path in image_paths:
                warnings.append(
                    f"Строка {line_number}: путь к изображению "
                    f"повторяется: {image_path}"
                )
            else:
                image_paths.add(image_path)

            if not image_path.exists():
                missing_images += 1
                message = (
                    f"Строка {line_number}: изображение пока "
                    f"не найдено: {image_path}"
                )

                if args.strict:
                    errors.append(message)
                else:
                    warnings.append(message)
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    existing_images += 1
                except (UnidentifiedImageError, OSError) as exc:
                    errors.append(
                        f"Строка {line_number}: повреждённое или "
                        f"неподдерживаемое изображение: "
                        f"{image_path} ({exc})"
                    )

            year_from = parse_optional_int(
                row.get("year_from"),
                "year_from",
                line_number,
                errors,
            )
            year_to = parse_optional_int(
                row.get("year_to"),
                "year_to",
                line_number,
                errors,
            )

            if (
                year_from is not None
                and year_to is not None
                and year_from > year_to
            ):
                errors.append(
                    f"Строка {line_number}: year_from "
                    f"({year_from}) больше year_to ({year_to})"
                )

            if not normalize(row.get("artifact_type")):
                warnings.append(
                    f"Строка {line_number}: не указан artifact_type"
                )

            if not normalize(row.get("material")):
                warnings.append(
                    f"Строка {line_number}: не указан material"
                )

            if not normalize(row.get("era")):
                warnings.append(
                    f"Строка {line_number}: не указана era"
                )

            object_group_id = normalize(
                row.get("object_group_id")
            )

            if not object_group_id:
                warnings.append(
                    f"Строка {line_number}: "
                    f"не указан object_group_id"
                )
            else:
                normalized_row = {
                    key: normalize(row.get(key))
                    for key in CONSISTENT_FIELDS
                }
                object_groups[object_group_id].append(
                    (line_number, normalized_row)
                )

    for object_group_id, group_rows in object_groups.items():
        reference_line, reference = group_rows[0]

        for line_number, row in group_rows[1:]:
            for field in CONSISTENT_FIELDS:
                if row[field] != reference[field]:
                    warnings.append(
                        f"Объект {object_group_id}: поле {field} "
                        f"различается в строках "
                        f"{reference_line} и {line_number}"
                    )

    print("\nПроверка датасета")
    print("=" * 50)
    print(f"CSV: {manifest}")
    print(f"Строк данных: {row_count}")
    print(f"Уникальных объектов: {len(object_groups)}")
    print(f"Найдено изображений: {existing_images}")
    print(f"Отсутствует изображений: {missing_images}")
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

    print("\nСтруктура датасета корректна.")

    if missing_images and not args.strict:
        print(
            "Изображения можно добавить позднее. "
            "После добавления запусти проверку с --strict."
        )


if __name__ == "__main__":
    main()