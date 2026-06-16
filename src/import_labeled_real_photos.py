from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config import BASE_DIR, CLASSES, METADATA_DIR, RAW_DIR


DEFAULT_SOURCE_DIR = BASE_DIR.parent / "Poze_Reale"
DEFAULT_LABELS_CSV = BASE_DIR.parent / "Etichete_Poze_Reale_template.csv"
DEFAULT_MANIFEST = METADATA_DIR / "labeled_real_photos_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_COLUMNS = {"image", "actual_class"}
MANIFEST_FIELDS = [
    "imported_at",
    "status",
    "class_name",
    "relative_path",
    "original_path",
    "sha256",
    "width",
    "height",
    "image_format",
    "image_mode",
    "file_size_bytes",
    "actual_species",
    "actual_status",
    "notes",
    "error",
]


def clean_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "image"


def normalize_class_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(source_dir: Path, value: str) -> Path:
    path = Path(value.strip().strip('"'))
    if path.is_absolute():
        return path
    return source_dir / path


def inspect_image(path: Path) -> dict[str, str | int]:
    with Image.open(path) as image:
        image.load()
        return {
            "width": image.width,
            "height": image.height,
            "image_format": image.format or "",
            "image_mode": image.mode,
        }


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Lipsesc coloanele obligatorii din CSV: {missing_list}")
        return list(reader)


def destination_for(source_path: Path, class_name: str, index: int) -> Path:
    suffix = source_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    filename = f"real_{index:06d}_{clean_filename(source_path.stem)}{suffix}"
    return RAW_DIR / class_name / "real_photos_labeled" / filename


def import_labeled_photos(
    *,
    source_dir: Path,
    labels_csv: Path,
    manifest_path: Path,
    overwrite: bool,
    dry_run: bool,
    skip_unknown: bool,
) -> None:
    rows = read_label_rows(labels_csv)
    allowed_classes = set(CLASSES)
    imported_at = datetime.now(timezone.utc).isoformat()
    manifest_rows: list[dict[str, object]] = []
    errors: list[str] = []
    imported_count = 0
    skipped_count = 0

    for index, row in enumerate(rows, start=1):
        raw_class_name = row.get("actual_class", "")
        class_name = normalize_class_name(raw_class_name)
        if not class_name:
            skipped_count += 1
            continue

        if class_name not in allowed_classes:
            message = f"Row {index}: unknown class '{raw_class_name}'"
            if skip_unknown:
                print(f"[WARN] {message}; skipping image.")
                skipped_count += 1
                continue
            errors.append(message)
            continue

        source_path = resolve_image_path(source_dir, row.get("image", ""))
        destination_path = destination_for(source_path, class_name, index)
        manifest_row: dict[str, object] = {
            "imported_at": imported_at,
            "status": "pending",
            "class_name": class_name,
            "relative_path": "",
            "original_path": str(source_path),
            "sha256": "",
            "width": "",
            "height": "",
            "image_format": "",
            "image_mode": "",
            "file_size_bytes": "",
            "actual_species": row.get("actual_species", ""),
            "actual_status": row.get("actual_status", ""),
            "notes": row.get("notes", ""),
            "error": "",
        }

        try:
            if not source_path.exists():
                raise FileNotFoundError(str(source_path))
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {source_path.suffix}")

            image_meta = inspect_image(source_path)
            manifest_row.update(image_meta)
            manifest_row["sha256"] = sha256_file(source_path)
            manifest_row["file_size_bytes"] = source_path.stat().st_size
            manifest_row["relative_path"] = destination_path.relative_to(RAW_DIR).as_posix()

            if dry_run:
                manifest_row["status"] = "dry_run"
            elif destination_path.exists() and not overwrite:
                manifest_row["status"] = "skipped_existing"
                skipped_count += 1
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                manifest_row["status"] = "imported"
                imported_count += 1
        except Exception as exc:
            manifest_row["status"] = "error"
            manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"Row {index}: {manifest_row['error']}")

        manifest_rows.append(manifest_row)

    if errors and not skip_unknown:
        print("[ERROR] Import stopped. Correct the labels/files below:")
        for message in errors[:20]:
            print(f"- {message}")
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more errors")
        raise SystemExit(1)

    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"[OK] Rows read: {len(rows)}")
    print(f"[OK] Imported images: {imported_count}")
    print(f"[OK] Skipped images: {skipped_count}")
    if dry_run:
        print("[OK] Dry-run: no files were copied and the manifest was not written.")
    else:
        print(f"[OK] Manifest import: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import labeled real photos into data/raw/<class>/real_photos_labeled."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-unknown", action="store_true")
    args = parser.parse_args()

    import_labeled_photos(
        source_dir=Path(args.source_dir),
        labels_csv=Path(args.labels_csv),
        manifest_path=Path(args.manifest),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        skip_unknown=args.skip_unknown,
    )


if __name__ == "__main__":
    main()
