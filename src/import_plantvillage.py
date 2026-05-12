from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config import BASE_DIR, METADATA_DIR, RAW_DIR
from src.download import slugify


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
MANIFEST_PATH = METADATA_DIR / "plantvillage_manifest.csv"

PLANTVILLAGE_CLASS_MAP = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "apple_black_rot",
    "Apple___Cedar_apple_rust": "apple_cedar_apple_rust",
    "Apple___healthy": "apple_healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry_powdery_mildew",
    "Cherry_(including_sour)___healthy": "cherry_healthy",
}

MANIFEST_FIELDS = [
    "imported_at",
    "source",
    "source_label",
    "class_name",
    "relative_path",
    "original_path",
    "sha256",
    "width",
    "height",
    "image_format",
    "image_mode",
    "file_size_bytes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        return {
            "width": image.width,
            "height": image.height,
            "image_format": image.format or "",
            "image_mode": image.mode,
            "file_size_bytes": path.stat().st_size,
        }


def resolve_variant_dir(source_dir: Path, variant: str) -> Path:
    variant_dir = source_dir / variant
    return variant_dir if variant_dir.exists() else source_dir


def find_label_dirs(source_dir: Path, variant: str) -> dict[str, Path]:
    search_dir = resolve_variant_dir(source_dir, variant)
    label_dirs = {}
    for path in search_dir.rglob("*"):
        if path.is_dir() and path.name in PLANTVILLAGE_CLASS_MAP:
            label_dirs[path.name] = path
    return label_dirs


def iter_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def next_destination_path(destination_dir: Path, source_path: Path, index: int) -> Path:
    suffix = source_path.suffix.lower()
    return destination_dir / f"plantvillage_{index:06d}{suffix}"


def append_manifest(rows: list[dict[str, object]]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    should_write_header = not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def import_plantvillage(
    source_dir: Path,
    *,
    variant: str = "color",
    limit_per_class: int | None = None,
    dry_run: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Folderul PlantVillage nu exista: {source_dir}")

    label_dirs = find_label_dirs(source_dir, variant)
    if not label_dirs:
        known_labels = ", ".join(PLANTVILLAGE_CLASS_MAP)
        raise ValueError(f"Nu am gasit foldere PlantVillage cunoscute. Caut etichete ca: {known_labels}")

    total_imported = 0
    manifest_rows: list[dict[str, object]] = []

    for source_label, class_name in PLANTVILLAGE_CLASS_MAP.items():
        label_dir = label_dirs.get(source_label)
        if not label_dir:
            print(f"[SKIP] Lipseste in PlantVillage: {source_label}")
            continue

        destination_dir = RAW_DIR / class_name / f"plantvillage_{slugify(source_label)}"
        images = list(iter_images(label_dir))
        if limit_per_class is not None:
            images = images[:limit_per_class]

        print(f"[INFO] {variant}/{source_label} -> {class_name}: {len(images)} imagini")
        if dry_run:
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(images, start=1):
            destination_path = next_destination_path(destination_dir, source_path, index)
            if destination_path.exists():
                continue

            shutil.copy2(source_path, destination_path)
            meta = image_metadata(destination_path)
            manifest_rows.append(
                {
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plantvillage",
                    "source_label": source_label,
                    "class_name": class_name,
                    "relative_path": destination_path.resolve().relative_to(BASE_DIR.resolve()).as_posix(),
                    "original_path": str(source_path),
                    "sha256": sha256_file(destination_path),
                    "width": meta["width"],
                    "height": meta["height"],
                    "image_format": meta["image_format"],
                    "image_mode": meta["image_mode"],
                    "file_size_bytes": meta["file_size_bytes"],
                }
            )
            total_imported += 1

    if manifest_rows:
        append_manifest(manifest_rows)

    print(f"[OK] Import PlantVillage complet: {total_imported} imagini copiate.")
    print(f"[OK] Manifest: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa clase relevante din PlantVillage in data/raw.")
    parser.add_argument("--source", required=True, help="Calea catre folderul PlantVillage dezarhivat.")
    parser.add_argument(
        "--variant",
        default="color",
        choices=["color", "grayscale", "segmented"],
        help="Varianta PlantVillage de importat. Implicit: color.",
    )
    parser.add_argument("--limit-per-class", type=int, help="Limita optionala de imagini pe clasa.")
    parser.add_argument("--dry-run", action="store_true", help="Afiseaza ce ar importa, fara copiere.")
    args = parser.parse_args()

    import_plantvillage(
        Path(args.source),
        variant=args.variant,
        limit_per_class=args.limit_per_class,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
