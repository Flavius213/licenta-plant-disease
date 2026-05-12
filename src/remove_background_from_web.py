from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.background_removal import IMAGE_EXTENSIONS, save_leaf_on_checkerboard
from src.config import BASE_DIR, METADATA_DIR, RAW_DIR


BACKUP_DIR_DEFAULT = BASE_DIR / "data" / "web_originals"
MANIFEST_DEFAULT = METADATA_DIR / "web_background_manifest.csv"

MANIFEST_FIELDS = [
    "processed_at",
    "class_name",
    "original_relative_path",
    "processed_relative_path",
    "backup_relative_path",
    "status",
    "error",
]


def is_web_image(path: Path) -> bool:
    relative = path.relative_to(RAW_DIR).as_posix().lower()
    if "plantvillage" in relative:
        return False
    if path.name.lower().endswith("_leaf_bg.png"):
        return False
    return path.suffix.lower() in IMAGE_EXTENSIONS


def safe_path_in_project(path: Path) -> Path:
    resolved_path = path.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_path.parents and resolved_path != resolved_base:
        raise ValueError(f"Refuz sa modific fisier in afara proiectului: {resolved_path}")
    return resolved_path


def processed_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_leaf_bg.png")


def backup_path_for(image_path: Path, backup_dir: Path) -> Path:
    relative = image_path.relative_to(RAW_DIR)
    destination = backup_dir / relative
    suffix_index = 1
    while destination.exists():
        destination = backup_dir / relative.with_name(f"{relative.stem}_{suffix_index}{relative.suffix}")
        suffix_index += 1
    return destination


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_web_images(
    *,
    raw_dir: Path,
    backup_dir: Path,
    manifest_path: Path,
    backend: str,
    keep_originals: bool,
    limit: int | None,
) -> int:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Nu exista folder raw: {raw_dir}")

    rows: list[dict[str, str]] = []
    processed = 0
    web_images = [path for path in sorted(raw_dir.rglob("*")) if path.is_file() and is_web_image(path)]
    if limit is not None:
        web_images = web_images[:limit]

    for image_path in web_images:
        class_name = image_path.relative_to(raw_dir).parts[0]
        processed_path = processed_path_for(image_path)
        backup_path = backup_path_for(image_path, backup_dir)
        status = "processed"
        error = ""

        try:
            safe_path_in_project(image_path)
            safe_path_in_project(processed_path)
            safe_path_in_project(backup_path)
            save_leaf_on_checkerboard(image_path, processed_path, backend=backend)
            if keep_originals:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, backup_path)
            else:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(backup_path))
            processed += 1
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "class_name": class_name,
                "original_relative_path": relative_to_project(image_path)
                if image_path.exists()
                else "",
                "processed_relative_path": relative_to_project(processed_path)
                if processed_path.exists()
                else "",
                "backup_relative_path": relative_to_project(backup_path)
                if backup_path.exists()
                else "",
                "status": status,
                "error": error,
            }
        )

        if processed and processed % 50 == 0:
            print(f"[INFO] Imagini web procesate: {processed}/{len(web_images)}")

    write_manifest(rows, manifest_path)
    print(f"[OK] Imagini web procesate cu fundal scos: {processed}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Originale web mutate/copiate in: {backup_dir}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoate fundalul imaginilor web din data/raw.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--backup-dir", default=str(BACKUP_DIR_DEFAULT))
    parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    parser.add_argument("--backend", choices=["auto", "rembg", "opencv"], default="auto")
    parser.add_argument("--keep-originals", action="store_true", help="Copiaza originalele in loc sa le mute.")
    parser.add_argument("--limit", type=int, help="Proceseaza doar primele N imagini web.")
    args = parser.parse_args()

    process_web_images(
        raw_dir=Path(args.raw_dir),
        backup_dir=Path(args.backup_dir),
        manifest_path=Path(args.manifest),
        backend=args.backend,
        keep_originals=args.keep_originals,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
