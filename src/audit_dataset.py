from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.config import CLASS_DISTRIBUTION, RAW_AUDIT, RAW_DIR

try:
    import imagehash
except ImportError:  
    imagehash = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

AUDIT_FIELDS = [
    "class_name",
    "keyword",
    "relative_path",
    "valid_image",
    "width",
    "height",
    "image_format",
    "image_mode",
    "file_size_bytes",
    "sha256",
    "phash",
    "exact_duplicate_of",
    "phash_duplicate_of",
    "error",
]

DISTRIBUTION_FIELDS = [
    "class_name",
    "total_files",
    "valid_images",
    "invalid_images",
    "exact_duplicates",
    "phash_duplicates",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_image_files(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def split_raw_path(path: Path) -> tuple[str, str, str]:
    relative = path.relative_to(RAW_DIR)
    parts = relative.parts
    class_name = parts[0] if len(parts) >= 1 else ""
    keyword = parts[1] if len(parts) >= 3 else ""
    return class_name, keyword, relative.as_posix()


def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            image.load()
            perceptual_hash = str(imagehash.phash(image)) if imagehash else ""
            return {
                "valid_image": True,
                "width": image.width,
                "height": image.height,
                "image_format": image.format or "",
                "image_mode": image.mode,
                "phash": perceptual_hash,
                "error": "",
            }
    except Exception as exc:
        return {
            "valid_image": False,
            "width": "",
            "height": "",
            "image_format": "",
            "image_mode": "",
            "phash": "",
            "error": type(exc).__name__,
        }


def audit_raw_dataset() -> tuple[int, int, int]:
    RAW_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    seen_sha: dict[str, str] = {}
    seen_phash: dict[str, str] = {}
    stats = defaultdict(Counter)

    total_files = 0
    valid_images = 0
    invalid_images = 0

    with RAW_AUDIT.open("w", newline="", encoding="utf-8") as audit_file:
        writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

        for path in iter_image_files(RAW_DIR):
            class_name, keyword, relative_path = split_raw_path(path)
            image_meta = inspect_image(path)
            sha256 = sha256_file(path)

            exact_duplicate_of = seen_sha.get(sha256, "")
            if not exact_duplicate_of:
                seen_sha[sha256] = relative_path

            phash = str(image_meta["phash"])
            phash_duplicate_of = seen_phash.get(phash, "") if phash else ""
            if phash and not phash_duplicate_of:
                seen_phash[phash] = relative_path

            row = {
                "class_name": class_name,
                "keyword": keyword,
                "relative_path": relative_path,
                "valid_image": image_meta["valid_image"],
                "width": image_meta["width"],
                "height": image_meta["height"],
                "image_format": image_meta["image_format"],
                "image_mode": image_meta["image_mode"],
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256,
                "phash": phash,
                "exact_duplicate_of": exact_duplicate_of,
                "phash_duplicate_of": phash_duplicate_of,
                "error": image_meta["error"],
            }
            writer.writerow(row)

            total_files += 1
            stats[class_name]["total_files"] += 1
            if image_meta["valid_image"]:
                valid_images += 1
                stats[class_name]["valid_images"] += 1
            else:
                invalid_images += 1
                stats[class_name]["invalid_images"] += 1
            if exact_duplicate_of:
                stats[class_name]["exact_duplicates"] += 1
            if phash_duplicate_of:
                stats[class_name]["phash_duplicates"] += 1

    with CLASS_DISTRIBUTION.open("w", newline="", encoding="utf-8") as distribution_file:
        writer = csv.DictWriter(distribution_file, fieldnames=DISTRIBUTION_FIELDS)
        writer.writeheader()
        for class_name in sorted(stats):
            row = {"class_name": class_name}
            row.update({field: stats[class_name][field] for field in DISTRIBUTION_FIELDS if field != "class_name"})
            writer.writerow(row)

    return total_files, valid_images, invalid_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw images and generate CSV reports.")
    parser.parse_args()

    total_files, valid_images, invalid_images = audit_raw_dataset()
    print(f"[OK] Audit complete: {total_files} files, {valid_images} valid, {invalid_images} invalid.")
    print(f"[OK] Image report: {RAW_AUDIT}")
    print(f"[OK] Class distribution: {CLASS_DISTRIBUTION}")


if __name__ == "__main__":
    main()
