from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import METADATA_DIR


RAW_AUDIT_DEFAULT = METADATA_DIR / "raw_audit.csv"
OUTPUT_DEFAULT = METADATA_DIR / "training_manifest.csv"

FIELDS = ["class_name", "relative_path", "source", "width", "height", "sha256", "phash"]


def parse_classes(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def infer_source(relative_path: str) -> str:
    return "plantvillage" if "plantvillage" in relative_path.lower() else "web_scraping"


def load_similarity_keep_set(path: Path | None) -> set[str] | None:
    if not path:
        return None
    keep_set: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["decision"] == "keep":
                keep_set.add(row["relative_path"])
    return keep_set


def build_manifest(
    *,
    raw_audit: Path,
    output_path: Path,
    classes: set[str] | None,
    min_size: int,
    keep_phash_duplicates: bool,
    similarity_keep_set: set[str] | None = None,
) -> int:
    rows_out: list[dict[str, object]] = []

    with raw_audit.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_name = row["class_name"]
            if classes and class_name not in classes:
                continue
            if row["valid_image"] != "True":
                continue
            if row["exact_duplicate_of"]:
                continue
            if row["phash_duplicate_of"] and not keep_phash_duplicates:
                continue
            if int(row["width"] or 0) < min_size or int(row["height"] or 0) < min_size:
                continue

            raw_relative_path = row["relative_path"]
            source = infer_source(raw_relative_path)
            if similarity_keep_set is not None and source == "web_scraping" and raw_relative_path not in similarity_keep_set:
                continue

            rows_out.append(
                {
                    "class_name": class_name,
                    "relative_path": f"data/raw/{raw_relative_path}",
                    "source": source,
                    "width": row["width"],
                    "height": row["height"],
                    "sha256": row["sha256"],
                    "phash": row["phash"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    return len(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construieste un manifest curatat din data/metadata/raw_audit.csv.")
    parser.add_argument("--raw-audit", default=str(RAW_AUDIT_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument(
        "--classes",
        default=(
            "apple_black_rot,apple_cedar_apple_rust,apple_healthy,"
            "apple_scab,cherry_healthy,cherry_powdery_mildew"
        ),
        help="Clase separate prin virgula.",
    )
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--keep-phash-duplicates", action="store_true")
    parser.add_argument(
        "--similarity-report",
        help="Optional: include doar imaginile web marcate keep in raportul de similaritate.",
    )
    args = parser.parse_args()

    count = build_manifest(
        raw_audit=Path(args.raw_audit),
        output_path=Path(args.output),
        classes=parse_classes(args.classes),
        min_size=args.min_size,
        keep_phash_duplicates=args.keep_phash_duplicates,
        similarity_keep_set=load_similarity_keep_set(Path(args.similarity_report)) if args.similarity_report else None,
    )
    print(f"[OK] Manifest creat: {args.output}")
    print(f"[OK] Imagini selectate: {count}")


if __name__ == "__main__":
    main()
