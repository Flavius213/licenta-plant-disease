from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.config import BASE_DIR, FINAL_DIR, METADATA_DIR


SPLIT_MANIFEST = METADATA_DIR / "final_split_manifest.csv"
SPLIT_DISTRIBUTION = METADATA_DIR / "final_split_distribution.csv"

MANIFEST_FIELDS = [
    "class_name",
    "split",
    "relative_path",
    "source_relative_path",
    "source",
]

DISTRIBUTION_FIELDS = ["class_name", "train", "val", "test", "total"]


def read_source_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def clean_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def clear_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_output == resolved_base or resolved_base not in resolved_output.parents:
        raise ValueError(f"Refusing to delete a folder outside the project: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def build_split(
    *,
    source_manifest: Path,
    output_dir: Path = FINAL_DIR,
    max_per_class: int | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    clear_output: bool = False,
    dry_run: bool = False,
) -> None:
    rows = read_source_manifest(source_manifest)
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row["class_name"]].append(row)

    if not rows_by_class:
        raise ValueError(f"The manifest does not contain images: {source_manifest}")

    random_generator = random.Random(seed)
    selected_by_class: dict[str, list[dict[str, str]]] = {}
    for class_name, class_rows in rows_by_class.items():
        shuffled_rows = class_rows[:]
        random_generator.shuffle(shuffled_rows)
        selected_by_class[class_name] = shuffled_rows[:max_per_class] if max_per_class else shuffled_rows

    split_rows: list[dict[str, str]] = []
    distribution = defaultdict(Counter)

    if clear_output and not dry_run:
        clear_output_dir(output_dir)

    for class_name, class_rows in sorted(selected_by_class.items()):
        train_count, val_count, _test_count = split_counts(len(class_rows), train_ratio, val_ratio)
        split_assignments = (
            [("train", row) for row in class_rows[:train_count]]
            + [("val", row) for row in class_rows[train_count : train_count + val_count]]
            + [("test", row) for row in class_rows[train_count + val_count :]]
        )

        for index, (split_name, row) in enumerate(split_assignments, start=1):
            source_path = BASE_DIR / row["relative_path"]
            source = row.get("source") or "dataset"
            destination_name = clean_filename(f"{source}_{index:06d}{source_path.suffix.lower()}")
            destination_path = output_dir / split_name / class_name / destination_name
            relative_destination = destination_path.relative_to(BASE_DIR).as_posix()

            split_rows.append(
                {
                    "class_name": class_name,
                    "split": split_name,
                    "relative_path": relative_destination,
                    "source_relative_path": row["relative_path"],
                    "source": source,
                }
            )
            distribution[class_name][split_name] += 1
            distribution[class_name]["total"] += 1

            if not dry_run:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                if not destination_path.exists():
                    shutil.copy2(source_path, destination_path)

    print(f"[INFO] Classes: {len(selected_by_class)}")
    print(f"[INFO] Selected images: {sum(len(rows) for rows in selected_by_class.values())}")

    if dry_run:
        for class_name in sorted(distribution):
            counts = distribution[class_name]
            print(
                f"[DRY-RUN] {class_name}: train={counts['train']}, "
                f"val={counts['val']}, test={counts['test']}, total={counts['total']}"
            )
        return

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with SPLIT_MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(split_rows)

    with SPLIT_DISTRIBUTION.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DISTRIBUTION_FIELDS)
        writer.writeheader()
        for class_name in sorted(distribution):
            counts = distribution[class_name]
            writer.writerow(
                {
                    "class_name": class_name,
                    "train": counts["train"],
                    "val": counts["val"],
                    "test": counts["test"],
                    "total": counts["total"],
                }
            )

    print(f"[OK] Split created in: {output_dir}")
    print(f"[OK] Manifest split: {SPLIT_MANIFEST}")
    print(f"[OK] Split distribution: {SPLIT_DISTRIBUTION}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a train/val/test split from an image manifest.")
    parser.add_argument(
        "--source-manifest",
        default=str(METADATA_DIR / "plantvillage_manifest.csv"),
        help="Source manifest. Default: data/metadata/plantvillage_manifest.csv",
    )
    parser.add_argument("--max-per-class", type=int, help="Optional limit for class balancing.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true", help="Delete the old split before copying.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_split(
        source_manifest=Path(args.source_manifest),
        max_per_class=args.max_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
