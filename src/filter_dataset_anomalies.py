from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import BASE_DIR, FINAL_AUGMENTED_DIR, METADATA_DIR
from src.filter_web_by_plantvillage import FeatureExtractor, topk_mean
from src.train_classifier import BEST_MODEL_PATH


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "val", "test"}

REPORT_DEFAULT = METADATA_DIR / "dataset_anomaly_report.csv"
SUMMARY_DEFAULT = METADATA_DIR / "dataset_anomaly_summary.csv"
QUARANTINE_DIR_DEFAULT = BASE_DIR / "data" / "rejected_anomalies"

REPORT_FIELDS = [
    "class_name",
    "split",
    "relative_path",
    "decision",
    "max_similarity",
    "top5_mean_similarity",
    "threshold",
    "action_path",
]

SUMMARY_FIELDS = [
    "class_name",
    "images",
    "references",
    "threshold",
    "kept",
    "rejected",
]


@dataclass(frozen=True)
class ImageEntry:
    class_name: str
    split: str
    path: Path


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_dataset_images(dataset_dir: Path) -> dict[str, list[ImageEntry]]:
    by_class: dict[str, list[ImageEntry]] = {}
    roots = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    split_roots = [path for path in roots if path.name in SPLIT_NAMES]
    if split_roots:
        for split_dir in split_roots:
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                entries = by_class.setdefault(class_dir.name, [])
                entries.extend(
                    ImageEntry(class_name=class_dir.name, split=split_dir.name, path=image_path)
                    for image_path in iter_images(class_dir)
                )
        return by_class

    for class_dir in roots:
        entries = by_class.setdefault(class_dir.name, [])
        entries.extend(
            ImageEntry(class_name=class_dir.name, split="", path=image_path)
            for image_path in iter_images(class_dir)
        )
    return by_class


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def is_trusted_entry(entry: ImageEntry, trusted_name_contains: list[str]) -> bool:
    return any(
        marker and marker.lower() in entry.path.name.lower()
        for marker in trusted_name_contains
    )


def safe_move_or_delete(path: Path, *, action: str, quarantine_dir: Path, entry: ImageEntry) -> str:
    if action == "report":
        return ""
    if not path.exists():
        return ""

    resolved_path = path.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_path.parents:
        raise ValueError(f"Refusing to modify a file outside the project: {resolved_path}")

    if action == "delete":
        path.unlink()
        return "deleted"

    parts = [quarantine_dir]
    if entry.split:
        parts.append(Path(entry.split))
    parts.append(Path(entry.class_name))
    destination_dir = Path(*parts)
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / path.name
    suffix_index = 1
    while destination_path.exists():
        destination_path = destination_dir / f"{path.stem}_{suffix_index}{path.suffix}"
        suffix_index += 1

    shutil.move(str(path), str(destination_path))
    return str(destination_path)


def compare_to_same_class_references(
    *,
    dataset_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    summary_path: Path,
    references_per_class: int,
    threshold: float,
    topk: int,
    action: str,
    quarantine_dir: Path,
    trusted_name_contains: list[str],
    batch_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    images_by_class = collect_dataset_images(dataset_dir)
    extractor = FeatureExtractor(checkpoint_path)

    report_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for class_name in sorted(images_by_class):
        entries = images_by_class[class_name]
        if not entries:
            continue

        trusted_entries = [
            entry for entry in entries if is_trusted_entry(entry, trusted_name_contains)
        ]
        candidate_entries = [
            entry for entry in entries if not is_trusted_entry(entry, trusted_name_contains)
        ]
        candidate_paths = [entry.path for entry in candidate_entries]
        reference_pool = trusted_entries if trusted_entries else entries
        shuffled_reference_pool = reference_pool[:]
        rng.shuffle(shuffled_reference_pool)
        reference_entries = shuffled_reference_pool[: min(references_per_class, len(shuffled_reference_pool))]
        reference_paths = [entry.path for entry in reference_entries]

        reference_embeddings = extractor.extract(reference_paths, batch_size=batch_size)
        image_embeddings = extractor.extract(candidate_paths, batch_size=batch_size)

        if len(reference_embeddings) == 0:
            summary_rows.append(
                {
                    "class_name": class_name,
                    "images": len(entries),
                    "references": len(reference_entries),
                    "threshold": threshold,
                    "kept": len(trusted_entries),
                    "rejected": len(candidate_entries),
                }
            )
            continue

        reference_path_strings = [str(path.resolve()) for path in reference_paths]
        kept = len(trusted_entries)
        rejected = 0

        for entry in trusted_entries:
            report_rows.append(
                {
                    "class_name": class_name,
                    "split": entry.split,
                    "relative_path": relative_to_project(entry.path),
                    "decision": "trusted_keep",
                    "max_similarity": "",
                    "top5_mean_similarity": "",
                    "threshold": threshold,
                    "action_path": "",
                }
            )

        for entry, embedding in zip(candidate_entries, image_embeddings):
            candidate_references = reference_embeddings
            resolved_entry_path = str(entry.path.resolve())
            if resolved_entry_path in reference_path_strings and len(reference_embeddings) > 1:
                keep_indices = [
                    index
                    for index, reference_path in enumerate(reference_path_strings)
                    if reference_path != resolved_entry_path
                ]
                candidate_references = reference_embeddings[keep_indices]

            similarities = embedding @ candidate_references.T
            max_similarity = float(similarities.max()) if similarities.size else 0.0
            topk_similarity = topk_mean(similarities, topk)
            decision = "keep" if max_similarity >= threshold else "reject"
            action_path = ""

            if decision == "keep":
                kept += 1
            else:
                rejected += 1
                action_path = safe_move_or_delete(
                    entry.path,
                    action=action,
                    quarantine_dir=quarantine_dir,
                    entry=entry,
                )

            report_rows.append(
                {
                    "class_name": class_name,
                    "split": entry.split,
                    "relative_path": relative_to_project(entry.path),
                    "decision": decision,
                    "max_similarity": round(max_similarity, 6),
                    "top5_mean_similarity": round(topk_similarity, 6),
                    "threshold": threshold,
                    "action_path": action_path,
                }
            )

        summary_rows.append(
            {
                "class_name": class_name,
                "images": len(entries),
                "references": len(reference_entries),
                "threshold": threshold,
                "kept": kept,
                "rejected": rejected,
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Anomaly report: {report_path}")
    print(f"[OK] Anomaly summary: {summary_path}")
    for row in summary_rows:
        print(
            f"[OK] {row['class_name']}: kept={row['kept']} "
            f"rejected={row['rejected']} refs={row['references']} threshold={row['threshold']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare images from each class with random references from the same class."
    )
    parser.add_argument("--dataset-dir", default=str(FINAL_AUGMENTED_DIR))
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH))
    parser.add_argument("--report", default=str(REPORT_DEFAULT))
    parser.add_argument("--summary", default=str(SUMMARY_DEFAULT))
    parser.add_argument("--references-per-class", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quarantine-dir", default=str(QUARANTINE_DIR_DEFAULT))
    parser.add_argument(
        "--trusted-name-contains",
        action="append",
        default=["plantvillage"],
        help="Automatically keep images whose file name contains this text. Can be repeated.",
    )
    parser.add_argument(
        "--action",
        choices=["report", "quarantine", "delete"],
        default="report",
        help="report = do not modify files, quarantine = move rejected files, delete = permanently delete them.",
    )
    args = parser.parse_args()

    compare_to_same_class_references(
        dataset_dir=Path(args.dataset_dir),
        checkpoint_path=Path(args.checkpoint),
        report_path=Path(args.report),
        summary_path=Path(args.summary),
        references_per_class=args.references_per_class,
        threshold=args.threshold,
        topk=args.topk,
        action=args.action,
        quarantine_dir=Path(args.quarantine_dir),
        trusted_name_contains=args.trusted_name_contains,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
