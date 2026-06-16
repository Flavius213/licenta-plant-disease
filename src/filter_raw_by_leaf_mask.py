from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.background_removal import IMAGE_EXTENSIONS, refine_leaf_mask
from src.config import BASE_DIR, METADATA_DIR, RAW_DIR


REPORT_DEFAULT = METADATA_DIR / "leaf_mask_filter_report.csv"
SUMMARY_DEFAULT = METADATA_DIR / "leaf_mask_filter_summary.csv"
QUARANTINE_DIR_DEFAULT = BASE_DIR / "data" / "rejected_leaf_mask"

REPORT_FIELDS = [
    "class_name",
    "relative_path",
    "decision",
    "leaf_ratio",
    "largest_component_ratio",
    "width",
    "height",
    "reason",
    "action_path",
]

SUMMARY_FIELDS = ["class_name", "kept", "rejected", "errors", "total"]


def parse_classes(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {value.strip() for value in values if value.strip()}


def iter_raw_images(raw_dir: Path, classes: set[str] | None) -> list[Path]:
    images: list[Path] = []
    if classes:
        class_dirs = [raw_dir / class_name for class_name in sorted(classes)]
    else:
        class_dirs = [path for path in sorted(raw_dir.iterdir()) if path.is_dir()]

    for class_dir in class_dirs:
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return images


def is_trusted(path: Path, trusted_markers: list[str]) -> bool:
    relative = path.relative_to(RAW_DIR).as_posix().lower()
    return any(marker.lower() in relative for marker in trusted_markers if marker)


def component_ratio(mask: np.ndarray) -> float:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return 0.0
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area / float(mask.shape[0] * mask.shape[1])


def leaf_scores(path: Path, *, max_side: int) -> tuple[float, float, int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
        rgb = np.array(image)

    mask = refine_leaf_mask(rgb, iterations=2, min_area_ratio=0.001)
    leaf_ratio = float(mask.mean() / 255.0)
    largest_ratio = component_ratio(mask)
    return leaf_ratio, largest_ratio, width, height


def safe_quarantine(path: Path, quarantine_dir: Path, class_name: str) -> str:
    resolved_path = path.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_path.parents:
        raise ValueError(f"Refusing to move a file outside the project: {resolved_path}")

    relative = path.relative_to(RAW_DIR)
    destination = quarantine_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix_index = 1
    while destination.exists():
        destination = quarantine_dir / relative.with_name(f"{relative.stem}_{suffix_index}{relative.suffix}")
        suffix_index += 1
    shutil.move(str(path), str(destination))
    return str(destination)


def filter_raw_images(
    *,
    raw_dir: Path,
    classes: set[str] | None,
    report_path: Path,
    summary_path: Path,
    quarantine_dir: Path,
    min_leaf_ratio: float,
    min_largest_component_ratio: float,
    min_size: int,
    max_side: int,
    trusted_markers: list[str],
    action: str,
    limit: int | None,
) -> None:
    images = iter_raw_images(raw_dir, classes)
    if limit is not None:
        images = images[:limit]

    report_rows: list[dict[str, object]] = []
    stats: dict[str, Counter] = defaultdict(Counter)

    for index, path in enumerate(images, start=1):
        class_name = path.relative_to(raw_dir).parts[0]
        relative_path = path.relative_to(raw_dir).as_posix()
        action_path = ""
        reason = ""

        try:
            if is_trusted(path, trusted_markers):
                decision = "trusted_keep"
                leaf_ratio, largest_ratio, width, height = "", "", "", ""
                stats[class_name]["kept"] += 1
            else:
                leaf_ratio, largest_ratio, width, height = leaf_scores(path, max_side=max_side)
                if width < min_size or height < min_size:
                    decision = "reject"
                    reason = "small_image"
                elif leaf_ratio < min_leaf_ratio:
                    decision = "reject"
                    reason = "low_leaf_ratio"
                elif largest_ratio < min_largest_component_ratio:
                    decision = "reject"
                    reason = "low_component_ratio"
                else:
                    decision = "keep"

                if decision == "reject":
                    stats[class_name]["rejected"] += 1
                    if action == "quarantine":
                        action_path = safe_quarantine(path, quarantine_dir, class_name)
                else:
                    stats[class_name]["kept"] += 1
        except Exception as exc:
            decision = "error"
            leaf_ratio, largest_ratio, width, height = "", "", "", ""
            reason = f"{type(exc).__name__}: {exc}"
            stats[class_name]["errors"] += 1

        report_rows.append(
            {
                "class_name": class_name,
                "relative_path": relative_path,
                "decision": decision,
                "leaf_ratio": round(leaf_ratio, 6) if isinstance(leaf_ratio, float) else leaf_ratio,
                "largest_component_ratio": round(largest_ratio, 6)
                if isinstance(largest_ratio, float)
                else largest_ratio,
                "width": width,
                "height": height,
                "reason": reason,
                "action_path": action_path,
            }
        )

        if index % 250 == 0:
            print(f"[INFO] Checked: {index}/{len(images)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for class_name in sorted(stats):
            counts = stats[class_name]
            writer.writerow(
                {
                    "class_name": class_name,
                    "kept": counts["kept"],
                    "rejected": counts["rejected"],
                    "errors": counts["errors"],
                    "total": counts["kept"] + counts["rejected"] + counts["errors"],
                }
            )
            print(
                f"[OK] {class_name}: kept={counts['kept']} "
                f"rejected={counts['rejected']} errors={counts['errors']}"
            )

    print(f"[OK] Leaf filter report: {report_path}")
    print(f"[OK] Leaf filter summary: {summary_path}")
    if action == "quarantine":
        print(f"[OK] Rejected files moved to: {quarantine_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw images that do not appear to contain leaves.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--report", default=str(REPORT_DEFAULT))
    parser.add_argument("--summary", default=str(SUMMARY_DEFAULT))
    parser.add_argument("--quarantine-dir", default=str(QUARANTINE_DIR_DEFAULT))
    parser.add_argument("--min-leaf-ratio", type=float, default=0.04)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.015)
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--trusted-marker", action="append", default=["plantvillage", "real_photos_labeled"])
    parser.add_argument("--action", choices=["report", "quarantine"], default="report")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    filter_raw_images(
        raw_dir=Path(args.raw_dir),
        classes=parse_classes(args.classes),
        report_path=Path(args.report),
        summary_path=Path(args.summary),
        quarantine_dir=Path(args.quarantine_dir),
        min_leaf_ratio=args.min_leaf_ratio,
        min_largest_component_ratio=args.min_largest_component_ratio,
        min_size=args.min_size,
        max_side=args.max_side,
        trusted_markers=args.trusted_marker,
        action=args.action,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
