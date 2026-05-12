from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.background_removal import load_model_image
from src.classifier_model import build_transforms, create_model
from src.config import BASE_DIR, METADATA_DIR
from src.train_classifier import BEST_MODEL_PATH


RAW_AUDIT_DEFAULT = METADATA_DIR / "raw_audit.csv"
PLANTVILLAGE_MANIFEST_DEFAULT = METADATA_DIR / "plantvillage_manifest.csv"
REPORT_DEFAULT = METADATA_DIR / "web_similarity_report.csv"
SUMMARY_DEFAULT = METADATA_DIR / "web_similarity_summary.csv"
QUARANTINE_DIR_DEFAULT = BASE_DIR / "data" / "rejected_similarity"

REPORT_FIELDS = [
    "class_name",
    "relative_path",
    "source",
    "decision",
    "max_similarity",
    "top5_mean_similarity",
    "threshold",
    "action_path",
]

SUMMARY_FIELDS = [
    "class_name",
    "plantvillage_refs",
    "calibration_images",
    "threshold",
    "web_candidates",
    "kept",
    "rejected",
    "skipped",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def is_web_row(row: dict[str, str]) -> bool:
    relative_path = row["relative_path"].lower()
    return "plantvillage" not in relative_path and row.get("valid_image", "True") == "True"


def image_path_from_raw_audit(row: dict[str, str]) -> Path:
    return BASE_DIR / "data" / "raw" / row["relative_path"]


def image_path_from_manifest(row: dict[str, str]) -> Path:
    return BASE_DIR / row["relative_path"]


class FeatureExtractor:
    def __init__(self, checkpoint_path: Path, *, remove_background: bool = False, background_backend: str = "auto"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.classes = checkpoint["classes"]
        self.model_name = checkpoint["model_name"]
        self.img_size = checkpoint["img_size"]
        self.remove_background = remove_background
        self.background_backend = background_backend
        self.model = create_model(self.model_name, len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = build_transforms(img_size=self.img_size, train=False)

    def extract_batch(self, paths: list[Path]) -> np.ndarray:
        tensors = []
        for path in paths:
            try:
                tensors.append(
                    self.transform(
                        load_model_image(
                            path,
                            remove_background=self.remove_background,
                            backend=self.background_backend,
                        )
                    )
                )
            except Exception:
                continue

        if not tensors:
            return np.empty((0, 1), dtype=np.float32)

        batch = torch.stack(tensors)
        with torch.no_grad():
            features = self.model.features(batch)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
            features = torch.nn.functional.normalize(features, dim=1)
        return features.cpu().numpy().astype(np.float32)

    def extract(self, paths: list[Path], batch_size: int) -> np.ndarray:
        chunks = []
        for start in range(0, len(paths), batch_size):
            chunk = self.extract_batch(paths[start : start + batch_size])
            if len(chunk):
                chunks.append(chunk)
        if not chunks:
            return np.empty((0, 1), dtype=np.float32)
        return np.vstack(chunks)


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = min(k, values.size)
    return float(np.partition(values, -k)[-k:].mean())


def compute_threshold(
    *,
    reference_embeddings: np.ndarray,
    calibration_embeddings: np.ndarray,
    percentile: float,
    margin: float,
    min_threshold: float,
) -> float:
    if len(calibration_embeddings) == 0 or len(reference_embeddings) == 0:
        return min_threshold

    similarities = calibration_embeddings @ reference_embeddings.T
    max_similarities = similarities.max(axis=1)
    threshold = float(np.percentile(max_similarities, percentile) - margin)
    return max(min_threshold, threshold)


def safe_move_or_delete(path: Path, *, action: str, quarantine_dir: Path, class_name: str) -> str:
    if action == "report":
        return ""
    if not path.exists():
        return ""

    resolved_path = path.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_path.parents:
        raise ValueError(f"Refuz sa modific fisier in afara proiectului: {resolved_path}")

    if action == "delete":
        path.unlink()
        return "deleted"

    destination_dir = quarantine_dir / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / path.name
    suffix_index = 1
    while destination_path.exists():
        destination_path = destination_dir / f"{path.stem}_{suffix_index}{path.suffix}"
        suffix_index += 1
    shutil.move(str(path), str(destination_path))
    return str(destination_path)


def filter_web_images(
    *,
    raw_audit_path: Path,
    plantvillage_manifest_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    summary_path: Path,
    references_per_class: int,
    calibration_per_class: int,
    percentile: float,
    margin: float,
    min_threshold: float,
    fixed_threshold: float | None,
    topk: int,
    action: str,
    quarantine_dir: Path,
    remove_background_references: bool,
    remove_background_candidates: bool,
    background_backend: str,
    batch_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    raw_rows = read_csv(raw_audit_path)
    plantvillage_rows = read_csv(plantvillage_manifest_path)

    pv_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    web_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in plantvillage_rows:
        pv_by_class[row["class_name"]].append(row)

    for row in raw_rows:
        if is_web_row(row):
            web_by_class[row["class_name"]].append(row)

    reference_extractor = FeatureExtractor(
        checkpoint_path,
        remove_background=remove_background_references,
        background_backend=background_backend,
    )
    candidate_extractor = (
        reference_extractor
        if remove_background_candidates == remove_background_references
        else FeatureExtractor(
            checkpoint_path,
            remove_background=remove_background_candidates,
            background_backend=background_backend,
        )
    )
    report_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for class_name in sorted(web_by_class):
        pv_rows = pv_by_class.get(class_name, [])
        web_rows = web_by_class[class_name]

        if not pv_rows:
            summary_rows.append(
                {
                    "class_name": class_name,
                    "plantvillage_refs": 0,
                    "calibration_images": 0,
                    "threshold": "",
                    "web_candidates": len(web_rows),
                    "kept": 0,
                    "rejected": 0,
                    "skipped": len(web_rows),
                }
            )
            continue

        shuffled_pv = pv_rows[:]
        rng.shuffle(shuffled_pv)
        reference_rows = shuffled_pv[:references_per_class]
        calibration_rows = shuffled_pv[references_per_class : references_per_class + calibration_per_class]

        if not calibration_rows:
            calibration_rows = shuffled_pv[: min(len(shuffled_pv), calibration_per_class)]

        reference_paths = [image_path_from_manifest(row) for row in reference_rows]
        calibration_paths = [image_path_from_manifest(row) for row in calibration_rows]
        web_paths = [image_path_from_raw_audit(row) for row in web_rows]

        reference_embeddings = reference_extractor.extract(reference_paths, batch_size=batch_size)
        calibration_embeddings = (
            np.empty((0, 1), dtype=np.float32)
            if fixed_threshold is not None
            else reference_extractor.extract(calibration_paths, batch_size=batch_size)
        )
        web_embeddings = candidate_extractor.extract(web_paths, batch_size=batch_size)

        threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else compute_threshold(
                reference_embeddings=reference_embeddings,
                calibration_embeddings=calibration_embeddings,
                percentile=percentile,
                margin=margin,
                min_threshold=min_threshold,
            )
        )

        kept = 0
        rejected = 0
        for row, path, embedding in zip(web_rows, web_paths, web_embeddings):
            similarities = embedding @ reference_embeddings.T
            max_similarity = float(similarities.max()) if similarities.size else 0.0
            topk_similarity = topk_mean(similarities, topk)
            decision = "keep" if max_similarity >= threshold else "reject"
            action_path = ""

            if decision == "keep":
                kept += 1
            else:
                rejected += 1
                action_path = safe_move_or_delete(
                    path,
                    action=action,
                    quarantine_dir=quarantine_dir,
                    class_name=class_name,
                )

            report_rows.append(
                {
                    "class_name": class_name,
                    "relative_path": row["relative_path"],
                    "source": "web_scraping",
                    "decision": decision,
                    "max_similarity": round(max_similarity, 6),
                    "top5_mean_similarity": round(topk_similarity, 6),
                    "threshold": round(threshold, 6),
                    "action_path": action_path,
                }
            )

        summary_rows.append(
            {
                "class_name": class_name,
                "plantvillage_refs": len(reference_rows),
                "calibration_images": len(calibration_rows),
                "threshold": round(threshold, 6),
                "web_candidates": len(web_rows),
                "kept": kept,
                "rejected": rejected,
                "skipped": 0,
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

    print(f"[OK] Raport similaritate: {report_path}")
    print(f"[OK] Sumar similaritate: {summary_path}")
    for row in summary_rows:
        print(
            f"[OK] {row['class_name']}: kept={row['kept']} "
            f"rejected={row['rejected']} skipped={row['skipped']} threshold={row['threshold']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtreaza imaginile web prin comparatie cu PlantVillage.")
    parser.add_argument("--raw-audit", default=str(RAW_AUDIT_DEFAULT))
    parser.add_argument("--plantvillage-manifest", default=str(PLANTVILLAGE_MANIFEST_DEFAULT))
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH))
    parser.add_argument("--report", default=str(REPORT_DEFAULT))
    parser.add_argument("--summary", default=str(SUMMARY_DEFAULT))
    parser.add_argument("--references-per-class", type=int, default=100)
    parser.add_argument("--calibration-per-class", type=int, default=200)
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--min-threshold", type=float, default=0.45)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        help="Foloseste un prag fix de similaritate in locul pragului calibrat pe PlantVillage.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quarantine-dir", default=str(QUARANTINE_DIR_DEFAULT))
    parser.add_argument(
        "--remove-background-references",
        action="store_true",
        help="Scoate fundalul si pentru referintele PlantVillage in timpul compararii.",
    )
    parser.add_argument(
        "--remove-background-candidates",
        action="store_true",
        help="Scoate fundalul pentru imaginile web in timpul compararii daca nu au fost deja procesate.",
    )
    parser.add_argument("--background-backend", choices=["auto", "rembg", "opencv"], default="auto")
    parser.add_argument(
        "--action",
        choices=["report", "quarantine", "delete"],
        default="report",
        help="report = nu modifica fisiere, quarantine = muta rejecturile, delete = sterge definitiv.",
    )
    args = parser.parse_args()

    filter_web_images(
        raw_audit_path=Path(args.raw_audit),
        plantvillage_manifest_path=Path(args.plantvillage_manifest),
        checkpoint_path=Path(args.checkpoint),
        report_path=Path(args.report),
        summary_path=Path(args.summary),
        references_per_class=args.references_per_class,
        calibration_per_class=args.calibration_per_class,
        percentile=args.percentile,
        margin=args.margin,
        min_threshold=args.min_threshold,
        fixed_threshold=args.fixed_threshold,
        topk=args.topk,
        action=args.action,
        quarantine_dir=Path(args.quarantine_dir),
        remove_background_references=args.remove_background_references,
        remove_background_candidates=args.remove_background_candidates,
        background_backend=args.background_backend,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
