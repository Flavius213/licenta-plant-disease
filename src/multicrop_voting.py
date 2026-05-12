from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.background_removal import load_model_image, remove_leaf_background


@dataclass(frozen=True)
class CropRegion:
    index: int
    bbox: tuple[int, int, int, int]
    source: str
    area_ratio: float
    weight: float


@dataclass(frozen=True)
class CropPrediction:
    crop_index: int
    bbox: tuple[int, int, int, int]
    source: str
    weight: float
    predicted_class: str
    confidence: float
    top_predictions: list[tuple[str, float]]


@dataclass(frozen=True)
class MultiCropResult:
    final_predictions: list[tuple[str, float]]
    crop_predictions: list[CropPrediction]
    regions: list[CropRegion]
    aggregation_scores: dict[str, float]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def foreground_mask(image: Image.Image, *, backend: str) -> np.ndarray:
    rgba = remove_leaf_background(image, backend=backend)
    alpha = np.array(rgba.getchannel("A"))
    _, mask = cv2.threshold(alpha, 32, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(0, min(left, width - 1)),
        max(0, min(top, height - 1)),
        max(1, min(right, width)),
        max(1, min(bottom, height)),
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    box_width = right - left
    box_height = bottom - top
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return clamp_bbox((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), width, height)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bbox
    return max(0, right - left) * max(0, bottom - top)


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = bbox_area((left, top, right, bottom))
    if intersection <= 0:
        return 0.0
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / max(union, 1)


def crop_weight(source: str, area_ratio: float) -> float:
    source_weight = {
        "full_image": 0.65,
        "component": 1.15,
        "grid": 0.9,
        "center": 1.0,
    }.get(source, 1.0)
    area_weight = float(np.clip(np.sqrt(max(area_ratio, 0.01)) * 1.8, 0.55, 1.35))
    return round(source_weight * area_weight, 4)


def add_region(
    regions: list[CropRegion],
    bbox: tuple[int, int, int, int],
    *,
    source: str,
    image_area: int,
    max_overlap: float = 0.82,
) -> None:
    if bbox_area(bbox) <= 0:
        return
    for existing in regions:
        if existing.source != "full_image" and bbox_iou(existing.bbox, bbox) > max_overlap:
            return
    area_ratio = bbox_area(bbox) / max(image_area, 1)
    regions.append(
        CropRegion(
            index=len(regions),
            bbox=bbox,
            source=source,
            area_ratio=area_ratio,
            weight=crop_weight(source, area_ratio),
        )
    )


def component_regions(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    min_area_ratio: float,
) -> list[tuple[int, int, int, int]]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = width * height
    min_area = max(256, int(image_area * min_area_ratio))
    boxes: list[tuple[int, int, int, int]] = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if box_width < 24 or box_height < 24:
            continue
        boxes.append((left, top, left + box_width, top + box_height))

    boxes.sort(key=bbox_area, reverse=True)
    return boxes


def grid_regions(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    min_mask_ratio: float,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for rows, cols in [(2, 2), (3, 3)]:
        cell_width = width // cols
        cell_height = height // rows
        for row in range(rows):
            for col in range(cols):
                left = col * cell_width
                top = row * cell_height
                right = width if col == cols - 1 else (col + 1) * cell_width
                bottom = height if row == rows - 1 else (row + 1) * cell_height
                cell_mask = mask[top:bottom, left:right]
                if cell_mask.size and float((cell_mask > 0).mean()) >= min_mask_ratio:
                    boxes.append((left, top, right, bottom))
    return boxes


def center_region(width: int, height: int, scale: float = 0.72) -> tuple[int, int, int, int]:
    crop_width = int(width * scale)
    crop_height = int(height * scale)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return (left, top, left + crop_width, top + crop_height)


def generate_crop_regions(
    image_path: Path,
    *,
    max_crops: int = 8,
    detection_backend: str = "opencv",
    min_area_ratio: float = 0.018,
) -> list[CropRegion]:
    image = load_rgb_image(image_path)
    width, height = image.size
    image_area = width * height
    mask = foreground_mask(image, backend=detection_backend)

    regions: list[CropRegion] = []
    add_region(regions, (0, 0, width, height), source="full_image", image_area=image_area)

    for bbox in component_regions(mask, width=width, height=height, min_area_ratio=min_area_ratio):
        expanded = expand_bbox(bbox, width=width, height=height, padding_ratio=0.18)
        add_region(regions, expanded, source="component", image_area=image_area)
        if len(regions) >= max_crops:
            return reindex_regions(regions[:max_crops])

    if len(regions) < max_crops:
        add_region(regions, center_region(width, height), source="center", image_area=image_area, max_overlap=0.92)

    for bbox in grid_regions(mask, width=width, height=height, min_mask_ratio=0.08):
        if len(regions) >= max_crops:
            break
        expanded = expand_bbox(bbox, width=width, height=height, padding_ratio=0.08)
        add_region(regions, expanded, source="grid", image_area=image_area, max_overlap=0.72)

    return reindex_regions(regions[:max_crops])


def reindex_regions(regions: list[CropRegion]) -> list[CropRegion]:
    return [
        CropRegion(
            index=index,
            bbox=region.bbox,
            source=region.source,
            area_ratio=region.area_ratio,
            weight=region.weight,
        )
        for index, region in enumerate(regions)
    ]


def crop_images_for_regions(
    image_path: Path,
    regions: list[CropRegion],
    *,
    remove_background: bool,
    background_backend: str,
) -> list[Image.Image]:
    image = load_model_image(
        image_path,
        remove_background=remove_background,
        backend=background_backend,
    )
    return [image.crop(region.bbox) for region in regions]


def aggregate_crop_predictions(
    *,
    classes: list[str],
    regions: list[CropRegion],
    crop_top_predictions: list[list[tuple[str, float]]],
    top_k: int,
) -> MultiCropResult:
    scores = {class_name: 0.0 for class_name in classes}
    crop_predictions: list[CropPrediction] = []

    for region, predictions in zip(regions, crop_top_predictions):
        predicted_class, confidence = predictions[0]
        vote = confidence * region.weight
        scores[predicted_class] += vote
        crop_predictions.append(
            CropPrediction(
                crop_index=region.index,
                bbox=region.bbox,
                source=region.source,
                weight=region.weight,
                predicted_class=predicted_class,
                confidence=confidence,
                top_predictions=predictions,
            )
        )

    total_score = sum(scores.values()) or 1.0
    final_predictions = sorted(
        ((class_name, score / total_score) for class_name, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return MultiCropResult(
        final_predictions=final_predictions,
        crop_predictions=crop_predictions,
        regions=regions,
        aggregation_scores=scores,
    )
