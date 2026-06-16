from __future__ import annotations

import argparse
from io import BytesIO
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import BASE_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_BACKGROUND = (245, 245, 245)
CHECKER_LIGHT = (245, 245, 245)
CHECKER_DARK = (205, 205, 205)
CHECKER_TILE_SIZE = 18
BACKGROUND_BACKENDS = {"auto", "rembg", "opencv"}
_REMBG_SESSION = None


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (*MODEL_BACKGROUND, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    return np.array(image)


def rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session

        _REMBG_SESSION = new_session("isnet-general-use")
    return _REMBG_SESSION


def remove_leaf_background_rembg(image: Image.Image) -> Image.Image:
    from rembg import remove

    image = ImageOps.exif_transpose(image).convert("RGBA")
    output = remove(
        image,
        session=rembg_session(),
        post_process_mask=True,
    )
    if isinstance(output, Image.Image):
        return output.convert("RGBA")
    return Image.open(BytesIO(output)).convert("RGBA")


def initial_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)

    red = rgb[:, :, 0].astype(np.int16)
    green_channel = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    excess_green = (2 * green_channel) - red - blue

    green_leaf = (hue >= 20) & (hue <= 100) & (saturation >= 24) & (value >= 30)
    yellow_or_brown_leaf = (hue >= 5) & (hue <= 45) & (saturation >= 30) & (value >= 25)
    dark_leaf_texture = (saturation >= 20) & (value >= 20) & (excess_green >= -45) & (hue <= 105)
    pale_leaf = (excess_green >= -10) & (saturation >= 12) & (value >= 80)

    plain_light_background = (value >= 218) & (saturation <= 35)
    plain_dark_background = (value <= 18) & (saturation <= 40)
    likely_leaf = (green_leaf | yellow_or_brown_leaf | dark_leaf_texture | pale_leaf)
    return (likely_leaf & ~plain_light_background & ~plain_dark_background).astype(np.uint8)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask

    flood = mask.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_leaf_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(64, int(image_area * min_area_ratio))
    kept = np.zeros_like(mask)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept[labels == label] = 255

    if not kept.any():
        kept[labels == largest_label] = 255
    return kept


def refine_leaf_mask(rgb: np.ndarray, *, iterations: int = 5, min_area_ratio: float = 0.001) -> np.ndarray:
    rough = initial_leaf_mask(rgb)
    height, width = rough.shape
    rough_ratio = float(rough.mean())

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if rough_ratio < 0.01:
        margin_x = max(1, int(width * 0.06))
        margin_y = max(1, int(height * 0.06))
        rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))
        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.grabCut(bgr, grabcut_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    else:
        grabcut_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        grabcut_mask[rough == 1] = cv2.GC_PR_FGD

        kernel = np.ones((5, 5), np.uint8)
        sure_foreground = cv2.erode(rough, kernel, iterations=1)
        grabcut_mask[sure_foreground == 1] = cv2.GC_FGD

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        sure_background = ((value >= 230) & (saturation <= 28)).astype(np.uint8)

        border = max(2, min(height, width) // 35)
        sure_background[:border, :] = 1
        sure_background[-border:, :] = 1
        sure_background[:, :border] = 1
        sure_background[:, -border:] = 1
        grabcut_mask[sure_background == 1] = cv2.GC_BGD

        cv2.grabCut(bgr, grabcut_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    if refined.mean() < max(rough.mean() * 40, 1):
        refined = (rough * 255).astype(np.uint8)

    close_kernel = np.ones((5, 5), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel, iterations=1)
    refined = fill_mask_holes(refined)
    refined = keep_leaf_components(refined, min_area_ratio=min_area_ratio)
    return cv2.GaussianBlur(refined, (5, 5), 0)


def remove_leaf_background_opencv(image: Image.Image) -> Image.Image:
    rgb = image_to_rgb_array(image)
    alpha = refine_leaf_mask(rgb)
    alpha_float = (alpha.astype(np.float32) / 255.0)[:, :, None]
    neutral_background = np.full_like(rgb, MODEL_BACKGROUND, dtype=np.uint8)
    stored_rgb = (rgb.astype(np.float32) * alpha_float) + (
        neutral_background.astype(np.float32) * (1.0 - alpha_float)
    )
    rgba = np.dstack([stored_rgb.astype(np.uint8), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def remove_leaf_background(image: Image.Image, *, backend: str = "auto") -> Image.Image:
    if backend not in BACKGROUND_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose one of {sorted(BACKGROUND_BACKENDS)}")

    if backend in {"auto", "rembg"}:
        try:
            return remove_leaf_background_rembg(image)
        except Exception:
            if backend == "rembg":
                raise

    return remove_leaf_background_opencv(image)


def transparent_leaf_image(path: Path, *, backend: str = "auto") -> Image.Image:
    with Image.open(path) as image:
        return remove_leaf_background(image, backend=backend)


def checkerboard_background(
    size: tuple[int, int],
    *,
    tile_size: int = CHECKER_TILE_SIZE,
    light: tuple[int, int, int] = CHECKER_LIGHT,
    dark: tuple[int, int, int] = CHECKER_DARK,
) -> Image.Image:
    width, height = size
    y_indices, x_indices = np.indices((height, width))
    squares = ((x_indices // tile_size) + (y_indices // tile_size)) % 2 == 0
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[squares] = light
    background[~squares] = dark
    return Image.fromarray(background, mode="RGB")


def composite_for_model(image: Image.Image, background_color: tuple[int, int, int] = MODEL_BACKGROUND) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (*background_color, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def composite_on_checkerboard(image: Image.Image, *, tile_size: int = CHECKER_TILE_SIZE) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = checkerboard_background(image.size, tile_size=tile_size).convert("RGBA")
    background.alpha_composite(image)
    return background.convert("RGB")


def leaf_on_checkerboard_image(
    path: Path,
    *,
    tile_size: int = CHECKER_TILE_SIZE,
    backend: str = "auto",
) -> Image.Image:
    return composite_on_checkerboard(transparent_leaf_image(path, backend=backend), tile_size=tile_size)


def load_model_image(path: Path, *, remove_background: bool = False, backend: str = "auto") -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if remove_background:
            return composite_on_checkerboard(remove_leaf_background(image, backend=backend))
        if image.mode == "RGBA":
            return composite_for_model(image)
        return image.convert("RGB")


def save_transparent_leaf(input_path: Path, output_path: Path, *, backend: str = "auto") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transparent_leaf_image(input_path, backend=backend).save(output_path, "PNG", optimize=True)


def save_leaf_on_checkerboard(input_path: Path, output_path: Path, *, backend: str = "auto") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaf_on_checkerboard_image(input_path, backend=backend).save(output_path, "PNG", optimize=True)


def clear_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_output == resolved_base or resolved_base not in resolved_output.parents:
        raise ValueError(f"Refusing to delete a folder outside the project: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    clear: bool,
    overwrite: bool,
    transparent: bool,
    backend: str,
) -> int:
    if clear:
        clear_output_dir(output_dir)

    converted = 0
    for image_path in iter_images(source_dir):
        relative_path = image_path.relative_to(source_dir)
        output_path = output_dir / relative_path.with_suffix(".png")
        if output_path.exists() and not overwrite:
            continue
        if transparent:
            save_transparent_leaf(image_path, output_path, backend=backend)
        else:
            save_leaf_on_checkerboard(image_path, output_path, backend=backend)
        converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove the background and place the leaf/leaves on a gray-white checkerboard.")
    parser.add_argument("--input", help="Source image.")
    parser.add_argument("--output", help="Output PNG image.")
    parser.add_argument("--source-dir", help="Source folder for batch processing.")
    parser.add_argument("--output-dir", help="Destination folder for batch processing.")
    parser.add_argument("--clear", action="store_true", help="Delete output-dir before batch processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG images.")
    parser.add_argument("--transparent", action="store_true", help="Save a transparent PNG instead of a gray-white checkerboard.")
    parser.add_argument("--backend", choices=sorted(BACKGROUND_BACKENDS), default="auto")
    args = parser.parse_args()

    if args.input or args.output:
        if not args.input or not args.output:
            parser.error("--input and --output must be used together.")
        if args.transparent:
            save_transparent_leaf(Path(args.input), Path(args.output), backend=args.backend)
        else:
            save_leaf_on_checkerboard(Path(args.input), Path(args.output), backend=args.backend)
        print(f"[OK] Image saved: {args.output}")
        return

    if not args.source_dir or not args.output_dir:
        parser.error("Use either --input/--output or --source-dir/--output-dir.")

    converted = process_directory(
        Path(args.source_dir),
        Path(args.output_dir),
        clear=args.clear,
        overwrite=args.overwrite,
        transparent=args.transparent,
        backend=args.backend,
    )
    print(f"[OK] Processed images: {converted}")


if __name__ == "__main__":
    main()
