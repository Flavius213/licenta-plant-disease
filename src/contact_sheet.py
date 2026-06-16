from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.config import METADATA_DIR, RAW_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def iter_class_images(class_dir: Path):
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_thumbnail(path: Path, thumb_size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        left = (thumb_size - image.width) // 2
        top = (thumb_size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def make_contact_sheet(class_name: str, *, limit: int, thumb_size: int, columns: int, output_dir: Path) -> Path | None:
    class_dir = RAW_DIR / class_name
    image_paths = list(iter_class_images(class_dir))[:limit]
    if not image_paths:
        return None

    label_height = 34
    cell_width = thumb_size
    cell_height = thumb_size + label_height
    rows = ceil(len(image_paths) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, path in enumerate(image_paths):
        row = index // columns
        column = index % columns
        x = column * cell_width
        y = row * cell_height
        thumb = load_thumbnail(path, thumb_size)
        sheet.paste(thumb, (x, y))
        relative_label = path.relative_to(RAW_DIR / class_name).as_posix()
        label = f"{index + 1}: {relative_label}"[:32]
        draw.text((x + 4, y + thumb_size + 4), label, fill="black", font=font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{class_name}.jpg"
    sheet.save(output_path, quality=90)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contact sheets for manual image review.")
    parser.add_argument("--classes", nargs="+", help="Classes to render. Default: all classes from data/raw.")
    parser.add_argument("--limit", type=int, default=120, help="Maximum number of images per class.")
    parser.add_argument("--thumb-size", type=int, default=160, help="Thumbnail size.")
    parser.add_argument("--columns", type=int, default=5, help="Number of columns in the contact sheet.")
    args = parser.parse_args()

    if args.classes:
        class_names = args.classes
    else:
        class_names = sorted(path.name for path in RAW_DIR.iterdir() if path.is_dir())

    output_dir = METADATA_DIR / "contact_sheets"
    generated = 0
    for class_name in class_names:
        output_path = make_contact_sheet(
            class_name,
            limit=args.limit,
            thumb_size=args.thumb_size,
            columns=args.columns,
            output_dir=output_dir,
        )
        if output_path:
            generated += 1
            print(f"[OK] {class_name}: {output_path}")
        else:
            print(f"[SKIP] {class_name}: nu exista imagini")

    print(f"[OK] Contact sheets generate: {generated}")


if __name__ == "__main__":
    main()
