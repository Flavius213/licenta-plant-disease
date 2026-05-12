from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import BASE_DIR, FINAL_AUGMENTED_DIR, FINAL_DIR, IMG_SIZE


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clear_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_output == resolved_base or resolved_base not in resolved_output.parents:
        raise ValueError(f"Refuz sa sterg un folder in afara proiectului: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_split(source_dir: Path, output_dir: Path, split: str) -> None:
    split_dir = source_dir / split
    if not split_dir.exists():
        return
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        destination_class_dir = output_dir / split / class_dir.name
        destination_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in iter_images(class_dir):
            destination_path = destination_class_dir / image_path.name
            if not destination_path.exists():
                shutil.copy2(image_path, destination_path)


def random_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    scale = rng.uniform(0.72, 1.0)
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    left = rng.randint(0, max(width - crop_width, 0))
    top = rng.randint(0, max(height - crop_height, 0))
    return image.crop((left, top, left + crop_width, top + crop_height))


def augment_image(image_path: Path, rng: random.Random, img_size: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")

    image = random_crop(image, rng)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.2:
        image = ImageOps.flip(image)

    angle = rng.uniform(-28, 28)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.28))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.82, 1.22))
    if rng.random() < 0.2:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.7, 1.8))
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))

    image = ImageOps.fit(image, (img_size, img_size), method=Image.Resampling.BICUBIC)
    return image


def class_counts(output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                counts[class_dir.name] += len(iter_images(class_dir))
    return dict(counts)


def trim_augmented_files(class_dir: Path, target_count: int) -> int:
    files = iter_images(class_dir)
    excess = len(files) - target_count
    if excess <= 0:
        return 0

    augmented_files = sorted((path for path in files if path.name.startswith("aug_")), reverse=True)
    deleted = 0
    for path in augmented_files[:excess]:
        path.unlink()
        deleted += 1
    return deleted


def augment_to_target(
    source_dir: Path,
    output_dir: Path,
    *,
    target_total: int,
    img_size: int,
    seed: int,
    clear: bool,
    trim_extra: bool,
) -> None:
    if clear:
        clear_output_dir(output_dir)

    copy_split(source_dir, output_dir, "train")
    copy_split(source_dir, output_dir, "val")
    copy_split(source_dir, output_dir, "test")

    train_dir = output_dir / "train"
    class_dirs = sorted(path for path in train_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"Nu exista clase in {train_dir}")

    fixed_val_test_count = 0
    for split in ["val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            fixed_val_test_count += sum(len(iter_images(class_dir)) for class_dir in split_dir.iterdir() if class_dir.is_dir())

    train_budget = max(len(class_dirs), target_total - fixed_val_test_count)
    base_target_train_per_class, remainder = divmod(train_budget, len(class_dirs))
    target_train_by_class = {
        class_dir.name: base_target_train_per_class + (1 if index < remainder else 0)
        for index, class_dir in enumerate(class_dirs)
    }
    rng = random.Random(seed)
    total_created = 0

    for class_dir in class_dirs:
        originals = iter_images(class_dir)
        if not originals:
            continue

        current_count = len(originals)
        target_train_count = target_train_by_class[class_dir.name]
        if trim_extra and current_count > target_train_count:
            deleted = trim_augmented_files(class_dir, target_train_count)
            current_count -= deleted
            originals = iter_images(class_dir)
            print(f"[INFO] {class_dir.name}: surplus sters={deleted}")

        needed = max(0, target_train_count - current_count)
        print(f"[INFO] {class_dir.name}: existent train={current_count}, augmentari necesare={needed}")

        for index in range(needed):
            source_path = originals[index % len(originals)]
            image = augment_image(source_path, rng, img_size)
            output_path = class_dir / f"aug_{index + 1:06d}_{source_path.stem}.jpg"
            image.save(output_path, "JPEG", quality=90, optimize=True)
            total_created += 1

    counts = class_counts(output_dir)
    print(f"[OK] Augmentari create: {total_created}")
    print(f"[OK] Total final aproximativ: {sum(counts.values())}")
    for class_name, count in sorted(counts.items()):
        print(f"[OK] {class_name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Creeaza un dataset augmentat pana la un total dorit.")
    parser.add_argument("--source-dir", default=str(FINAL_DIR))
    parser.add_argument("--output-dir", default=str(FINAL_AUGMENTED_DIR))
    parser.add_argument("--target-total", type=int, default=50000)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--trim-extra", action="store_true", help="Sterge augmentarile in plus daca o clasa depaseste tinta.")
    args = parser.parse_args()

    augment_to_target(
        Path(args.source_dir),
        Path(args.output_dir),
        target_total=args.target_total,
        img_size=args.img_size,
        seed=args.seed,
        clear=args.clear,
        trim_extra=args.trim_extra,
    )


if __name__ == "__main__":
    main()
