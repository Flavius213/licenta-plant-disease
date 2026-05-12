from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import RAW_DIR, USER_FEEDBACK_DIR
from src.download import IMAGE_EXTENSIONS, slugify


def import_feedback_images(source_dir: Path, class_name: str) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Folderul nu exista: {source_dir}")

    destination_dir = RAW_DIR / class_name / "user_feedback"
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in sorted(source_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        destination_name = f"user_{slugify(image_path.stem)}_{copied + 1:04d}{image_path.suffix.lower()}"
        destination_path = destination_dir / destination_name
        shutil.copy2(image_path, destination_path)
        copied += 1

    return copied


def import_saved_feedback() -> int:
    total = 0
    if not USER_FEEDBACK_DIR.exists():
        return 0

    for class_dir in sorted(path for path in USER_FEEDBACK_DIR.iterdir() if path.is_dir()):
        total += import_feedback_images(class_dir, class_dir.name)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa poze corect etichetate de utilizator in data/raw.")
    parser.add_argument("--source", help="Folder cu imagini de importat.")
    parser.add_argument("--class-name", help="Clasa corecta pentru --source.")
    parser.add_argument(
        "--saved-feedback",
        action="store_true",
        help="Importa toate pozele salvate de interfata in data/user_feedback/<clasa>.",
    )
    args = parser.parse_args()

    if args.saved_feedback:
        copied = import_saved_feedback()
    else:
        if not args.source or not args.class_name:
            parser.error("--source si --class-name sunt necesare daca nu folosesti --saved-feedback")
        copied = import_feedback_images(Path(args.source), args.class_name)

    print(f"[OK] Imagini feedback importate: {copied}")


if __name__ == "__main__":
    main()
