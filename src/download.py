from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from icrawler.builtin import BingImageCrawler
from icrawler.downloader import ImageDownloader
from PIL import Image

from src.config import BASE_DIR, KEYWORDS, MAX_IMAGES_PER_KEYWORD, METADATA_DIR, MULTI_LEAF_KEYWORDS, RAW_DIR, RAW_MANIFEST


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
BLOCKED_DOMAINS = {
    "deviantart.com",
    "wixmp.com",
    "images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com",
    "pinterest.com",
    "pinimg.com",
    "facebook.com",
    "instagram.com",
    "researchgate.net",
    "wallpapercrafter.com",
    "dreamstime.com",
    "shutterstock.com",
    "alamy.com",
    "istockphoto.com",
}

MANIFEST_FIELDS = [
    "downloaded_at",
    "source",
    "class_name",
    "keyword",
    "relative_path",
    "source_url",
    "sha256",
    "width",
    "height",
    "image_format",
    "image_mode",
    "file_size_bytes",
    "valid_image",
    "error",
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "query"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, str | int | bool]:
    try:
        with Image.open(path) as image:
            image.load()
            return {
                "width": image.width,
                "height": image.height,
                "image_format": image.format or "",
                "image_mode": image.mode,
                "valid_image": True,
                "error": "",
            }
    except Exception as exc:
        return {
            "width": "",
            "height": "",
            "image_format": "",
            "image_mode": "",
            "valid_image": False,
            "error": type(exc).__name__,
        }


def is_blocked_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in BLOCKED_DOMAINS)


class ManifestImageDownloader(ImageDownloader):
    def __init__(
        self,
        thread_num,
        signal,
        session,
        storage,
        *,
        manifest_path: str,
        base_dir: str,
        class_name: str,
        keyword: str,
        source: str,
    ):
        super().__init__(thread_num, signal, session, storage)
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir)
        self.class_name = class_name
        self.keyword = keyword
        self.source = source
        self.manifest_lock = Lock()

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        if is_blocked_url(task.get("file_url", "")):
            self.logger.info("skip blocked domain %s", task.get("file_url", ""))
            task["success"] = False
            task["filename"] = None
            return False

        super().download(task, default_ext, timeout, max_retry, overwrite, **kwargs)
        return task.get("success", False)

    def process_meta(self, task):
        if not task.get("success") or not task.get("filename"):
            return

        file_path = Path(self.storage.root_dir) / task["filename"]
        image_meta = inspect_image(file_path)

        row = {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "class_name": self.class_name,
            "keyword": self.keyword,
            "relative_path": file_path.resolve().relative_to(self.base_dir.resolve()).as_posix(),
            "source_url": task.get("file_url", ""),
            "sha256": sha256_file(file_path) if file_path.exists() else "",
            "width": image_meta["width"],
            "height": image_meta["height"],
            "image_format": image_meta["image_format"],
            "image_mode": image_meta["image_mode"],
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else "",
            "valid_image": image_meta["valid_image"],
            "error": image_meta["error"],
        }

        self._append_manifest_row(row)

    def _append_manifest_row(self, row: dict[str, object]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_lock:
            should_write_header = not self.manifest_path.exists() or self.manifest_path.stat().st_size == 0
            with self.manifest_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
                if should_write_header:
                    writer.writeheader()
                writer.writerow(row)


def parse_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None

    clean_value = value.lower().replace(" ", "")
    if "x" in clean_value:
        width, height = clean_value.split("x", 1)
        return int(width), int(height)

    size = int(clean_value)
    return size, size


def keyword_map_for_group(keyword_group: str) -> dict[str, list[str]]:
    if keyword_group == "base":
        return KEYWORDS
    if keyword_group == "multi_leaf":
        return MULTI_LEAF_KEYWORDS
    if keyword_group == "all":
        merged = {class_name: keywords[:] for class_name, keywords in KEYWORDS.items()}
        for class_name, keywords in MULTI_LEAF_KEYWORDS.items():
            merged.setdefault(class_name, [])
            merged[class_name].extend(keyword for keyword in keywords if keyword not in merged[class_name])
        return merged

    raise ValueError(f"Grup keyword necunoscut: {keyword_group}")


def selected_classes(class_names: list[str] | None, keyword_map: dict[str, list[str]]) -> list[str]:
    if not class_names:
        return list(keyword_map.keys())

    unknown = sorted(set(class_names) - set(keyword_map))
    if unknown:
        available = ", ".join(keyword_map.keys())
        raise ValueError(f"Clase necunoscute: {', '.join(unknown)}. Clase disponibile: {available}")

    return class_names


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def download_images(
    *,
    class_names: list[str] | None = None,
    max_images_per_keyword: int = MAX_IMAGES_PER_KEYWORD,
    min_size: tuple[int, int] | None = (300, 300),
    downloader_threads: int = 2,
    keyword_group: str = "base",
    dry_run: bool = False,
    log_level: int = logging.INFO,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    keyword_map = keyword_map_for_group(keyword_group)

    for class_name in selected_classes(class_names, keyword_map):
        for keyword in keyword_map[class_name]:
            keyword_dir = RAW_DIR / class_name / slugify(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            if dry_run:
                print(f"[DRY-RUN] {class_name}: '{keyword}' -> {keyword_dir}")
                continue

            before_count = count_images(keyword_dir)
            print(f"[INFO] Descarc {max_images_per_keyword} imagini pentru {class_name}: '{keyword}'")

            crawler = BingImageCrawler(
                downloader_cls=ManifestImageDownloader,
                downloader_threads=downloader_threads,
                storage={"root_dir": str(keyword_dir)},
                log_level=log_level,
                extra_downloader_args={
                    "manifest_path": str(RAW_MANIFEST),
                    "base_dir": str(BASE_DIR),
                    "class_name": class_name,
                    "keyword": keyword,
                    "source": "bing_image_search",
                },
            )

            crawler.crawl(
                keyword=keyword,
                max_num=max_images_per_keyword,
                min_size=min_size,
                file_idx_offset="auto",
                overwrite=False,
                max_idle_time=20,
            )

            after_count = count_images(keyword_dir)
            print(f"[OK] {class_name}: +{after_count - before_count} imagini noi in {keyword_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colecteaza imagini brute pentru bolile pomilor fructiferi.")
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Clasele de descarcat. Exemplu: --classes apple_scab strawberry_leaf_scorch tomato_early_blight",
    )
    parser.add_argument(
        "--max-per-keyword",
        type=int,
        default=MAX_IMAGES_PER_KEYWORD,
        help=f"Numar maxim de imagini pentru fiecare keyword. Implicit: {MAX_IMAGES_PER_KEYWORD}",
    )
    parser.add_argument(
        "--min-size",
        default="300x300",
        help="Dimensiune minima acceptata. Exemplu: 300x300, 512 sau none.",
    )
    parser.add_argument("--threads", type=int, default=2, help="Numar thread-uri pentru downloader.")
    parser.add_argument(
        "--keyword-set",
        default="base",
        choices=["base", "multi_leaf", "all"],
        help="Setul de keyword-uri folosit pentru cautare.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Afiseaza ce ar descarca, fara download.")
    parser.add_argument("--quiet", action="store_true", help="Reduce log-urile interne ale crawlerului.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    min_size = None if args.min_size.lower() == "none" else parse_size(args.min_size)
    log_level = logging.WARNING if args.quiet else logging.INFO

    download_images(
        class_names=args.classes,
        max_images_per_keyword=args.max_per_keyword,
        min_size=min_size,
        downloader_threads=args.threads,
        keyword_group=args.keyword_set,
        dry_run=args.dry_run,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
