from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.background_removal import IMAGE_EXTENSIONS, save_leaf_on_checkerboard
from src.config import BASE_DIR, FINAL_DIR, FINAL_DUAL_BACKGROUND_DIR, METADATA_DIR


SPLIT_MANIFEST_DEFAULT = METADATA_DIR / "final_split_manifest.csv"
WEB_BACKGROUND_MANIFEST_DEFAULT = METADATA_DIR / "web_background_manifest.csv"
OUTPUT_MANIFEST_DEFAULT = METADATA_DIR / "dual_background_manifest.csv"
OUTPUT_DISTRIBUTION_DEFAULT = METADATA_DIR / "dual_background_distribution.csv"

MANIFEST_FIELDS = [
    "class_name",
    "split",
    "variant",
    "relative_path",
    "source_relative_path",
    "source",
    "source_variant_path",
    "backend",
]

DISTRIBUTION_FIELDS = ["class_name", "split", "background", "no_background", "total"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def clear_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_output == resolved_base or resolved_base not in resolved_output.parents:
        raise ValueError(f"Refuz sa sterg un folder in afara proiectului: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def safe_project_path(path: Path) -> Path:
    resolved_path = path.resolve()
    resolved_base = BASE_DIR.resolve()
    if resolved_base not in resolved_path.parents and resolved_path != resolved_base:
        raise ValueError(f"Refuz sa folosesc un fisier in afara proiectului: {resolved_path}")
    return resolved_path


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def load_web_background_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    return {
        normalize_relative_path(row["processed_relative_path"]): row
        for row in rows
        if row.get("status") == "processed" and row.get("processed_relative_path")
    }


def copy_image(source_path: Path, destination_path: Path, *, overwrite: bool) -> bool:
    if destination_path.exists() and not overwrite:
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


def make_destination_name(stem: str, variant: str, suffix: str) -> str:
    clean_suffix = suffix.lower() if suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
    return f"{stem}_{variant}{clean_suffix}"


def source_paths_for_row(
    row: dict[str, str],
    *,
    web_background_map: dict[str, dict[str, str]],
) -> tuple[Path, Path | None, bool]:
    final_path = BASE_DIR / normalize_relative_path(row["relative_path"])
    source_relative = normalize_relative_path(row["source_relative_path"])
    source_path = BASE_DIR / source_relative

    if row["source"] == "web_scraping":
        web_row = web_background_map.get(source_relative)
        if web_row and web_row.get("backup_relative_path"):
            backup_path = BASE_DIR / normalize_relative_path(web_row["backup_relative_path"])
            if backup_path.exists():
                no_background_path = source_path if source_path.exists() else final_path
                return backup_path, no_background_path, True

    background_path = source_path if source_path.exists() else final_path
    return background_path, None, False


def build_dual_background_dataset(
    *,
    source_dir: Path,
    split_manifest_path: Path,
    web_background_manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    output_distribution_path: Path,
    backend: str,
    clear: bool,
    overwrite: bool,
    limit: int | None,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Nu exista datasetul sursa: {source_dir}")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Nu exista manifestul split: {split_manifest_path}")

    if clear:
        clear_output_dir(output_dir)

    rows = read_csv(split_manifest_path)
    if limit is not None:
        rows = rows[:limit]

    web_background_map = load_web_background_map(web_background_manifest_path)
    manifest_rows: list[dict[str, str]] = []
    distribution: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for index, row in enumerate(rows, start=1):
        class_name = row["class_name"]
        split = row["split"]
        final_path = BASE_DIR / normalize_relative_path(row["relative_path"])
        if not final_path.exists():
            raise FileNotFoundError(f"Lipseste imaginea din split: {final_path}")

        background_source, existing_no_background_source, no_background_ready = source_paths_for_row(
            row,
            web_background_map=web_background_map,
        )
        background_source = safe_project_path(background_source)
        if existing_no_background_source is not None:
            existing_no_background_source = safe_project_path(existing_no_background_source)

        class_dir = output_dir / split / class_name
        stem = final_path.stem
        background_destination = class_dir / make_destination_name(stem, "background", background_source.suffix)
        no_background_destination = class_dir / make_destination_name(stem, "no_background", ".png")

        copy_image(background_source, background_destination, overwrite=overwrite)
        if no_background_ready and existing_no_background_source and existing_no_background_source.exists():
            copy_image(existing_no_background_source, no_background_destination, overwrite=overwrite)
            no_background_backend = "preprocessed_web"
        else:
            if not no_background_destination.exists() or overwrite:
                no_background_destination.parent.mkdir(parents=True, exist_ok=True)
                save_leaf_on_checkerboard(background_source, no_background_destination, backend=backend)
            no_background_backend = backend

        for variant, destination, source_variant_path, used_backend in [
            ("background", background_destination, background_source, "original"),
            ("no_background", no_background_destination, existing_no_background_source or background_source, no_background_backend),
        ]:
            manifest_rows.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "variant": variant,
                    "relative_path": relative_to_project(destination),
                    "source_relative_path": row["source_relative_path"],
                    "source": row["source"],
                    "source_variant_path": relative_to_project(source_variant_path),
                    "backend": used_backend,
                }
            )
            distribution[(class_name, split)][variant] += 1
            distribution[(class_name, split)]["total"] += 1

        if index % 250 == 0:
            print(f"[INFO] Perechi background/no_background create: {index}/{len(rows)}")

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with output_distribution_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DISTRIBUTION_FIELDS)
        writer.writeheader()
        for class_name, split in sorted(distribution):
            counts = distribution[(class_name, split)]
            writer.writerow(
                {
                    "class_name": class_name,
                    "split": split,
                    "background": counts["background"],
                    "no_background": counts["no_background"],
                    "total": counts["total"],
                }
            )

    print(f"[OK] Dataset dual creat: {output_dir}")
    print(f"[OK] Imagini totale: {len(manifest_rows)}")
    print(f"[OK] Manifest: {output_manifest_path}")
    print(f"[OK] Distributie: {output_distribution_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Creeaza dataset cu fiecare imagine in varianta cu si fara background.")
    parser.add_argument("--source-dir", default=str(FINAL_DIR))
    parser.add_argument("--split-manifest", default=str(SPLIT_MANIFEST_DEFAULT))
    parser.add_argument("--web-background-manifest", default=str(WEB_BACKGROUND_MANIFEST_DEFAULT))
    parser.add_argument("--output-dir", default=str(FINAL_DUAL_BACKGROUND_DIR))
    parser.add_argument("--output-manifest", default=str(OUTPUT_MANIFEST_DEFAULT))
    parser.add_argument("--output-distribution", default=str(OUTPUT_DISTRIBUTION_DEFAULT))
    parser.add_argument("--backend", choices=["auto", "rembg", "opencv"], default="opencv")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Proceseaza doar primele N randuri din manifest.")
    args = parser.parse_args()

    build_dual_background_dataset(
        source_dir=Path(args.source_dir),
        split_manifest_path=Path(args.split_manifest),
        web_background_manifest_path=Path(args.web_background_manifest),
        output_dir=Path(args.output_dir),
        output_manifest_path=Path(args.output_manifest),
        output_distribution_path=Path(args.output_distribution),
        backend=args.backend,
        clear=args.clear,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
