from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.background_removal import IMAGE_EXTENSIONS, load_model_image
from src.multicrop_voting import aggregate_crop_predictions, generate_crop_regions
from src.predict_image import LoadedClassifier
from src.train_classifier import BEST_MODEL_PATH


CONFIGS = [
    {
        "id": "simple",
        "label": "Simplu",
        "remove_background": False,
        "multi_crop": False,
    },
    {
        "id": "background",
        "label": "Doar background removal",
        "remove_background": True,
        "multi_crop": False,
    },
    {
        "id": "multicrop",
        "label": "Doar multi-crop",
        "remove_background": False,
        "multi_crop": True,
    },
    {
        "id": "multicrop_background",
        "label": "Multi-crop + background removal",
        "remove_background": True,
        "multi_crop": True,
    },
]


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prediction_margin(predictions: list[tuple[str, float]]) -> float:
    if len(predictions) < 2:
        return predictions[0][1] if predictions else 0.0
    return predictions[0][1] - predictions[1][1]


def safe_float(value: float) -> str:
    return f"{value:.6f}"


def predict_single(
    classifier: LoadedClassifier,
    image_path: Path,
    *,
    remove_background: bool,
    top_k: int,
    background_backend: str,
) -> list[tuple[str, float]]:
    image = load_model_image(
        image_path,
        remove_background=remove_background,
        backend=background_backend,
    )
    return classifier.predict_images([image], top_k=top_k)[0]


def run_predictions(
    *,
    classifier: LoadedClassifier,
    images: list[Path],
    source_dir: Path,
    top_k: int,
    max_crops: int,
    background_backend: str,
    detection_backend: str,
) -> tuple[list[dict], list[dict]]:
    prediction_rows: list[dict] = []
    image_summary_rows: list[dict] = []

    for image_index, image_path in enumerate(images, start=1):
        print(f"[{image_index}/{len(images)}] {image_path.name}", flush=True)
        per_image: list[dict] = []

        original_model_image = None
        background_model_image = None
        regions = None

        for config in CONFIGS:
            start_time = time.perf_counter()
            crop_count = 1
            aggregation_scores = ""

            if config["multi_crop"]:
                if regions is None:
                    regions = generate_crop_regions(
                        image_path,
                        max_crops=max_crops,
                        detection_backend=detection_backend,
                    )
                if config["remove_background"]:
                    if background_model_image is None:
                        background_model_image = load_model_image(
                            image_path,
                            remove_background=True,
                            backend=background_backend,
                        )
                    base_image = background_model_image
                else:
                    if original_model_image is None:
                        original_model_image = load_model_image(
                            image_path,
                            remove_background=False,
                            backend=background_backend,
                        )
                    base_image = original_model_image

                crops = [base_image.crop(region.bbox) for region in regions]
                crop_top_predictions = classifier.predict_images(crops, top_k=top_k)
                result = aggregate_crop_predictions(
                    classes=list(classifier.classes),
                    regions=regions,
                    crop_top_predictions=crop_top_predictions,
                    top_k=top_k,
                )
                predictions = result.final_predictions
                crop_count = len(result.crop_predictions)
                aggregation_scores = json.dumps(result.aggregation_scores, ensure_ascii=False)
            else:
                if config["remove_background"]:
                    if background_model_image is None:
                        background_model_image = load_model_image(
                            image_path,
                            remove_background=True,
                            backend=background_backend,
                        )
                    image = background_model_image
                else:
                    if original_model_image is None:
                        original_model_image = load_model_image(
                            image_path,
                            remove_background=False,
                            backend=background_backend,
                        )
                    image = original_model_image
                predictions = classifier.predict_images([image], top_k=top_k)[0]

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            top1_class, top1_confidence = predictions[0]
            top2_class, top2_confidence = predictions[1] if len(predictions) > 1 else ("", 0.0)
            top3_class, top3_confidence = predictions[2] if len(predictions) > 2 else ("", 0.0)

            row = {
                "image": str(image_path.relative_to(source_dir)),
                "config_id": config["id"],
                "config_label": config["label"],
                "remove_background": str(config["remove_background"]),
                "multi_crop": str(config["multi_crop"]),
                "crop_count": str(crop_count),
                "top1_class": top1_class,
                "top1_confidence": safe_float(top1_confidence),
                "top2_class": top2_class,
                "top2_confidence": safe_float(top2_confidence),
                "top3_class": top3_class,
                "top3_confidence": safe_float(top3_confidence),
                "margin_top1_top2": safe_float(prediction_margin(predictions)),
                "elapsed_ms": safe_float(elapsed_ms),
                "aggregation_scores": aggregation_scores,
            }
            prediction_rows.append(row)
            per_image.append(row)

        top_classes = [row["top1_class"] for row in per_image]
        class_counts = Counter(top_classes)
        majority_class, majority_count = class_counts.most_common(1)[0]
        best_row = max(per_image, key=lambda row: float(row["top1_confidence"]))
        average_confidence = statistics.mean(float(row["top1_confidence"]) for row in per_image)
        agreement_ratio = majority_count / len(CONFIGS)

        image_summary_rows.append(
            {
                "image": str(image_path.relative_to(source_dir)),
                "majority_class": majority_class,
                "majority_votes": str(majority_count),
                "agreement_ratio": safe_float(agreement_ratio),
                "all_configs_agree": str(majority_count == len(CONFIGS)),
                "best_config_by_confidence": best_row["config_label"],
                "best_config_id": best_row["config_id"],
                "best_prediction": best_row["top1_class"],
                "best_confidence": best_row["top1_confidence"],
                "average_confidence": safe_float(average_confidence),
                "simple_prediction": per_image[0]["top1_class"],
                "simple_confidence": per_image[0]["top1_confidence"],
                "background_prediction": per_image[1]["top1_class"],
                "background_confidence": per_image[1]["top1_confidence"],
                "multicrop_prediction": per_image[2]["top1_class"],
                "multicrop_confidence": per_image[2]["top1_confidence"],
                "multicrop_background_prediction": per_image[3]["top1_class"],
                "multicrop_background_confidence": per_image[3]["top1_confidence"],
            }
        )

    return prediction_rows, image_summary_rows


def summarize(prediction_rows: list[dict], image_summary_rows: list[dict]) -> dict:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        by_config[row["config_id"]].append(row)

    best_config_counts = Counter(row["best_config_id"] for row in image_summary_rows)
    summary_rows = []
    for config in CONFIGS:
        rows = by_config[config["id"]]
        confidences = [float(row["top1_confidence"]) for row in rows]
        margins = [float(row["margin_top1_top2"]) for row in rows]
        elapsed = [float(row["elapsed_ms"]) for row in rows]
        class_distribution = Counter(row["top1_class"] for row in rows)
        summary_rows.append(
            {
                "config_id": config["id"],
                "config_label": config["label"],
                "images": len(rows),
                "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
                "median_confidence": statistics.median(confidences) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "avg_margin": statistics.mean(margins) if margins else 0.0,
                "high_conf_70": sum(value >= 0.70 for value in confidences),
                "high_conf_85": sum(value >= 0.85 for value in confidences),
                "wins_by_confidence": best_config_counts[config["id"]],
                "avg_elapsed_ms": statistics.mean(elapsed) if elapsed else 0.0,
                "class_distribution": dict(class_distribution),
            }
        )

    agreement_counts = Counter(row["majority_votes"] for row in image_summary_rows)
    majority_distribution = Counter(row["majority_class"] for row in image_summary_rows)

    return {
        "summary_rows": summary_rows,
        "agreement_counts": dict(agreement_counts),
        "majority_distribution": dict(majority_distribution),
        "image_count": len(image_summary_rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    value_suffix: str = "",
    color: tuple[int, int, int] = (27, 107, 92),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = max(420, 120 + len(labels) * 70)
    margin_left = 290
    margin_right = 70
    margin_top = 80
    bar_height = 34
    gap = 34
    max_value = max(values) if values else 1
    if max_value <= 0:
        max_value = 1

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=26)
    font = ImageFont.load_default(size=18)
    font_small = ImageFont.load_default(size=16)

    draw.text((40, 28), title, fill=(18, 49, 43), font=font_title)
    chart_width = width - margin_left - margin_right

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + gap)
        bar_width = int(chart_width * (value / max_value))
        draw.text((40, y + 6), label, fill=(38, 59, 54), font=font)
        draw.rectangle((margin_left, y, margin_left + chart_width, y + bar_height), outline=(210, 224, 218))
        draw.rectangle((margin_left, y, margin_left + bar_width, y + bar_height), fill=color)
        value_text = f"{value:.2f}{value_suffix}" if isinstance(value, float) else f"{value}{value_suffix}"
        if value_suffix == " imagini":
            value_text = f"{int(value)} imagini"
        draw.text((margin_left + bar_width + 10, y + 7), value_text, fill=(18, 49, 43), font=font_small)

    image.save(path)


def draw_class_distribution(path: Path, distribution: dict[str, int]) -> None:
    items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    labels = [label.replace("_", " ") for label, _count in items]
    values = [count for _label, count in items]
    draw_bar_chart(
        path,
        "Majority class distribution on real images",
        labels,
        [float(value) for value in values],
        value_suffix=" imagini",
        color=(36, 63, 57),
    )


def write_html_report(
    path: Path,
    *,
    source_dir: Path,
    output_dir: Path,
    summary: dict,
    prediction_rows: list[dict],
    image_summary_rows: list[dict],
    chart_paths: dict[str, Path],
    background_backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = summary["summary_rows"]

    def fmt(value: float) -> str:
        return f"{value:.2%}"

    summary_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['config_label'])}</td>"
        f"<td>{row['images']}</td>"
        f"<td>{fmt(row['avg_confidence'])}</td>"
        f"<td>{fmt(row['median_confidence'])}</td>"
        f"<td>{fmt(row['avg_margin'])}</td>"
        f"<td>{row['high_conf_70']}</td>"
        f"<td>{row['high_conf_85']}</td>"
        f"<td>{row['wins_by_confidence']}</td>"
        f"<td>{row['avg_elapsed_ms']:.0f} ms</td>"
        "</tr>"
        for row in summary_rows
    )

    agreement_table = "\n".join(
        f"<tr><td>{votes} of 4 modes predicted the same class</td><td>{count}</td></tr>"
        for votes, count in sorted(summary["agreement_counts"].items(), reverse=True)
    )

    image_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['image'])}</td>"
        f"<td>{html.escape(row['majority_class'])}</td>"
        f"<td>{row['majority_votes']}/4</td>"
        f"<td>{html.escape(row['best_config_by_confidence'])}</td>"
        f"<td>{html.escape(row['best_prediction'])}</td>"
        f"<td>{float(row['best_confidence']):.2%}</td>"
        f"<td>{html.escape(row['simple_prediction'])} ({float(row['simple_confidence']):.1%})</td>"
        f"<td>{html.escape(row['background_prediction'])} ({float(row['background_confidence']):.1%})</td>"
        f"<td>{html.escape(row['multicrop_prediction'])} ({float(row['multicrop_confidence']):.1%})</td>"
        f"<td>{html.escape(row['multicrop_background_prediction'])} ({float(row['multicrop_background_confidence']):.1%})</td>"
        "</tr>"
        for row in image_summary_rows
    )

    best_overall = max(summary_rows, key=lambda row: (row["wins_by_confidence"], row["avg_confidence"]))
    chart_html = "\n".join(
        f'<figure><img src="{html.escape(chart.relative_to(output_dir).as_posix())}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name, chart in chart_paths.items()
    )

    path.write_text(
        f"""<!doctype html>
<html lang="ro">
<head>
  <meta charset="utf-8">
  <title>Real Photo Evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172b25; }}
    h1, h2 {{ color: #12312b; }}
    .note {{ border: 1px solid #8db8a8; background: #eef7f3; padding: 12px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 12px; }}
    th, td {{ border: 1px solid #cad8d2; padding: 7px; vertical-align: top; }}
    th {{ background: #d9f0e8; text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #cad8d2; }}
    figure {{ margin: 18px 0; page-break-inside: avoid; }}
    figcaption {{ font-size: 12px; color: #526b63; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Real Photo Evaluation</h1>
  <p><strong>Folder analizat:</strong> {html.escape(str(source_dir))}</p>
  <p><strong>Numar imagini:</strong> {summary['image_count']}</p>
  <p><strong>Backend background removal:</strong> {html.escape(background_backend)}</p>
  <div class="note">
    The folder does not contain real class labels, so this report does not calculate real accuracy, precision, recall or F1.
    The comparison below uses model confidence, the top1-top2 margin and the consistency between the four modes.
    The configuration that appears best according to these criteria is: <strong>{html.escape(best_overall['config_label'])}</strong>.
  </div>

  <h2>Summary by configuration</h2>
  <table>
    <thead>
      <tr>
        <th>Configuration</th><th>Images</th><th>Average confidence</th><th>Median confidence</th>
        <th>Average top1-top2 margin</th><th>Conf. >= 70%</th><th>Conf. >= 85%</th>
        <th>Confidence wins</th><th>Average time</th>
      </tr>
    </thead>
    <tbody>{summary_table}</tbody>
  </table>

  <h2>Consistency between modes</h2>
  <table><thead><tr><th>Agreement type</th><th>Number of images</th></tr></thead><tbody>{agreement_table}</tbody></table>

  <h2>Charts</h2>
  {chart_html}

  <h2>Results for each image</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th><th>Majority class</th><th>Agreement</th><th>Most confident mode</th>
        <th>Best prediction</th><th>Best confidence</th><th>Simple</th><th>Background</th>
        <th>Multi-crop</th><th>Multi-crop + background</th>
      </tr>
    </thead>
    <tbody>{image_table}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real photos in four prediction modes.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default=ROOT_DIR / "reports" / "real_photos_evaluation", type=Path)
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-crops", type=int, default=8)
    parser.add_argument("--background-backend", choices=["auto", "rembg", "opencv"], default="opencv")
    parser.add_argument("--detection-backend", choices=["auto", "rembg", "opencv"], default="opencv")
    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(source_dir)
    if not images:
        raise SystemExit(f"Nu am gasit imagini in {source_dir}")

    classifier = LoadedClassifier(args.checkpoint)
    prediction_rows, image_summary_rows = run_predictions(
        classifier=classifier,
        images=images,
        source_dir=source_dir,
        top_k=args.top_k,
        max_crops=args.max_crops,
        background_backend=args.background_backend,
        detection_backend=args.detection_backend,
    )
    summary = summarize(prediction_rows, image_summary_rows)

    write_csv(output_dir / "predictions_all_modes.csv", prediction_rows)
    write_csv(output_dir / "per_image_summary.csv", image_summary_rows)
    write_csv(
        output_dir / "summary_by_mode.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "class_distribution"},
                "class_distribution": json.dumps(row["class_distribution"], ensure_ascii=False),
            }
            for row in summary["summary_rows"]
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    charts_dir = output_dir / "charts"
    chart_paths = {
        "Average confidence by mode": charts_dir / "avg_confidence_by_mode.png",
        "Confidence wins": charts_dir / "wins_by_mode.png",
        "Agreement between the four modes": charts_dir / "agreement.png",
        "Majority class distribution": charts_dir / "majority_class_distribution.png",
    }
    draw_bar_chart(
        chart_paths["Average confidence by mode"],
        "Average confidence by configuration",
        [row["config_label"] for row in summary["summary_rows"]],
        [row["avg_confidence"] * 100 for row in summary["summary_rows"]],
        value_suffix="%",
    )
    draw_bar_chart(
        chart_paths["Confidence wins"],
        "How often each configuration has the highest confidence",
        [row["config_label"] for row in summary["summary_rows"]],
        [float(row["wins_by_confidence"]) for row in summary["summary_rows"]],
        value_suffix=" images",
        color=(25, 91, 132),
    )
    agreement_labels = [f"{votes}/4 modes predicted the same class" for votes in sorted(summary["agreement_counts"], reverse=True)]
    agreement_values = [float(summary["agreement_counts"][votes]) for votes in sorted(summary["agreement_counts"], reverse=True)]
    draw_bar_chart(
        chart_paths["Agreement between the four modes"],
        "Prediction consistency between modes",
        agreement_labels,
        agreement_values,
        value_suffix=" images",
        color=(112, 86, 36),
    )
    draw_class_distribution(chart_paths["Majority class distribution"], summary["majority_distribution"])

    write_html_report(
        output_dir / "report.html",
        source_dir=source_dir,
        output_dir=output_dir,
        summary=summary,
        prediction_rows=prediction_rows,
        image_summary_rows=image_summary_rows,
        chart_paths=chart_paths,
        background_backend=args.background_backend,
    )

    print(f"Report generated in: {output_dir}")
    print(f"Analyzed images: {len(images)}")
    best = max(summary["summary_rows"], key=lambda row: (row["wins_by_confidence"], row["avg_confidence"]))
    print(f"Recommended configuration by confidence/consistency: {best['config_label']}")


if __name__ == "__main__":
    main()
