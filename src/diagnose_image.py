from __future__ import annotations

import argparse
from pathlib import Path

from src.predict_image import LoadedClassifier
from src.symbolic_rules import format_rule, rule_for_class
from src.train_classifier import BEST_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prezice boala si aplica reguli simbolice pentru recomandari.")
    parser.add_argument("--image", required=True, help="Calea catre imagine.")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH), help="Checkpoint model.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--remove-background", action="store_true", help="Clasifica pe varianta fara background.")
    parser.add_argument("--single-crop", action="store_true", help="Dezactiveaza multi-crop voting.")
    parser.add_argument("--max-crops", type=int, default=8)
    args = parser.parse_args()

    classifier = LoadedClassifier(Path(args.checkpoint), remove_background=args.remove_background)
    if args.single_crop:
        predictions = classifier.predict(Path(args.image), top_k=args.top_k)
        crop_predictions = []
    else:
        result = classifier.predict_multi_crop(Path(args.image), top_k=args.top_k, max_crops=args.max_crops)
        predictions = result.final_predictions
        crop_predictions = result.crop_predictions

    best_class, best_probability = predictions[0]
    print(format_rule(best_class, best_probability, rule_for_class(best_class)))

    if crop_predictions:
        print("")
        print("Multi-crop voting:")
        print(f"- crop-uri generate: {len(crop_predictions)}")
        print("- vot: majoritar ponderat cu confidence")
        for crop in crop_predictions:
            print(
                f"- crop {crop.crop_index} [{crop.source}] -> "
                f"{crop.predicted_class}: {crop.confidence * 100:.2f}% "
                f"(weight={crop.weight:.2f})"
            )

    if len(predictions) > 1:
        print("")
        print("Alternative:")
        for class_name, probability in predictions[1:]:
            print(f"- {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
