from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.background_removal import load_model_image
from src.classifier_model import build_transforms, create_model
from src.multicrop_voting import (
    MultiCropResult,
    aggregate_crop_predictions,
    crop_images_for_regions,
    generate_crop_regions,
)
from src.train_classifier import BEST_MODEL_PATH


class LoadedClassifier:
    def __init__(self, checkpoint_path: Path, *, remove_background: bool = False):
        self.checkpoint_path = checkpoint_path
        self.remove_background = remove_background
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.classes = self.checkpoint["classes"]
        self.model = create_model(self.checkpoint["model_name"], len(self.classes), pretrained=False)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.transform = build_transforms(img_size=self.checkpoint["img_size"], train=False)

    def predict_images(self, images, *, top_k: int = 3) -> list[list[tuple[str, float]]]:
        tensors = [self.transform(image).unsqueeze(0) for image in images]
        image_tensor = torch.cat(tensors, dim=0)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(image_tensor), dim=1)
            values, indices = torch.topk(probabilities, k=min(top_k, len(self.classes)), dim=1)

        batch_predictions: list[list[tuple[str, float]]] = []
        for row_values, row_indices in zip(values, indices):
            batch_predictions.append(
                [
                    (self.classes[index.item()], row_values[position].item())
                    for position, index in enumerate(row_indices)
                ]
            )
        return batch_predictions

    def predict(
        self,
        image_path: Path,
        *,
        top_k: int = 3,
        remove_background: bool | None = None,
    ) -> list[tuple[str, float]]:
        use_background_removal = self.remove_background if remove_background is None else remove_background
        image = load_model_image(image_path, remove_background=use_background_removal)
        return self.predict_images([image], top_k=top_k)[0]

    def predict_multi_crop(
        self,
        image_path: Path,
        *,
        top_k: int = 3,
        remove_background: bool | None = None,
        max_crops: int = 8,
        detection_backend: str = "opencv",
        background_backend: str = "auto",
    ) -> MultiCropResult:
        use_background_removal = self.remove_background if remove_background is None else remove_background
        regions = generate_crop_regions(
            image_path,
            max_crops=max_crops,
            detection_backend=detection_backend,
        )
        crops = crop_images_for_regions(
            image_path,
            regions,
            remove_background=use_background_removal,
            background_backend=background_backend,
        )
        crop_top_predictions = self.predict_images(crops, top_k=top_k)

        return aggregate_crop_predictions(
            classes=list(self.classes),
            regions=regions,
            crop_top_predictions=crop_top_predictions,
            top_k=top_k,
        )


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    *,
    top_k: int,
    remove_background: bool = False,
    multi_crop: bool = True,
    max_crops: int = 8,
    detection_backend: str = "opencv",
) -> list[tuple[str, float]]:
    classifier = LoadedClassifier(checkpoint_path, remove_background=remove_background)
    if multi_crop:
        return classifier.predict_multi_crop(
            image_path,
            top_k=top_k,
            max_crops=max_crops,
            detection_backend=detection_backend,
        ).final_predictions
    return classifier.predict(image_path, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the class of an image with the trained classifier.")
    parser.add_argument("--image", required=True, help="Path to the image.")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH), help="Checkpoint model.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--remove-background", action="store_true", help="Remove the background before prediction.")
    parser.add_argument("--single-crop", action="store_true", help="Run the classic prediction on the full image.")
    parser.add_argument("--max-crops", type=int, default=8)
    parser.add_argument("--detection-backend", choices=["auto", "rembg", "opencv"], default="opencv")
    parser.add_argument("--show-crops", action="store_true", help="Show the prediction for each crop.")
    args = parser.parse_args()

    classifier = LoadedClassifier(Path(args.checkpoint), remove_background=args.remove_background)
    if args.single_crop:
        predictions = classifier.predict(Path(args.image), top_k=args.top_k)
    else:
        result = classifier.predict_multi_crop(
            Path(args.image),
            top_k=args.top_k,
            max_crops=args.max_crops,
            detection_backend=args.detection_backend,
        )
        predictions = result.final_predictions
        if args.show_crops:
            print(f"Generated crops: {len(result.crop_predictions)}")
            for crop in result.crop_predictions:
                print(
                    f"crop {crop.crop_index} [{crop.source}] "
                    f"{crop.predicted_class}: {crop.confidence:.4f} weight={crop.weight:.3f}"
                )

    for class_name, probability in predictions:
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
