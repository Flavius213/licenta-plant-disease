from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.predict_image import LoadedClassifier
from src.symbolic_rules import format_rule, rule_for_class
from src.train_classifier import BEST_MODEL_PATH


MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BEST_MODEL_PATH)))
DEFAULT_MAX_CROPS = int(os.getenv("MAX_CROPS", "8"))
DEFAULT_REMOVE_BACKGROUND = os.getenv("REMOVE_BACKGROUND", "true").lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="Plant Diagnosis API",
    description="CNN multi-crop voting + symbolic rules pentru diagnosticul bolilor la pomi.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: LoadedClassifier | None = None


def get_classifier() -> LoadedClassifier:
    global classifier
    if classifier is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Modelul nu exista: {MODEL_PATH}")
        classifier = LoadedClassifier(MODEL_PATH)
    return classifier


def rule_payload(class_name: str, confidence: float) -> dict[str, Any]:
    rule = rule_for_class(class_name)
    return {
        "class_name": class_name,
        "confidence": confidence,
        "rule": rule,
        "explanation_text": format_rule(class_name, confidence, rule),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/classes")
def classes() -> dict[str, Any]:
    model = get_classifier()
    return {"classes": list(model.classes)}


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    remove_background: bool = Query(DEFAULT_REMOVE_BACKGROUND),
    multi_crop: bool = Query(True),
    max_crops: int = Query(DEFAULT_MAX_CROPS, ge=1, le=16),
    top_k: int = Query(3, ge=1, le=10),
) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fisierul trimis nu este o imagine.")

    model = get_classifier()

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await image.read())

    try:
        if multi_crop:
            result = model.predict_multi_crop(
                temp_path,
                top_k=top_k,
                remove_background=remove_background,
                max_crops=max_crops,
                detection_backend="opencv",
            )
            predictions = result.final_predictions
            crops = [
                {
                    "crop_index": crop.crop_index,
                    "bbox": crop.bbox,
                    "source": crop.source,
                    "weight": crop.weight,
                    "predicted_class": crop.predicted_class,
                    "confidence": crop.confidence,
                    "top_predictions": crop.top_predictions,
                }
                for crop in result.crop_predictions
            ]
        else:
            predictions = model.predict(
                temp_path,
                top_k=top_k,
                remove_background=remove_background,
            )
            crops = []

        best_class, confidence = predictions[0]
        return {
            "prediction": rule_payload(best_class, confidence),
            "alternatives": [
                {"class_name": class_name, "confidence": probability}
                for class_name, probability in predictions[1:]
            ],
            "pipeline": {
                "preprocessing": "background_removal_checkerboard" if remove_background else "original_image",
                "region_detection": "foreground_components_and_grid",
                "crop_generation": "multi_crop" if multi_crop else "full_image",
                "classifier": "MobileNetV3 Small CNN",
                "aggregation": "weighted_majority_vote_by_confidence" if multi_crop else "single_prediction",
                "symbolic_ai": "rule_based_reasoning",
            },
            "crops": crops,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare diagnostic: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
