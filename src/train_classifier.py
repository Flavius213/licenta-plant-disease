from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.classifier_model import build_transforms, create_model, freeze_backbone
from src.config import BATCH_SIZE, EPOCHS, FINAL_DIR, IMG_SIZE, METADATA_DIR, MODELS_DIR


CLASSIFIER_DIR = MODELS_DIR / "classifier"
BEST_MODEL_PATH = CLASSIFIER_DIR / "best_model.pt"
LAST_MODEL_PATH = CLASSIFIER_DIR / "last_model.pt"
HISTORY_PATH = CLASSIFIER_DIR / "training_history.csv"
TEST_METRICS_PATH = CLASSIFIER_DIR / "test_metrics.json"
CLASSIFICATION_REPORT_PATH = CLASSIFIER_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = CLASSIFIER_DIR / "confusion_matrix.csv"
CLASS_INDEX_PATH = METADATA_DIR / "class_to_idx.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ImageFolder, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if max_batches and batch_index >= max_batches:
            break

    return total_loss / max(total, 1), correct / max(total, 1)


def class_weights_for_dataset(dataset: ImageFolder) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes))
    total = counts.sum()
    weights = total / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def set_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            all_true.extend(labels.tolist())
            all_pred.extend(predictions)

    return all_true, all_pred


def write_history(history: list[dict[str, float | int]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"],
        )
        writer.writeheader()
        writer.writerows(history)


def write_classification_report(report: dict[str, dict[str, float] | float]) -> None:
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({"label": label, **metrics})
        else:
            rows.append({"label": label, "value": metrics})

    with CLASSIFICATION_REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["label", "precision", "recall", "f1-score", "support", "value"]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(matrix: np.ndarray, classes: list[str]) -> None:
    with CONFUSION_MATRIX_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *classes])
        for class_name, values in zip(classes, matrix):
            writer.writerow([class_name, *values.tolist()])


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    model_name: str,
    classes: list[str],
    class_to_idx: dict[str, int],
    img_size: int,
    epoch: int,
    val_acc: float,
    pretrained: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {index: class_name for class_name, index in class_to_idx.items()},
        "img_size": img_size,
        "epoch": epoch,
        "val_acc": val_acc,
        "pretrained": pretrained,
    }
    torch.save(checkpoint, path)


def train_classifier(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_dataset = ImageFolder(train_dir, transform=build_transforms(img_size=args.img_size, train=True))
    val_dataset = ImageFolder(val_dir, transform=build_transforms(img_size=args.img_size, train=False))
    test_dataset = ImageFolder(test_dir, transform=build_transforms(img_size=args.img_size, train=False))

    if train_dataset.class_to_idx != val_dataset.class_to_idx or train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Clasele din train/val/test nu sunt aliniate.")

    classes = train_dataset.classes
    CLASS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_INDEX_PATH.write_text(json.dumps(train_dataset.class_to_idx, indent=2), encoding="utf-8")

    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, workers=args.workers)
    test_loader = make_loader(test_dataset, batch_size=args.batch_size, shuffle=False, workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = create_model(args.model, len(classes), pretrained=not args.no_pretrained)

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"Checkpointul este pentru {checkpoint['model_name']}, dar ai cerut {args.model}."
            )
        if checkpoint["classes"] != classes:
            raise ValueError("Clasele din checkpoint nu coincid cu clasele datasetului curent.")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[INFO] Resume checkpoint: {resume_path}")

    staged_fine_tuning = args.freeze_epochs > 0 and not args.freeze_backbone
    if args.freeze_backbone or staged_fine_tuning:
        freeze_backbone(model, args.model)

    model = model.to(device)
    class_weights = class_weights_for_dataset(train_dataset).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Clase: {classes}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(
        f"[INFO] Model: {args.model}, pretrained={not args.no_pretrained}, "
        f"freeze_backbone={args.freeze_backbone}, freeze_epochs={args.freeze_epochs}"
    )
    if args.class_weights:
        print("[INFO] Class weights active")

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        if staged_fine_tuning and epoch == args.freeze_epochs + 1:
            set_trainable(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.fine_tune_learning_rate,
                weight_decay=args.weight_decay,
            )
            print(f"[INFO] Backbone deblocat pentru fine-tuning la lr={args.fine_tune_learning_rate}")

        start_time = time.time()
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.limit_batches,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            max_batches=args.limit_batches,
        )
        seconds = round(time.time() - start_time, 2)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "seconds": seconds,
            }
        )

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={seconds}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                model_name=args.model,
                classes=classes,
                class_to_idx=train_dataset.class_to_idx,
                img_size=args.img_size,
                epoch=epoch,
                val_acc=val_acc,
                pretrained=not args.no_pretrained,
            )
            print(f"[OK] Best model salvat: {BEST_MODEL_PATH}")

        if args.patience and epoch - int(history[np.argmax([item["val_acc"] for item in history])]["epoch"]) >= args.patience:
            print(f"[INFO] Early stopping dupa {args.patience} epoci fara imbunatatire.")
            break

    save_checkpoint(
        path=LAST_MODEL_PATH,
        model=model,
        model_name=args.model,
        classes=classes,
        class_to_idx=train_dataset.class_to_idx,
        img_size=args.img_size,
        epoch=args.epochs,
        val_acc=history[-1]["val_acc"],
        pretrained=not args.no_pretrained,
    )
    write_history(history)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    true_labels, predicted_labels = collect_predictions(model, test_loader, device)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(classes))))
    test_acc = float(np.mean(np.array(true_labels) == np.array(predicted_labels)))

    write_classification_report(report)
    write_confusion_matrix(matrix, classes)
    TEST_METRICS_PATH.write_text(
        json.dumps(
            {
                "test_acc": test_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": checkpoint["epoch"],
                "classes": classes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Test accuracy: {test_acc:.4f}")
    print(f"[OK] Istoric: {HISTORY_PATH}")
    print(f"[OK] Raport test: {CLASSIFICATION_REPORT_PATH}")
    print(f"[OK] Confusion matrix: {CONFUSION_MATRIX_PATH}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Antreneaza primul classifier pentru bolile pomilor.")
    parser.add_argument("--data-dir", default=str(FINAL_DIR), help="Folder cu train/val/test.")
    parser.add_argument("--model", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="Numar thread-uri CPU pentru torch. 0 = implicit.")
    parser.add_argument("--cpu", action="store_true", help="Forteaza rularea pe CPU.")
    parser.add_argument("--no-pretrained", action="store_true", help="Nu foloseste greutati ImageNet.")
    parser.add_argument("--resume-checkpoint", help="Continua fine-tuning-ul dintr-un checkpoint existent.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Antreneaza doar capul de clasificare.")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Epoci initiale cu backbone inghetat, apoi fine-tuning.")
    parser.add_argument("--class-weights", action="store_true", help="Compenseaza clasele dezechilibrate.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping dupa N epoci fara imbunatatire. 0 = dezactivat.")
    parser.add_argument("--limit-batches", type=int, help="Ruleaza doar cateva batch-uri; util pentru smoke test.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_classifier(args)


if __name__ == "__main__":
    main()
