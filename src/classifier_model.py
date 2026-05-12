from __future__ import annotations

from torch import nn
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_model(model_name: str, num_classes: int, *, pretrained: bool) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Model necunoscut: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "mobilenet_v3_small":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if model_name == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Model necunoscut: {model_name}")


def build_transforms(*, img_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0)),
                        transforms.Compose(
                            [
                                transforms.Resize((img_size + 32, img_size + 32)),
                                transforms.CenterCrop(img_size),
                            ]
                        ),
                        transforms.Resize((img_size, img_size)),
                    ]
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
