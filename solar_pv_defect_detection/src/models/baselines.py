"""
Baseline CNN Models for Solar PV Defect Detection Benchmark.
Includes MobileNetV3-Small (standard edge benchmark) and ResNet-18.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_mobilenet_v3_small(num_classes=12, pretrained=True):
    """
    MobileNetV3-Small baseline (~2.5M parameters).
    Standard industrial benchmark for UAV edge inference.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_resnet18(num_classes=12, pretrained=True):
    """
    ResNet-18 baseline (~11.2M parameters).
    Standard deep residual CNN benchmark.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_baseline_model(architecture="mobilenet_v3_small", num_classes=12, pretrained=True):
    if architecture.lower() == "mobilenet_v3_small":
        return get_mobilenet_v3_small(num_classes=num_classes, pretrained=pretrained)
    elif architecture.lower() in ["resnet18", "resnet-18"]:
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown baseline architecture: {architecture}")
