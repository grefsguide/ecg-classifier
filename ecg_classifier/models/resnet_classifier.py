from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)

_MODEL_BUILDERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}

_MODEL_WEIGHTS = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
    "resnet152": ResNet152_Weights,
    "resnext50_32x4d": ResNeXt50_32X4D_Weights,
    "resnext101_32x8d": ResNeXt101_32X8D_Weights,
    "wide_resnet50_2": Wide_ResNet50_2_Weights,
    "wide_resnet101_2": Wide_ResNet101_2_Weights,
}


def create_resnet(backbone_name: str, num_classes: int, pretrained: bool, weights_name: str | None = None) -> nn.Module:
    if backbone_name not in _MODEL_BUILDERS:
        available = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(
            f"Unknown backbone_name: {backbone_name}. Available: {available}"
        )

    builder = _MODEL_BUILDERS[backbone_name]
    weights_enum = _MODEL_WEIGHTS[backbone_name]

    if not pretrained:
        weights = None
    elif weights_name is None:
        weights = weights_enum.IMAGENET1K_V1
    else:
        weights = getattr(weights_enum, weights_name)

    model = builder(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model