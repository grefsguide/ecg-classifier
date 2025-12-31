from __future__ import annotations

import timm
from torch import nn


def create_vit(
    timm_name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model