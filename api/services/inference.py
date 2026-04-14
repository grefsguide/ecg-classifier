from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from ecg_classifier.models.cnn_classifier import SimpleCnn
from ecg_classifier.models.vit_classifier import create_vit


DEFAULT_CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_VIT_TIMM_NAME = "vit_base_patch16_224"


@dataclass
class InferenceResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_model(
    model_name: str,
    class_names: list[str],
    config_snapshot: dict[str, Any] | None = None,
) -> torch.nn.Module:
    config_snapshot = config_snapshot or {}
    num_classes = len(class_names)

    if model_name == "cnn":
        model = SimpleCnn(num_classes=num_classes)
    elif model_name == "vit":
        timm_name = str(config_snapshot.get("timm_name", DEFAULT_VIT_TIMM_NAME))
        model = create_vit(
            timm_name=timm_name,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_key = key[len("model.") :]
        else:
            cleaned_key = key
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    class_names: list[str],
    config_snapshot: dict[str, Any] | None = None,
) -> torch.nn.Module:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    device = get_device()
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    model = build_inference_model(
        model_name=model_name,
        class_names=class_names,
        config_snapshot=config_snapshot,
    )

    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def read_image_from_bytes(file_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(file_bytes))
    return image.convert("RGB")


def run_inference(
    *,
    file_bytes: bytes,
    checkpoint_path: str,
    model_name: str,
    class_names: list[str] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> InferenceResult:
    class_names = class_names or DEFAULT_CLASS_NAMES
    config_snapshot = config_snapshot or {}

    image_size = int(config_snapshot.get("image_size", DEFAULT_IMAGE_SIZE))
    transform = build_eval_transform(image_size=image_size)
    image = read_image_from_bytes(file_bytes)

    tensor = transform(image).unsqueeze(0)
    device = get_device()
    tensor = tensor.to(device)

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        class_names=class_names,
        config_snapshot=config_snapshot,
    )

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    top_index = int(torch.argmax(probs).item())
    probabilities = {
        class_name: float(probs[idx].item())
        for idx, class_name in enumerate(class_names)
    }

    return InferenceResult(
        predicted_class=class_names[top_index],
        confidence=float(probs[top_index].item()),
        probabilities=probabilities,
    )