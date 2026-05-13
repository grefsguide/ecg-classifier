from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from api.observability.metrics import (
    MODEL_CACHE_ENTRIES,
    MODEL_CACHE_HITS_TOTAL,
    MODEL_CACHE_MISSES_TOTAL,
    observe_inference_forward,
    observe_model_load,
)
from ecg_classifier.models.cnn_classifier import SimpleCnn
from ecg_classifier.models.resnet_classifier import create_resnet
from ecg_classifier.models.vit_classifier import create_vit
from api.services.artifact_storage import resolve_artifact_uri
from ecg_classifier.models.unet_transformer import UnetSeriesTransformer

DEFAULT_CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_VIT_TIMM_NAME = "vit_base_patch16_224"

_MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}
_MODEL_CACHE_LOCK = Lock()


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
    elif model_name == "resnet":
        backbone_name = str(config_snapshot.get("backbone_name", "resnet18"))
        model = create_resnet(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=False,
        )
    elif model_name == "unet_transformer":
        model = UnetSeriesTransformer(
            num_classes=num_classes,
            in_channels=3,
            num_signal_maps=int(config_snapshot.get("num_signal_maps", 8)),
            seq_len=int(config_snapshot.get("seq_len", 256)),
            unet_base_channels=int(config_snapshot.get("unet_base_channels", 32)),
            transformer_d_model=int(config_snapshot.get("transformer_d_model", 128)),
            transformer_nhead=int(config_snapshot.get("transformer_nhead", 8)),
            transformer_num_layers=int(config_snapshot.get("transformer_num_layers", 4)),
            transformer_ff_dim=int(config_snapshot.get("transformer_ff_dim", 256)),
            dropout=float(config_snapshot.get("dropout", 0.1)),
            softmax_temperature=float(config_snapshot.get("softmax_temperature", 10.0)),
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
        cleaned_key = key[len("model.") :] if key.startswith("model.") else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    class_names: list[str],
    config_snapshot: dict[str, Any] | None = None,
) -> torch.nn.Module:
    checkpoint_file = resolve_artifact_uri(checkpoint_path)
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


def get_or_load_model(
    *,
    checkpoint_path: str | Path,
    model_name: str,
    model_key: str,
    class_names: list[str],
    config_snapshot: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, bool, float]:
    device = get_device().type
    cache_key = (model_key, str(checkpoint_path), device)

    started = perf_counter()
    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)

    if cached_model is not None:
        MODEL_CACHE_HITS_TOTAL.labels(
            model_name=model_name,
            model_key=model_key,
            device=device,
        ).inc()
        observe_model_load(
            model_name=model_name,
            model_key=model_key,
            device=device,
            cache_hit=True,
            seconds=perf_counter() - started,
        )
        return cached_model, True, perf_counter() - started

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        class_names=class_names,
        config_snapshot=config_snapshot,
    )

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = model
        MODEL_CACHE_ENTRIES.set(len(_MODEL_CACHE))

    MODEL_CACHE_MISSES_TOTAL.labels(
        model_name=model_name,
        model_key=model_key,
        device=device,
    ).inc()

    load_seconds = perf_counter() - started
    observe_model_load(
        model_name=model_name,
        model_key=model_key,
        device=device,
        cache_hit=False,
        seconds=load_seconds,
    )
    return model, False, load_seconds


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
    model_key: str,
    class_names: list[str] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    source: str = "api",
) -> InferenceResult:
    class_names = class_names or DEFAULT_CLASS_NAMES
    config_snapshot = config_snapshot or {}

    image_size = int(config_snapshot.get("image_size", DEFAULT_IMAGE_SIZE))
    transform = build_eval_transform(image_size=image_size)

    image = read_image_from_bytes(file_bytes)
    tensor = transform(image).unsqueeze(0)

    device = get_device()
    tensor = tensor.to(device)

    model, _cache_hit, _load_seconds = get_or_load_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        model_key=model_key,
        class_names=class_names,
        config_snapshot=config_snapshot,
    )

    with torch.inference_mode():
        forward_started = perf_counter()
        logits = model(tensor)
        forward_seconds = perf_counter() - forward_started

        observe_inference_forward(
            model_name=model_name,
            model_key=model_key,
            device=device.type,
            source=source,
            status="success",
            seconds=forward_seconds,
        )

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