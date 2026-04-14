from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from hydra import compose, initialize_config_module

import torch

from ecg_classifier.commands import _ensure_dataset_available, _ensure_splits_available
from ecg_classifier.training.eval import evaluate
from ecg_classifier.training.train import train
from ecg_classifier.utils.seed import seed_everything

from api.services.metrics import load_test_metrics
from api.core.settings import settings


def build_training_overrides(payload: dict[str, Any]) -> list[str]:
    overrides: list[str] = []

    model_name = str(payload["model_name"])
    split_name = str(payload.get("split_name", "default"))
    max_epochs = int(payload.get("max_epochs", 1))

    use_gpu = torch.cuda.is_available()

    overrides.append(f"model={model_name}")
    overrides.append(f"split.output_name={split_name}")
    overrides.append(f"train.max_epochs={max_epochs}")
    overrides.append(f"data.root_dir={settings.shared_dataset_dir}")
    overrides.append(f"mlflow.tracking_uri={settings.mlflow_tracking_uri}")
    overrides.append("data.num_workers=0")

    if use_gpu:
        overrides.append("train.accelerator=gpu")
        overrides.append("train.devices=1")
    else:
        overrides.append("train.accelerator=cpu")
        overrides.append("train.devices=1")
        overrides.append("train.precision=32")

    if payload.get("batch_size") is not None:
        overrides.append(f"model.batch_size={int(payload['batch_size'])}")

    if payload.get("img_size") is not None:
        overrides.append(f"data.image_size={int(payload['img_size'])}")

    if payload.get("learning_rate") is not None:
        overrides.append(f"model.learning_rate={payload['learning_rate']}")

    if payload.get("weight_decay") is not None:
        overrides.append(f"model.weight_decay={payload['weight_decay']}")

    if payload.get("pretrained") is not None:
        overrides.append(f"model.pretrained={str(payload['pretrained']).lower()}")

    if payload.get("timm_name"):
        overrides.append(f"model.timm_name={payload['timm_name']}")

    overrides.extend(payload.get("extra_overrides", []))
    return overrides


def compose_cfg(overrides: list[str]) -> Any:
    with initialize_config_module(
        config_module="ecg_classifier.conf",
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def build_model_key(model_name: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{now}_{uuid4().hex[:8]}"


def build_display_name(payload: dict[str, Any], cfg: Any) -> str:
    if payload.get("display_name"):
        return str(payload["display_name"])
    return (
        f"{cfg.model.name.upper()} | split={cfg.split.output_name} | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def extract_config_snapshot(cfg: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "image_size": int(cfg.data.image_size),
        "class_names": list(cfg.data.class_names),
        "batch_size": int(cfg.model.batch_size),
        "learning_rate": float(cfg.model.learning_rate),
        "weight_decay": float(cfg.model.weight_decay),
        "split_name": str(cfg.split.output_name),
        "max_epochs": int(cfg.train.max_epochs),
    }

    if str(cfg.model.name) == "vit":
        snapshot["timm_name"] = str(cfg.model.timm_name)
        snapshot["pretrained"] = bool(cfg.model.pretrained)

    return snapshot


def run_training_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    overrides = build_training_overrides(payload)
    cfg = compose_cfg(overrides)

    seed_everything(int(cfg.seed))

    data_root = Path(str(cfg.data.root_dir))
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_root}. "
            "Expected data-init to prepare the dataset in shared storage."
        )

    _ensure_splits_available(cfg)

    training_artifacts = train(cfg)
    checkpoint_path = Path(training_artifacts.checkpoint_path).resolve()

    metrics_path = evaluate(cfg=cfg, checkpoint_path=checkpoint_path)
    metrics = load_test_metrics(metrics_path)

    model_key = build_model_key(str(cfg.model.name))
    display_name = build_display_name(payload, cfg)
    config_snapshot = extract_config_snapshot(cfg)

    return {
        "model_key": model_key,
        "display_name": display_name,
        "model_name": str(cfg.model.name),
        "checkpoint_path": str(checkpoint_path),
        "split_name": str(cfg.split.output_name),
        "mlflow_run_id": training_artifacts.mlflow_run_id,
        "config_snapshot": config_snapshot,
        "metrics": metrics,
        "tags": dict(payload.get("tags", {})),
        "make_default": bool(payload.get("make_default", False)),
        "applied_overrides": overrides,
        "metrics_path": str(metrics_path),
    }