from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from ecg_classifier.data.datamodule import EcgDataModule
from ecg_classifier.models.lightning_module import EcgLightningModule
from ecg_classifier.models.cnn_classifier import SimpleCnn
from ecg_classifier.models.vit_classifier import create_vit
from ecg_classifier.utils.io_utils import save_json


def load_lightning_module(cfg, checkpoint_path: Path) -> EcgLightningModule:
    class_names = list(cfg.data.class_names)
    num_classes = len(class_names)

    if cfg.model.name == "cnn":
        neural_network = SimpleCnn(num_classes=num_classes)
    elif cfg.model.name == "vit":
        neural_network = create_vit(
            timm_name=cfg.model.timm_name,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unknown model.name: {cfg.model.name}")

    lightning_module = EcgLightningModule(
        model=neural_network,
        num_classes=num_classes,
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
    )

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    lightning_module.load_state_dict(checkpoint_state["state_dict"])
    return lightning_module


def evaluate(cfg, checkpoint_path: Path) -> Path:
    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.mlflow.tracking_uri),
        experiment_name=str(cfg.mlflow.experiment_name),
    )

    data_root = Path(cfg.data.root_dir)
    splits_dir = Path(cfg.data.splits_dir) / cfg.split.output_name

    datamodule = EcgDataModule(
        data_root=data_root,
        class_names=list(cfg.data.class_names),
        train_csv=splits_dir / "train.csv",
        val_csv=splits_dir / "val.csv",
        test_csv=splits_dir / "test.csv",
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.model.batch_size),
        num_workers=int(cfg.data.num_workers),
    )

    lightning_module = load_lightning_module(cfg=cfg, checkpoint_path=checkpoint_path)

    trainer = pl.Trainer(
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        logger=mlflow_logger,
    )

    test_results = trainer.test(model=lightning_module, datamodule=datamodule, verbose=False)
    metrics_path = Path(cfg.data.artifacts_dir) / "metrics" / cfg.model.name / "test_metrics.json"
    save_json(metrics_path, {"results": test_results})
    return metrics_path