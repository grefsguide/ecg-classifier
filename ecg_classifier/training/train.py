from __future__ import annotations

from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from ecg_classifier.data.datamodule import EcgDataModule
from ecg_classifier.models.lightning_module import EcgLightningModule
from ecg_classifier.models.cnn_classifier import SimpleCnn
from ecg_classifier.models.vit_classifier import create_vit
from ecg_classifier.utils.git_info import get_git_commit_id
from ecg_classifier.utils.io_utils import ensure_dir

def build_model(cfg) -> pl.LightningModule:
    class_names = list(cfg.data.class_names)
    num_classes = len(class_names)

    if cfg.model.name == "cnn":
        neural_network = SimpleCnn(num_classes=num_classes)
    elif cfg.model.name == "vit":
        neural_network = create_vit(
            timm_name=cfg.model.timm_name,
            num_classes=num_classes,
            pretrained=bool(cfg.model.pretrained),
        )
    else:
        raise ValueError(f"Unknown model.name: {cfg.model.name}")

    lightning_module = EcgLightningModule(
        model=neural_network,
        num_classes=num_classes,
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
    )
    return lightning_module

def train(cfg) -> Path:
    torch.set_float32_matmul_precision("medium")

    artifacts_dir = Path(cfg.data.artifacts_dir)
    date_str = datetime.now().strftime("%Y-%m-%d")
    checkpoints_dir = artifacts_dir / "checkpoints" / str(cfg.model.name) / date_str
    ensure_dir(checkpoints_dir)

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.mlflow.tracking_uri),
        experiment_name=str(cfg.mlflow.experiment_name),
    )

    git_commit_id = get_git_commit_id()
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit_id", git_commit_id)

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

    lightning_module = build_model(cfg=cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename=str(cfg.model.name),
        monitor="val/recall",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.logger.log_hyperparams(
        {
            "seed": int(cfg.seed),
            "model_name": str(cfg.model.name),
            "learning_rate": float(cfg.model.learning_rate),
            "weight_decay": float(cfg.model.weight_decay),
            "batch_size": int(cfg.model.batch_size),
            "image_size": int(cfg.data.image_size),
            "split_name": str(cfg.split.output_name),
            "git_commit_id": git_commit_id,
        }
    )

    trainer.fit(model=lightning_module, datamodule=datamodule)
    return Path(checkpoint_callback.best_model_path)