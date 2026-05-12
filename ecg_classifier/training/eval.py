from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from ecg_classifier.data.datamodule import EcgDataModule
from ecg_classifier.models.lightning_module import EcgLightningModule
from ecg_classifier.models.cnn_classifier import SimpleCnn
from ecg_classifier.models.resnet_classifier import create_resnet
from ecg_classifier.models.vit_classifier import create_vit
from ecg_classifier.models.unet_transformer import UnetSeriesTransformer
from ecg_classifier.utils.io_utils import save_json

def _is_date_part(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _is_time_part(value: str) -> bool:
    try:
        datetime.strptime(value, "%H-%M-%S")
        return True
    except ValueError:
        return False

def _resolve_metrics_dir(cfg, checkpoint_path: Path) -> Path:
    artifacts_dir = Path(cfg.data.artifacts_dir)
    model_name = str(cfg.model.name)

    parent_name = checkpoint_path.parent.name
    grandparent_name = checkpoint_path.parent.parent.name

    # Поддержка обоих вариантов:
    # old:  artifacts/checkpoints/model/YYYY-MM-DD/model.ckpt
    # new:  artifacts/checkpoints/model/YYYY-MM-DD/HH-MM-SS/model.ckpt
    if _is_date_part(parent_name):
        metrics_dir = artifacts_dir / "metrics" / model_name / parent_name
    elif _is_time_part(parent_name) and _is_date_part(grandparent_name):
        metrics_dir = artifacts_dir / "metrics" / model_name / grandparent_name / parent_name
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        metrics_dir = artifacts_dir / "metrics" / model_name / date_str / time_str

    return metrics_dir

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
    elif cfg.model.name == "resnet":
        neural_network = create_resnet(
            backbone_name=str(cfg.model.backbone_name),
            num_classes=num_classes,
            pretrained=False,
        )
    elif cfg.model.name == "unet_transformer":
        neural_network = UnetSeriesTransformer(
            num_classes=num_classes,
            in_channels=3,
            num_signal_maps=int(cfg.model.num_signal_maps),
            seq_len=int(cfg.model.seq_len),
            unet_base_channels=int(cfg.model.unet_base_channels),
            transformer_d_model=int(cfg.model.transformer_d_model),
            transformer_nhead=int(cfg.model.transformer_nhead),
            transformer_num_layers=int(cfg.model.transformer_num_layers),
            transformer_ff_dim=int(cfg.model.transformer_ff_dim),
            dropout=float(cfg.model.dropout),
            softmax_temperature=float(cfg.model.softmax_temperature),
        )
    else:
        raise ValueError(f"Unknown model.name: {cfg.model.name}")

    lightning_module = EcgLightningModule(
        model=neural_network,
        num_classes=num_classes,
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
        ece_bins=int(cfg.model.ece_bins),
        log_train_prob_metrics=bool(cfg.model.log_train_prob_metrics),
        use_signal_supervision=bool(cfg.model.get("use_signal_supervision", False)),
        signal_loss_weight=float(cfg.model.get("signal_loss_weight", 0.2)),
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
        signal_length=int(cfg.data.get("signal_length", 5000)),
    )

    lightning_module = load_lightning_module(cfg=cfg, checkpoint_path=checkpoint_path)

    trainer = pl.Trainer(
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        logger=mlflow_logger,
    )

    test_results = trainer.test(model=lightning_module, datamodule=datamodule, verbose=False)

    metrics_dir = _resolve_metrics_dir(cfg=cfg, checkpoint_path=checkpoint_path)
    metrics_filename = f"{checkpoint_path.stem}_test_metrics.json"
    metrics_path = metrics_dir / metrics_filename

    save_json(
        metrics_path,
        {
            "model_name": str(cfg.model.name),
            "split_name": str(cfg.split.output_name),
            "checkpoint_path": str(checkpoint_path),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "results": test_results,
        }
    )

    return metrics_path