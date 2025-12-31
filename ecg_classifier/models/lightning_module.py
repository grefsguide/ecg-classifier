from __future__ import annotations

import torch
import torch.nn.functional as functional
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import MulticlassAUROC, MulticlassPrecision, MulticlassRecall


class EcgLightningModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Train metrics (registered as modules -> moved to device automatically)
        self.train_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")

        # Val metrics
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")

        # Test metrics
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def _get_metrics(self, stage_name: str):
        if stage_name == "train":
            return self.train_recall, self.train_precision, self.train_auroc
        if stage_name == "val":
            return self.val_recall, self.val_precision, self.val_auroc
        if stage_name == "test":
            return self.test_recall, self.test_precision, self.test_auroc
        raise ValueError(f"Unknown stage_name: {stage_name}")

    def _shared_step(self, batch: dict, stage_name: str) -> torch.Tensor:
        images: torch.Tensor = batch["image"]
        targets_tensor = torch.as_tensor(batch["target"], device=self.device, dtype=torch.long)

        logits = self(images)
        loss = functional.cross_entropy(logits, targets_tensor)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        recall_metric, precision_metric, auroc_metric = self._get_metrics(stage_name)
        recall_metric.update(predictions, targets_tensor)
        precision_metric.update(predictions, targets_tensor)
        auroc_metric.update(probabilities, targets_tensor)

        self.log(f"{stage_name}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        return self._shared_step(batch=batch, stage_name="train")

    def validation_step(self, batch: dict, batch_index: int) -> None:
        self._shared_step(batch=batch, stage_name="val")

    def test_step(self, batch: dict, batch_index: int) -> None:
        self._shared_step(batch=batch, stage_name="test")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(stage_name="train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(stage_name="val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(stage_name="test")

    def _log_epoch_metrics(self, stage_name: str) -> None:
        recall_metric, precision_metric, auroc_metric = self._get_metrics(stage_name)

        recall_value = recall_metric.compute()
        precision_value = precision_metric.compute()
        auroc_value = auroc_metric.compute()

        self.log(f"{stage_name}/recall", recall_value, prog_bar=True)
        self.log(f"{stage_name}/precision", precision_value, prog_bar=False)
        self.log(f"{stage_name}/auroc", auroc_value, prog_bar=False)

        recall_metric.reset()
        precision_metric.reset()
        auroc_metric.reset()