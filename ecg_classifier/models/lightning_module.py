import torch
import torch.nn.functional as functional
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

class MulticlassBrierScore(Metric):
    higher_is_better = False
    is_differentiable = False
    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.add_state(
            "sum_squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim != 2 or preds.size(1) != self.num_classes:
            raise ValueError(
                f"Expected preds shape [N, {self.num_classes}], got {tuple(preds.shape)}"
            )

        target = target.view(-1)
        target_one_hot = functional.one_hot(
            target,
            num_classes=self.num_classes,
        ).to(dtype=preds.dtype)

        squared_error = torch.sum((preds - target_one_hot) ** 2, dim=1)
        self.sum_squared_error += squared_error.sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_squared_error / self.total.clamp_min(1)

class EcgLightningModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        ece_bins: int = 15,
        log_train_prob_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ece_bins = ece_bins
        self.log_train_prob_metrics = log_train_prob_metrics

        self.train_metrics = self._build_metrics(
            include_probability_metrics=log_train_prob_metrics
        )
        self.val_metrics = self._build_metrics(include_probability_metrics=True)
        self.test_metrics = self._build_metrics(include_probability_metrics=True)

    def _build_metrics(self, include_probability_metrics: bool) -> nn.ModuleDict:
        metrics = nn.ModuleDict(
            {
                "recall": MulticlassRecall(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "precision": MulticlassPrecision(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "f1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",
                ),
            }
        )

        if include_probability_metrics:
            metrics["auroc"] = MulticlassAUROC(
                num_classes=self.num_classes,
                average="macro",
            )
            metrics["pr_auc"] = MulticlassAveragePrecision(
                num_classes=self.num_classes,
                average="macro",
            )
            metrics["ece"] = MulticlassCalibrationError(
                num_classes=self.num_classes,
                n_bins=self.ece_bins,
                norm="l1",
            )
            metrics["brier"] = MulticlassBrierScore(
                num_classes=self.num_classes,
            )

        return metrics

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def _get_metrics(self, stage_name: str) -> nn.ModuleDict:
        if stage_name == "train":
            return self.train_metrics
        if stage_name == "val":
            return self.val_metrics
        if stage_name == "test":
            return self.test_metrics
        raise ValueError(f"Unknown stage_name: {stage_name}")

    def _shared_step(self, batch: dict, stage_name: str) -> torch.Tensor:
        images: torch.Tensor = batch["image"]
        targets_tensor = torch.as_tensor(
            batch["target"],
            device=self.device,
            dtype=torch.long,
        )

        logits = self(images)
        loss = functional.cross_entropy(logits, targets_tensor)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        metrics = self._get_metrics(stage_name)
        for metric_name, metric in metrics.items():
            if metric_name in {"auroc", "pr_auc", "ece", "brier"}:
                metric.update(probabilities, targets_tensor)
            else:
                metric.update(predictions, targets_tensor)

        self.log(
            f"{stage_name}/loss",
            loss,
            prog_bar=(stage_name != "test"),
            on_step=False,
            on_epoch=True,
            batch_size=targets_tensor.size(0),
            sync_dist=True,
        )
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
        metrics = self._get_metrics(stage_name)

        for metric_name, metric in metrics.items():
            metric_value = metric.compute()
            self.log(
                f"{stage_name}/{metric_name}",
                metric_value,
                prog_bar=metric_name in {"recall", "f1"},
                sync_dist=True,
            )
            metric.reset()