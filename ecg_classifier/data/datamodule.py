from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from ecg_classifier.data.dataset import EcgImageDataset


class EcgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        class_names: list[str],
        train_csv: Path,
        val_csv: Path,
        test_csv: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.class_names = class_names
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = EcgImageDataset(
            data_root=self.data_root,
            split_csv=self.train_csv,
            class_names=self.class_names,
            transform=self.train_transform,
        )
        self.val_dataset = EcgImageDataset(
            data_root=self.data_root,
            split_csv=self.val_csv,
            class_names=self.class_names,
            transform=self.eval_transform,
        )
        self.test_dataset = EcgImageDataset(
            data_root=self.data_root,
            split_csv=self.test_csv,
            class_names=self.class_names,
            transform=self.eval_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )